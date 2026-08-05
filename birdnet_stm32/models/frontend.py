"""AudioFrontendLayer: in-model audio feature extraction for the STM32N6 NPU.

This Keras layer implements three interchangeable frontend modes that produce
a fixed-size mel-like spectrogram [B, mel_bins, spec_width, 1] from different
input representations:

- **precomputed**: Pass-through for offline mel spectrograms.
- **hybrid**: Linear STFT magnitude -> 1x1 Conv2D mel mixer.
- **raw**: Raw waveform -> learned Gabor quadrature filterbank -> L1 magnitude.

Design constraints, in priority order:

1. **Cover the signal.** The raw filterbank uses a kernel at least as long as
   its hop, so every input sample reaches at least one output frame, and the
   analysis window is long enough to resolve bird harmonics.
2. **Represent magnitude, not phase.** A single real-valued bandpass filter
   oscillates with the carrier; sampling it at the hop rate aliases. The raw
   path therefore learns a quadrature (cosine/sine) pair and combines them as
   ``|re| + |im|`` — an L1 magnitude that is exact up to a per-band constant
   and, unlike ``sqrt(re^2 + im^2)``, keeps its dynamic range INT8-friendly.
3. **Stay on the NPU.** No global reductions and no data-dependent scaling:
   every op here is a convolution, an elementwise op, or a layout change.
   Per-band calibration is learned (BatchNorm + trainable magnitude scaling)
   rather than computed per sample at inference time.

The caller is expected to hand over a peak-normalized waveform (the training,
evaluation, calibration and firmware paths all do this), which is what makes
fixed learned gains valid in place of a per-sample normalization.
"""

from typing import NamedTuple

import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import constraints, layers

VALID_FRONTENDS = ("librosa", "hybrid", "raw", "mfcc", "log_mel")

# Overlap factor of the raw analysis window: window = RAW_OVERLAP * hop.
RAW_OVERLAP = 2
# Hops are rounded to this so the derived fold (hop // 2) is a multiple of 8.
_HOP_ALIGN = 16
# Width of the per-band temporal lowpass applied after the modulus.
SMOOTH_TAPS = 5


def _hann_depthwise_init(shape, dtype=None):
    """Initialize a depthwise temporal kernel to a normalized Hann window."""
    taps = int(shape[1])
    w = np.hanning(taps + 2)[1:-1].astype(np.float32)
    w /= w.sum()
    kernel = np.zeros(tuple(int(s) for s in shape), dtype=np.float32)
    kernel[0, :, :, 0] = w[:, None]
    return tf.convert_to_tensor(kernel, dtype=dtype or tf.float32)


def normalize_frontend_name(name: str) -> str:
    """Validate a frontend name.

    Args:
        name: Frontend name.

    Returns:
        The frontend name, unchanged.

    Raises:
        ValueError: If name is not a valid frontend.
    """
    if name in VALID_FRONTENDS:
        return name
    raise ValueError(f"Invalid audio frontend: '{name}'. Valid options: {VALID_FRONTENDS}")


def hybrid_fft_bins(fft_length: int) -> int:
    """Return the number of linear STFT bins the hybrid frontend consumes.

    The Nyquist bin is dropped so the count is ``fft_length // 2`` — a multiple
    of 8 for every sane FFT size, which lets the mel mixer run at full NPU
    channel utilization without a runtime zero-pad (a FILL + CONCATENATION pair
    on every inference). The discarded bin carries no bird signal.

    Args:
        fft_length: FFT size.

    Returns:
        Number of spectrogram rows expected as model input.
    """
    return int(fft_length) // 2


class RawGeometry(NamedTuple):
    """Layout of the folded raw filterbank.

    Attributes:
        hop: Distance between output frames, in input samples.
        window: Analysis window length, in input samples.
        fold: Interleaved channels the waveform is reshaped into.
        kernel: Convolution kernel width, in folded frames.
        stride: Convolution stride, in folded frames (always 2).
        crop: Input samples actually consumed (a whole number of folds).
        frames: Frames the convolution emits before slicing to spec_width.
    """

    hop: int
    window: int
    fold: int
    kernel: int
    stride: int
    crop: int
    frames: int


def raw_filterbank_geometry(
    num_samples: int,
    spec_width: int,
    overlap: int = RAW_OVERLAP,
) -> RawGeometry:
    """Lay out the raw filterbank so its convolution runs on the NPU.

    A learned STFT wants a long kernel and a hop of a few hundred samples. Fed
    to the N6 directly that is a large-stride convolution, which the compiler
    hands to the Cortex-M55 — measured, and the reason the previous frontend's
    filterbank never ran on the NPU at all.

    The fix is a polyphase view: reinterpret the waveform as ``hop / 2``
    interleaved channels (free — NHWC memory is contiguous, so ``[T, 1]`` and
    ``[T/fold, fold]`` are the same bytes) and convolve with **stride 2**. The
    arithmetic is identical to the long strided kernel, but every convolution
    now sits inside the NPU's supported stride range and sees a full channel
    group instead of a single input channel.

    Args:
        num_samples: Waveform length in samples.
        spec_width: Required number of output frames.
        overlap: Window length as a multiple of the hop.

    Returns:
        The resolved :class:`RawGeometry`.

    Raises:
        ValueError: If no hop yields ``spec_width`` frames for this chunk.
    """
    if spec_width < 2:
        raise ValueError(f"spec_width must be >= 2, got {spec_width}")

    # frames >= spec_width  <=>  hop <= num_samples / (spec_width - 1 + overlap)
    hop = (num_samples // (spec_width - 1 + overlap) // _HOP_ALIGN) * _HOP_ALIGN
    if hop < 2 * _HOP_ALIGN:
        raise ValueError(
            f"Cannot fit {spec_width} frames into {num_samples} samples. "
            f"Lower --spec_width or lengthen --chunk_duration."
        )

    fold = hop // 2
    window = overlap * hop
    kernel = 2 * overlap  # window // fold
    crop = (num_samples // fold) * fold
    frames = (crop // fold - kernel) // 2 + 1
    return RawGeometry(hop, window, fold, kernel, 2, crop, frames)


def gabor_filterbank(
    mel_bins: int,
    kernel: int,
    sample_rate: int,
    fmin: float,
    fmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build mel-spaced Gabor filters as a cosine/sine quadrature pair.

    Each band is a Gaussian-windowed complex exponential centred on a mel
    frequency, with its time-domain width set from the spacing to its
    neighbours so the bank's frequency response approximates a mel filterbank.
    Starting from this initialization, an untrained frontend already produces a
    mel-like spectrogram; training only refines it.

    Args:
        mel_bins: Number of filters (output channels).
        kernel: Filter length in samples.
        sample_rate: Sampling rate (Hz).
        fmin: Lowest band centre (Hz).
        fmax: Highest band centre (Hz).

    Returns:
        Tuple of ``(real, imag)`` arrays, each ``[mel_bins, kernel]`` float32,
        normalized to unit energy per band.
    """
    edges = librosa.mel_frequencies(n_mels=int(mel_bins) + 2, fmin=float(fmin), fmax=float(fmax), htk=False)
    centers = edges[1:-1].astype(np.float64)
    # Half the distance between neighbouring centres is the target bandwidth.
    bandwidths = np.maximum((edges[2:] - edges[:-2]) / 2.0, 1.0).astype(np.float64)

    n = np.arange(kernel, dtype=np.float64) - (kernel - 1) / 2.0
    # Gaussian envelope whose spectral width matches the target bandwidth,
    # clamped so the window stays inside the kernel.
    sigma = np.minimum(sample_rate / (2.0 * np.pi * bandwidths), kernel / 6.0)

    envelope = np.exp(-0.5 * (n[None, :] / sigma[:, None]) ** 2)
    phase = 2.0 * np.pi * centers[:, None] * n[None, :] / float(sample_rate)
    real = envelope * np.cos(phase)
    imag = envelope * np.sin(phase)

    # Unit energy per band so every filter starts on a comparable scale and the
    # INT8 activation range is not dominated by a few loud bands.
    norm = np.sqrt((real**2 + imag**2).sum(axis=1, keepdims=True)) + 1e-12
    return (real / norm).astype(np.float32), (imag / norm).astype(np.float32)


from birdnet_stm32.models.magnitude import MagnitudeScalingLayer  # noqa: E402


class AudioFrontendLayer(layers.Layer):
    """Audio frontend with interchangeable input modes and magnitude scaling.

    Modes:
        precomputed: Mel spectrogram [B, mel_bins, T, 1] -> slice to spec_width.
        hybrid: Linear STFT bins [B, fft_bins, T, 1] -> 1x1 mel mixer.
        raw: Waveform [B, T, 1] -> Gabor quadrature filterbank -> L1 magnitude.

    Magnitude scaling:
        'none': Pass-through.
        'pwl': Piecewise-linear compression (DW 1x1 branches + ReLU + Add).
        'pcen': PCEN-like compression (pool/conv/ReLU/Add).
        'db': Log compression (10*log10) — unfriendly to PTQ, avoid for deployment.

    Notes:
        - ``is_trainable`` controls the *filterbank* (raw) and *mel mixer*
          (hybrid). The per-band BatchNorm and the magnitude scaling are always
          trainable: they replace the per-sample normalization this layer used
          to apply, so they have to be learned from data to be calibrated.
        - The raw kernel length is at least the hop, so no input sample is
          skipped.
    """

    def __init__(
        self,
        mode: str,
        mel_bins: int,
        spec_width: int,
        sample_rate: int,
        chunk_duration: int,
        fft_length: int = 512,
        pcen_K: int = 8,
        init_mel: bool = True,
        mel_fmin: float = 150.0,
        mel_fmax: float | None = None,
        mel_norm: str = "slaney",
        mag_scale: str = "pwl",
        name: str = "audio_frontend",
        is_trainable: bool = False,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        assert mode in ("precomputed", "hybrid", "raw")
        assert mag_scale in ("pcen", "pwl", "db", "none")
        self.mode = mode
        self.mel_bins = int(mel_bins)
        self.spec_width = int(spec_width)
        self.sample_rate = int(sample_rate)
        self.chunk_duration = float(chunk_duration)
        self.fft_length = int(fft_length)
        self.pcen_K = int(pcen_K)
        self.init_mel = bool(init_mel)
        self.mel_fmin = float(mel_fmin)
        self.mel_fmax = mel_fmax
        self.mel_norm = mel_norm
        self.mag_scale = mag_scale
        self.is_trainable = bool(is_trainable)
        # Training may install a duck-typed quantization hook that simulates
        # the INT8 boundaries hidden inside this custom layer. It is never
        # serialized, so deployment models retain the ordinary clean graph.
        self._quantization_hook = None

        # Fixed input samples for one chunk
        self._T = int(self.sample_rate * self.chunk_duration)

        # Hybrid 1x1 mel mixer
        self.mel_mixer = layers.Conv2D(
            filters=int(self.mel_bins),
            kernel_size=(1, 1),
            padding="same",
            use_bias=False,
            kernel_constraint=constraints.NonNeg(),
            name=f"{name}_mel_mixer",
            trainable=self.is_trainable,
        )

        if self.mode == "raw":
            self._build_raw_filterbank(name)
        else:
            self.geom = None
            self.fb_re = None
            self.fb_im = None

        # Lowpass pooling over time, per band. A Gabor modulus is an envelope
        # estimate that rattles frame to frame, where a mel band integrates
        # power over its bandwidth; smoothing recovers that. Measured against
        # librosa mel, it lifts per-band agreement from ~0.43 to ~0.69.
        # Initialized to a Hann window and left trainable, so each band can
        # learn its own time constant.
        self.band_smooth = layers.DepthwiseConv2D(
            kernel_size=(1, SMOOTH_TAPS),
            padding="same",
            use_bias=False,
            depthwise_initializer=_hann_depthwise_init,
            name=f"{name}_band_smooth",
        )

        # Per-band calibration. This is what makes fixed gains work in place of
        # the old per-sample max normalization, so it always trains.
        self.band_bn = layers.BatchNormalization(
            momentum=0.99,
            epsilon=1e-3,
            name=f"{name}_band_bn",
        )
        # ReLU, not ReLU6: the magnitude is already non-negative before BN, and
        # an upper clip would emit a MINIMUM op that the N6 runs in software.
        self.band_relu = layers.ReLU(name=f"{name}_band_relu")

        # Magnitude scaling (composable layer)
        self.mag_layer = MagnitudeScalingLayer(
            method=self.mag_scale,
            channels=self.mel_bins,
            pcen_K=self.pcen_K,
            is_trainable=True,
            name=f"{name}_mag",
        )

    def _build_raw_filterbank(self, name: str) -> None:
        """Size and construct the quadrature filterbank for the raw path."""
        self.geom = raw_filterbank_geometry(self._T, int(self.spec_width))
        conv_kwargs = dict(
            filters=int(self.mel_bins),
            kernel_size=(1, self.geom.kernel),
            strides=(1, self.geom.stride),
            padding="valid",
            use_bias=False,
            trainable=self.is_trainable,
        )
        self.fb_re = layers.Conv2D(name=f"{name}_fb_re", **conv_kwargs)
        self.fb_im = layers.Conv2D(name=f"{name}_fb_im", **conv_kwargs)

    def build(self, input_shape):
        """Build the frontend layer based on the selected mode."""
        if self.mode == "hybrid":
            self._build_and_set_mel_mixer(n_fft=self.fft_length, cin=hybrid_fft_bins(self.fft_length))
        elif self.mode == "raw":
            folded = tf.TensorShape([None, 1, self.geom.crop // self.geom.fold, self.geom.fold])
            self.fb_re.build(folded)
            self.fb_im.build(folded)
            self._seed_gabor_weights()

        band_shape = tf.TensorShape([None, 1, int(self.spec_width), int(self.mel_bins)])
        if self.mode != "precomputed":
            if not self.band_smooth.built:
                self.band_smooth.build(band_shape)
            if not self.band_bn.built:
                self.band_bn.build(band_shape)
        self._build_mag_layer()
        super().build(input_shape)

    def _seed_gabor_weights(self) -> None:
        """Seed the raw filterbank with mel-spaced Gabor filters."""
        g = self.geom
        upper = float(self.mel_fmax) if self.mel_fmax is not None else (self.sample_rate / 2.0)
        real, imag = gabor_filterbank(
            mel_bins=self.mel_bins,
            kernel=g.window,
            sample_rate=self.sample_rate,
            fmin=self.mel_fmin,
            fmax=upper,
        )

        def _to_folded_kernel(taps: np.ndarray) -> np.ndarray:
            """[M, window] -> Conv2D kernel [1, kernel, fold, M].

            Folding sends input sample ``p * fold + c`` to channel ``c`` of
            frame ``p``, so tap ``j * fold + c`` of the filter is the weight at
            (frame offset ``j``, channel ``c``) — exactly a C-order reshape.
            """
            return taps.reshape(self.mel_bins, g.kernel, g.fold).transpose(1, 2, 0)[None]

        self.fb_re.set_weights([_to_folded_kernel(real)])
        self.fb_im.set_weights([_to_folded_kernel(imag)])

    def _build_and_set_mel_mixer(self, n_fft: int, cin: int):
        """Initialize mel_mixer from a Slaney mel basis."""
        upper = int(self.mel_fmax) if self.mel_fmax is not None else (self.sample_rate // 2)
        mel_mat = librosa.filters.mel(
            sr=int(self.sample_rate),
            n_fft=int(n_fft),
            n_mels=int(self.mel_bins),
            fmin=float(self.mel_fmin),
            fmax=float(upper),
            htk=False,
            norm="slaney",
        ).T.astype(np.float32)
        # Drop the Nyquist row to match hybrid_fft_bins(); no runtime pad needed.
        mel_mat = mel_mat[:cin, :]
        if not self.mel_mixer.built:
            self.mel_mixer.build(tf.TensorShape([None, 1, None, cin]))
        self.mel_mixer.set_weights([mel_mat[None, None, :, :]])

    def _build_mag_layer(self):
        """Ensure the magnitude scaling layer is built."""
        post_mel_shape = tf.TensorShape([None, 1, None, int(self.mel_bins)])
        if not self.mag_layer.built:
            self.mag_layer.build(post_mel_shape)

    def _apply_mag(self, x):
        """Dispatch to the magnitude scaling layer."""
        return self.mag_layer(x)

    def set_quantization_hook(self, hook) -> None:
        """Install or remove a training-only internal quantization hook."""
        self._quantization_hook = hook
        self.mag_layer.set_quantization_hook(hook)

    def _quantized_call(self, layer, inputs):
        """Call a kernel layer through the QAT hook when one is installed."""
        if self._quantization_hook is None:
            return layer(inputs)
        return self._quantization_hook.kernel(layer, inputs)

    def _quantized_activation(self, name: str, inputs):
        """Mark an internal tensor as an INT8 activation boundary for QAT."""
        if self._quantization_hook is None:
            return inputs
        return self._quantization_hook.activation(name, inputs)

    def _calibrate(self, y, training, smooth: bool = False):
        """Optional temporal lowpass, then per-band normalization and scaling."""
        if smooth:
            y = self._quantized_call(self.band_smooth, y)
        y = self.band_bn(y, training=training)
        y = self.band_relu(y)
        y = self._quantized_activation(self.band_relu.name, y)
        return self._apply_mag(y)

    def call(self, inputs, training=None):
        """Run the selected frontend path and return a fixed-size spectrogram.

        Shapes:
            precomputed: [B, mel_bins, T, 1] -> [B, mel_bins, spec_width, 1]
            hybrid: [B, fft_bins, T, 1] -> [B, mel_bins, spec_width, 1]
            raw: [B, T, 1] -> [B, mel_bins, spec_width, 1]
        """
        if self.mode == "precomputed":
            return inputs[:, :, : self.spec_width, :]

        if self.mode == "hybrid":
            fft_bins = hybrid_fft_bins(self.fft_length)
            if inputs.shape.rank != 4 or (inputs.shape[1] is not None and int(inputs.shape[1]) != fft_bins):
                raise ValueError(f"Hybrid expects [B,{fft_bins},T,1], got {inputs.shape}")
            y = tf.transpose(inputs, [0, 3, 2, 1])  # [B,1,T,fft_bins]
            y = y[:, :, : self.spec_width, :]
            y = self._quantized_call(self.mel_mixer, y)
            y = self._calibrate(y, training)
            y = tf.transpose(y, [0, 3, 2, 1])  # [B,mel,T,1]
            return y[:, :, : self.spec_width, :]

        # raw: fold -> quadrature filterbank -> magnitude -> calibrate
        g = self.geom
        # Reinterpret the waveform as `fold` interleaved channels. Free in NHWC
        # (same contiguous bytes), and it is what lets the convolution run at
        # stride 2 — the only strides the NPU takes — while the effective hop
        # stays at `g.hop` samples.
        y = tf.reshape(inputs[:, : g.crop, :], [-1, 1, g.crop // g.fold, g.fold])
        re = self._quantized_activation(self.fb_re.name, self._quantized_call(self.fb_re, y))
        im = self._quantized_activation(self.fb_im.name, self._quantized_call(self.fb_im, y))

        # alpha-max-plus-beta-min: |z| ~= max(|re|,|im|) + 0.4*min(|re|,|im|).
        # Within ~4% of the true magnitude, against ~17% ripple for |re|+|im| —
        # and that ripple would beat at the carrier frequency, aliasing into the
        # frame rate. Costs three elementwise ops, all of which stay on the NPU.
        a = self._quantized_activation(f"{self.name}_abs_re", tf.abs(re))
        b = self._quantized_activation(f"{self.name}_abs_im", tf.abs(im))
        maximum = self._quantized_activation(f"{self.name}_maximum", tf.maximum(a, b))
        minimum = self._quantized_activation(f"{self.name}_minimum", tf.minimum(a, b))
        scaled_minimum = self._quantized_activation(f"{self.name}_minimum_scale", 0.4 * minimum)
        mag = self._quantized_activation(f"{self.name}_magnitude", maximum + scaled_minimum)

        mag = self._calibrate(mag, training, smooth=True)
        mag = mag[:, :, : self.spec_width, :]
        return tf.transpose(mag, [0, 3, 2, 1])  # [B,mel,W,1]

    def compute_output_shape(self, input_shape):
        """Return static output shape: (batch, mel_bins, spec_width, 1)."""
        return (input_shape[0], int(self.mel_bins), int(self.spec_width), 1)

    def get_config(self):
        """Return a serializable configuration for model saving/loading."""
        cfg = {
            "mode": self.mode,
            "mel_bins": self.mel_bins,
            "spec_width": self.spec_width,
            "sample_rate": self.sample_rate,
            "chunk_duration": self.chunk_duration,
            "fft_length": self.fft_length,
            "pcen_K": self.pcen_K,
            "init_mel": self.init_mel,
            "mel_fmin": self.mel_fmin,
            "mel_fmax": self.mel_fmax,
            "mel_norm": self.mel_norm,
            "mag_scale": self.mag_scale,
            "name": self.name,
            "is_trainable": self.is_trainable,
        }
        base = super().get_config()
        base.update(cfg)
        return base
