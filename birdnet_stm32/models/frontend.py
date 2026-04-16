"""AudioFrontendLayer: in-model audio feature extraction for the STM32N6 NPU.

This Keras layer implements three interchangeable frontend modes that produce
a fixed-size mel-like spectrogram [B, mel_bins, spec_width, 1] from different
input representations:

- **precomputed**: Pass-through for offline mel spectrograms.
- **hybrid**: Linear STFT magnitude -> 1x1 Conv2D mel mixer (optionally trainable).
- **raw**: Raw waveform -> explicit-pad VALID Conv2D filterbank -> BN -> ReLU6.

Each mode supports configurable magnitude scaling (pwl, pcen, db, none) via
NPU-friendly depthwise convolution branches.
"""

import math
import warnings

import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import constraints, layers

# Canonical frontend names and deprecated aliases
VALID_FRONTENDS = ("librosa", "hybrid", "raw")
_FRONTEND_ALIASES = {"precomputed": "librosa", "tf": "raw"}


def normalize_frontend_name(name: str) -> str:
    """Normalize a frontend name, emitting deprecation warnings for aliases.

    Canonical names: 'librosa', 'hybrid', 'raw'.
    Deprecated aliases: 'precomputed' -> 'librosa', 'tf' -> 'raw'.

    Args:
        name: Frontend name (may be an alias).

    Returns:
        Canonical frontend name.

    Raises:
        ValueError: If name is not a valid frontend or alias.
    """
    if name in VALID_FRONTENDS:
        return name
    canonical = _FRONTEND_ALIASES.get(name)
    if canonical is not None:
        warnings.warn(
            f"Frontend name '{name}' is deprecated, use '{canonical}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return canonical
    raise ValueError(f"Invalid audio frontend: '{name}'. Valid options: {VALID_FRONTENDS}")


from birdnet_stm32.models.magnitude import MagnitudeScalingLayer  # noqa: E402


class AudioFrontendLayer(layers.Layer):
    """Audio frontend with interchangeable input modes and optional magnitude scaling.

    Modes:
        precomputed: Input mel spectrogram [B, mel_bins, spec_width, 1] -> slice to spec_width.
        hybrid: Linear STFT bins [B, fft_bins, spec_width, 1] -> 1x1 mel mixer.
        raw: Waveform [B, T, 1] -> explicit symmetric pad -> VALID Conv2D -> BN -> ReLU6.

    Magnitude scaling:
        'none': Pass-through.
        'pwl': Piecewise-linear compression (DW 1x1 branches + ReLU + Add).
        'pcen': PCEN-like compression (pool/conv/ReLU/Add).
        'db': Log compression (10*log10) — unfriendly to PTQ, avoid for deployment.

    Notes:
        - Slaney mel basis (librosa) seeds mel_mixer for parity in hybrid mode.
        - Raw branch uses explicit VALID padding for TF/TFLite SAME-padding parity.
        - Channel padding to multiples of 8 for NPU vectorization.
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
        mag_scale: str = "none",
        name: str = "audio_frontend",
        is_trainable: bool = False,
        train_mel_scale: bool = False,
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
        self.train_mel_scale = False

        # Fixed input samples for one chunk
        self._T = int(self.sample_rate * self.chunk_duration)
        self._pad_ch_in = 0

        # Hybrid 1x1 mel mixer
        self.mel_mixer = layers.Conv2D(
            filters=int(self.mel_bins),
            kernel_size=(1, 1),
            padding="same",
            use_bias=False,
            kernel_constraint=constraints.NonNeg(),
            name=f"{name}_mel_mixer",
            trainable=False,
        )

        # Placeholders for learnable mel
        self._bins_mel = None
        self._mel_fmin = None
        self._mel_fmax = None
        self._mel_range = None
        self._mel_seg_logits = None

        # RAW: single Conv2D with explicit VALID padding
        if self.mode == "raw":
            T = int(self.sample_rate * self.chunk_duration)
            W = int(self.spec_width)
            self._k_t = 16
            self._stride_t = int(math.ceil(T / float(W)))

            pad_total = max(0, self._stride_t * (W - 1) + self._k_t - T)
            self._pad_left = pad_total // 2
            self._pad_right = pad_total - self._pad_left

            self.fb2d = layers.Conv2D(
                filters=int(self.mel_bins),
                kernel_size=(1, self._k_t),
                strides=(1, self._stride_t),
                padding="valid",
                use_bias=False,
                name=f"{name}_raw_fb2d",
                trainable=self.is_trainable,
            )
            self.fb_bn = layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-3,
                name=f"{name}_raw_fb2d_bn",
                trainable=self.is_trainable,
            )
            self.fb_relu = layers.ReLU(max_value=6, name=f"{name}_raw_fb2d_relu")
        else:
            self._pad_left = 0
            self._pad_right = 0
            self._stride_t = 1
            self.fb2d = None
            self.fb_bn = None
            self.fb_relu = None

        # Magnitude scaling (composable layer)
        self.mag_layer = MagnitudeScalingLayer(
            method=self.mag_scale,
            channels=self.mel_bins,
            pcen_K=self.pcen_K,
            is_trainable=self.is_trainable,
            name=f"{name}_mag",
        )

    def build(self, input_shape):
        """Build the frontend layer based on the selected mode."""
        if self.mode == "hybrid":
            fft_bins = self.fft_length // 2 + 1
            self._build_and_set_mel_mixer(n_fft=self.fft_length, cin=fft_bins)

            if self.train_mel_scale:
                sr = int(self.sample_rate)
                fmax = int(self.mel_fmax) if self.mel_fmax is not None else (sr // 2)
                freqs = np.linspace(0.0, float(sr) / 2.0, fft_bins, dtype=np.float32)
                bins_mel = librosa.hz_to_mel(freqs)
                self._bins_mel = tf.constant(bins_mel.astype(np.float32), dtype=tf.float32)
                self._mel_fmin = float(librosa.hz_to_mel(float(self.mel_fmin)))
                self._mel_fmax = float(librosa.hz_to_mel(float(fmax)))
                self._mel_range = float(self._mel_fmax - self._mel_fmin)
                init_logits = np.zeros((self.mel_bins + 1,), dtype=np.float32)
                self._mel_seg_logits = self.add_weight(
                    name=f"{self.name}_mel_seg_logits",
                    shape=(self.mel_bins + 1,),
                    initializer=tf.keras.initializers.Constant(init_logits),
                    trainable=self.is_trainable,
                )

        elif self.mode == "raw":
            T = int(self.sample_rate * self.chunk_duration)
            static_w = int(self.spec_width)
            in_w = T + int(self._pad_left) + int(self._pad_right)
            self.fb2d.build(tf.TensorShape([None, 1, in_w, 1]))
            if self.fb_bn is not None:
                self.fb_bn.build(tf.TensorShape([None, 1, static_w, int(self.mel_bins)]))

        self._build_mag_layer()
        super().build(input_shape)

    def _compute_tri_matrix(self) -> tf.Tensor:
        """Build a triangular mel weight matrix from learnable breakpoints.

        Returns:
            [F, M] triangle weights for mel filterbank.
        """
        eps = tf.constant(1e-6, tf.float32)
        M = int(self.mel_bins)

        seg = tf.nn.softplus(self._mel_seg_logits) + 1e-3
        seg = seg / (tf.reduce_sum(seg) + eps) * tf.constant(self._mel_range, tf.float32)
        cs = tf.cumsum(seg)
        p_full = tf.concat(
            [tf.constant([self._mel_fmin], tf.float32), tf.constant([self._mel_fmin], tf.float32) + cs], axis=0
        )

        left = p_full[0:M]
        center = p_full[1 : M + 1]
        right = p_full[2 : M + 2]
        bm = self._bins_mel

        denom_l = tf.maximum(center - left, eps)
        denom_r = tf.maximum(right - center, eps)
        up = (bm[:, None] - left[None, :]) / denom_l[None, :]
        down = (right[None, :] - bm[:, None]) / denom_r[None, :]

        tri = tf.maximum(tf.minimum(up, down), 0.0)
        tri = tri / (tf.reduce_sum(tri, axis=0, keepdims=True) + eps)
        return tri

    def _assign_mel_kernel_from_tri(self, tri: tf.Tensor):
        """Mirror the [F, M] triangle matrix into the 1x1 Conv2D mel_mixer kernel."""
        if self.mel_mixer is None or not hasattr(self.mel_mixer, "kernel"):
            return
        if getattr(self, "_pad_ch_in", 0):
            pad = self._pad_ch_in
            zeros = tf.zeros([pad, int(self.mel_bins)], dtype=tri.dtype)
            tri = tf.concat([tri, zeros], axis=0)
        k = tf.reshape(tri, [1, 1, tf.shape(tri)[0], int(self.mel_bins)])
        self.mel_mixer.kernel.assign(k)

    def _build_and_set_mel_mixer(self, n_fft: int, cin: int):
        """Initialize mel_mixer from a Slaney mel basis and pad input channels if needed."""
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
        pad = (8 - (cin % 8)) % 8
        if pad:
            mel_mat = np.pad(mel_mat, ((0, pad), (0, 0)), mode="constant")
        mel_kernel = mel_mat[None, None, :, :]
        if not self.mel_mixer.built:
            self.mel_mixer.build(tf.TensorShape([None, 1, None, cin + pad]))
        self.mel_mixer.set_weights([mel_kernel])
        self._pad_ch_in = pad

    def _build_mag_layer(self):
        """Ensure the magnitude scaling layer is built."""
        post_mel_shape = tf.TensorShape([None, 1, None, int(self.mel_bins)])
        if not self.mag_layer.built:
            self.mag_layer.build(post_mel_shape)

    def _apply_mag(self, x):
        """Dispatch to the magnitude scaling layer."""
        return self.mag_layer(x)

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
            fft_bins = self.fft_length // 2 + 1
            if inputs.shape.rank != 4 or (inputs.shape[1] is not None and int(inputs.shape[1]) != fft_bins):
                raise ValueError(f"Hybrid expects [B,{fft_bins},T,1], got {inputs.shape}")
            y = tf.transpose(inputs, [0, 3, 2, 1])  # [B,1,T,fft_bins]
            y = y[:, :, : self.spec_width, :]

            if self.train_mel_scale and self.is_trainable:

                def _train_branch(y_in):
                    tri = self._compute_tri_matrix()
                    self._assign_mel_kernel_from_tri(tf.stop_gradient(tri))
                    B = tf.shape(y_in)[0]
                    Tt = tf.shape(y_in)[2]
                    F = tf.shape(y_in)[3]
                    y_flat = tf.reshape(y_in, [B * Tt, F])
                    y_mel = tf.matmul(y_flat, tri)
                    return tf.reshape(y_mel, [B, 1, Tt, int(self.mel_bins)])

                def _infer_branch(y_in):
                    if self._pad_ch_in:
                        b = tf.shape(y_in)[0]
                        t = tf.shape(y_in)[2]
                        z = tf.zeros([b, 1, t, self._pad_ch_in], dtype=y_in.dtype)
                        y_in = tf.concat([y_in, z], axis=-1)
                    return self.mel_mixer(y_in)

                if isinstance(training, bool):
                    y = _train_branch(y) if training else _infer_branch(y)
                else:
                    y = tf.cond(tf.cast(training, tf.bool), lambda: _train_branch(y), lambda: _infer_branch(y))
            else:
                if self._pad_ch_in:
                    b = tf.shape(y)[0]
                    t = tf.shape(y)[2]
                    z = tf.zeros([b, 1, t, self._pad_ch_in], dtype=y.dtype)
                    y = tf.concat([y, z], axis=-1)
                y = self.mel_mixer(y)

            y = tf.nn.relu(y)
            y = self._apply_mag(y)
            y = tf.transpose(y, [0, 3, 2, 1])  # [B,mel,T,1]
            return y[:, :, : self.spec_width, :]

        # raw: explicit symmetric pad -> VALID Conv2D -> BN -> ReLU6 -> mag -> transpose
        x = inputs[:, : int(self.sample_rate * self.chunk_duration), :]
        if self._pad_left or self._pad_right:
            x = tf.pad(x, [[0, 0], [int(self._pad_left), int(self._pad_right)], [0, 0]])
        y = tf.expand_dims(x, axis=1)
        y = self.fb2d(y)
        if self.fb_bn is not None:
            y = self.fb_bn(y, training=training)
        y = self.fb_relu(y)
        y = self._apply_mag(y)
        y = tf.transpose(y, [0, 3, 2, 1])
        return y

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
