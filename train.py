import os
import argparse
import numpy as np
from typing import Optional
import tensorflow as tf
from tensorflow.keras import layers, constraints, regularizers
import math
import json

from utils.audio import (
    load_audio_file,
    get_spectrogram_from_audio,
    get_linear_spectrogram_from_audio,
    sort_by_s2n,
    pick_random_samples,
    plot_spectrogram,
)

# Mute TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# random seed for reproducibility
np.random.seed(42)

# Enable dynamic GPU memory allocation
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"Could not set memory growth: {e}")
    
def dataset_sanity_check(file_paths, classes, sample_rate=22050, max_duration=30, chunk_duration=3, spec_width=128, mel_bins=64, audio_frontend='precomputed', tf_frontend_layer=None, fft_length=512, mag_scale='none'):
    """
    Quick visual check of frontend outputs on a few files.

    Modes:
      - precomputed/librosa: compute mel spectrograms in utils (mag_scale applied there).
      - hybrid:              linear power spectrogram -> fixed mel -> mag_scale (in-model).
      - raw/tf:              raw -> STFT power -> fixed mel -> mag_scale (in-model).

    Saves plots into ./samples for manual inspection.
    """
    os.makedirs("samples", exist_ok=True)
    np.random.shuffle(file_paths)

    tf_frontend = None
    if audio_frontend in ('hybrid', 'tf', 'raw'):
        tf_frontend = tf_frontend_layer or AudioFrontendLayer(
            mode='raw' if audio_frontend in ('tf', 'raw') else 'hybrid',
            mel_bins=mel_bins,
            spec_width=spec_width,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            fft_length=fft_length,
            mag_scale=mag_scale,
            name="audio_frontend_sanity",
        )
        dummy_shape = (1, chunk_duration * sample_rate, 1) if audio_frontend in ('tf', 'raw') else (1, (fft_length // 2 + 1), spec_width, 1)
        dummy = tf.zeros(dummy_shape, dtype=tf.float32)
        _ = tf_frontend(dummy, training=False)

    for i, path in enumerate(file_paths[:5]):
        audio_chunks = load_audio_file(path, sample_rate, max_duration, chunk_duration)
        if len(audio_chunks) == 0: continue

        if audio_frontend in ('librosa', 'precomputed'):
            specs = [get_spectrogram_from_audio(chunk, sample_rate, n_fft=fft_length, mel_bins=mel_bins, spec_width=spec_width, mag_scale=mag_scale) for chunk in audio_chunks]
            pool = sort_by_s2n(specs, threshold=0.5) or specs
            if len(pool) == 0: continue
            spec = pick_random_samples(pool, num_samples=1)
            spec = spec[0] if isinstance(spec, list) else spec
        elif audio_frontend == 'hybrid':
            specs = [get_linear_spectrogram_from_audio(chunk, sample_rate, n_fft=fft_length, spec_width=spec_width, power=2.0) for chunk in audio_chunks]
            pool = sort_by_s2n(specs, threshold=0.5) or specs
            if len(pool) == 0: continue
            spec_in = pick_random_samples(pool, num_samples=1)
            spec_in = spec_in[0] if isinstance(spec_in, list) else spec_in
            inp = spec_in[np.newaxis, ..., np.newaxis].astype(np.float32)
            spec = tf_frontend(inp, training=False).numpy()[0, :, :, 0]
        else:  # raw/tf
            pool = sort_by_s2n(audio_chunks, threshold=0.5) or audio_chunks
            if len(pool) == 0: continue
            chunk = pick_random_samples(pool, num_samples=1)
            chunk = (chunk[0] if isinstance(chunk, list) else chunk)[:sample_rate * chunk_duration]
            if len(chunk) < sample_rate * chunk_duration:
                chunk = np.pad(chunk, (0, sample_rate * chunk_duration - len(chunk)))
            inp = chunk[np.newaxis, ..., np.newaxis].astype(np.float32)
            spec = tf_frontend(inp, training=False).numpy()[0, :, :, 0]

        plot_spectrogram(spec, title=f"{audio_frontend}_{os.path.basename(path)}")
        
def get_classes_with_most_samples(directory, n_classes=25, include_noise=False):
    
    """    Get the most common classes from the dataset directory.      
    Args:
        n_classes (int): Number of classes to return.
        include_noise (bool): Whether to include noise-like classes.
    Returns:
        list: Sorted list of class names.
    """
    
    classes = {} # 'class_name': num_samples
    noise_classes = {'noise', 'silence', 'background', 'other'}
    
    for root, dirs, files in os.walk(directory):
        for fname in files:
            if not fname.endswith('.wav'):
                continue
            class_name = os.path.basename(root)
            if not include_noise and class_name.lower() in noise_classes:
                continue
            classes[class_name] = classes.get(class_name, 0) + 1
            
    # Sort by number of samples, descending
    sorted_classes = sorted(classes.items(), key=lambda x: x[1], reverse=True)
    
    # Return only the class names, up to n_classes
    return [cls for cls, _ in sorted_classes[:n_classes]]    

def load_file_paths_from_directory(directory, classes=None, max_samples=None):
    """
    Recursively collect .wav files from a dataset directory.

    Directory structure is expected to be class-sublabeled (root/class_x/*.wav).
    If classes is provided, only those class names are included.
    If max_samples is set (>0), up to max_samples files are taken per class (uniform random).

    Note:
      Noise-like classes ('noise', 'silence', 'background', 'other') are removed from the
      returned classes list, but their files remain in file_paths (treated as negatives later).

    Args:
        directory: Root dataset directory.
        classes: Optional list of class names to include.
        max_samples: Optional limit on the number of files per class.

    Returns:
        (file_paths, classes_sorted)
        - file_paths: Shuffled list of absolute file paths (limited per class if requested).
        - classes_sorted: Sorted list of class names (noise-like classes removed).
    """
    per_class = {}

    # Walk and bucket files by their immediate parent directory (class)
    for root, _, files in tf.io.gfile.walk(directory):
        for fname in files:
            if not fname.endswith('.wav'):
                continue
            full_path = tf.io.gfile.join(root, fname)
            parent_class = os.path.basename(os.path.dirname(full_path))

            # Filter by provided classes (if any)
            if classes is not None and parent_class not in classes:
                continue

            per_class.setdefault(parent_class, []).append(full_path)

    # Enforce max_samples per class (uniform random)
    all_paths = []
    for cls, paths in per_class.items():
        if max_samples is not None and max_samples > 0 and len(paths) > max_samples:
            idx = np.random.permutation(len(paths))[:max_samples]
            paths = [paths[i] for i in idx]
        all_paths.extend(paths)

    # Shuffle combined paths for randomness
    np.random.shuffle(all_paths)

    # Build classes list from collected keys, excluding noise-like labels
    noise_classes = {'noise', 'silence', 'background', 'other'}
    classes_out = sorted([c for c in per_class.keys() if c.lower() not in noise_classes])

    return all_paths, classes_out

def data_generator(file_paths, classes, batch_size=32, audio_frontend='librosa', sample_rate=22050, max_duration=30, chunk_duration=3, spec_width=128, mixup_alpha=0.2, mixup_probability=0.25, mel_bins=48, fft_length=512, mag_scale='none'):
    """
    Python generator producing (inputs, one_hot_labels) batches.

    Inputs per frontend:
      - precomputed/librosa: mel spectrogram (mel_bins, spec_width, 1)
      - hybrid:              linear power spectrogram (fft_bins, spec_width, 1)
      - raw/tf:              raw waveform (sample_rate*chunk_duration, 1)

    Basic SNR-based selection is applied to choose a representative chunk. Mixup optional.
    """
    T = sample_rate * chunk_duration
    while True:
        idxs = np.random.permutation(len(file_paths))
        for batch_start in range(0, len(idxs), batch_size):
            batch_idxs = idxs[batch_start:batch_start + batch_size]
            batch_samples, batch_labels = [], []
            for idx in batch_idxs:
                path = file_paths[idx]
                audio_chunks = load_audio_file(path, sample_rate, max_duration, chunk_duration)
                if len(audio_chunks) == 0:
                    continue

                if audio_frontend in ('librosa', 'precomputed'):
                    specs = [get_spectrogram_from_audio(chunk, sample_rate, n_fft=fft_length, mel_bins=mel_bins, spec_width=spec_width, mag_scale=mag_scale) for chunk in audio_chunks]
                    pool = sort_by_s2n(specs, threshold=0.5) or specs
                    if len(pool) == 0: continue
                    sample = pick_random_samples(pool, num_samples=1)
                    sample = sample[0] if isinstance(sample, list) else sample

                elif audio_frontend == 'hybrid':
                    specs = [get_linear_spectrogram_from_audio(chunk, sample_rate, n_fft=fft_length, spec_width=spec_width, power=2.0) for chunk in audio_chunks]
                    pool = sort_by_s2n(specs, threshold=0.5) or specs
                    if len(pool) == 0: continue
                    sample = pick_random_samples(pool, num_samples=1)
                    sample = sample[0] if isinstance(sample, list) else sample

                elif audio_frontend in ('tf', 'raw'):
                    pool = sort_by_s2n(audio_chunks, threshold=0.5) or audio_chunks
                    if len(pool) == 0: continue
                    sample = pick_random_samples(pool, num_samples=1)
                    x = sample[0] if isinstance(sample, list) else sample
                    x = x[:T]
                    if x.shape[0] < T: x = np.pad(x, (0, T - x.shape[0]))
                    sample = x
                else:
                    raise ValueError("Invalid audio frontend. Choose 'precomputed', 'hybrid', or 'raw'.")

                label_str = path.split('/')[-2]
                if label_str.lower() in ['noise', 'silence', 'background', 'other']:
                    one_hot_label = np.zeros(len(classes), dtype=np.float32)
                else:
                    if label_str not in classes: continue
                    one_hot_label = tf.one_hot(classes.index(label_str), depth=len(classes)).numpy()

                sample = np.expand_dims(sample, axis=-1)
                batch_samples.append(sample.astype(np.float32))
                batch_labels.append(one_hot_label.astype(np.float32))

            if len(batch_samples) == 0: continue
            batch_samples = np.stack(batch_samples)
            batch_labels = np.stack(batch_labels)

            # Mixup (unchanged)
            if mixup_alpha > 0 and mixup_probability > 0:
                num_mix = int(batch_samples.shape[0] * mixup_probability)
                if num_mix > 0:
                    mix_indices = np.random.choice(batch_samples.shape[0], size=num_mix, replace=False)
                    permuted_indices = np.random.permutation(batch_samples.shape[0])
                    lam = np.random.beta(mixup_alpha, mixup_alpha, size=(num_mix,))
                    lam_inp = lam.reshape((num_mix,) + (1,) * (batch_samples.ndim - 1))
                    lam_lbl = lam.reshape((num_mix, 1))
                    batch_samples[mix_indices] = lam_inp * batch_samples[mix_indices] + (1 - lam_inp) * batch_samples[permuted_indices[mix_indices]]
                    batch_labels[mix_indices] = lam_lbl * batch_labels[mix_indices] + (1 - lam_lbl) * batch_labels[permuted_indices[mix_indices]]

            yield batch_samples, batch_labels

def load_dataset(file_paths, classes, audio_frontend='precomputed', batch_size=32, spec_width=128, mel_bins=48, **kwargs):
    """
    Wrap a Python generator into tf.data with fixed signatures.

    Input shapes by frontend:
      - precomputed/librosa: (None, mel_bins, spec_width, 1)
      - hybrid:              (None, fft_bins, spec_width, 1)
      - raw/tf:              (None, sample_rate*chunk_duration, 1)
    """
    sr = kwargs.get('sample_rate', 16000)
    cd = kwargs.get('chunk_duration', 3)
    fft_length = kwargs.get('fft_length', 512)
    mag_scale = kwargs.get('mag_scale', 'none')
    chunk_len = sr * cd

    if audio_frontend in ('librosa', 'precomputed'):
        input_spec = tf.TensorSpec(shape=(None, mel_bins, spec_width, 1), dtype=tf.float32)
    elif audio_frontend == 'hybrid':
        fft_bins = fft_length // 2 + 1
        input_spec = tf.TensorSpec(shape=(None, fft_bins, spec_width, 1), dtype=tf.float32)
    elif audio_frontend in ('tf', 'raw'):
        input_spec = tf.TensorSpec(shape=(None, chunk_len, 1), dtype=tf.float32)
    else:
        raise ValueError("Invalid audio frontend. Choose 'precomputed', 'hybrid', or 'raw'.")

    output_signature = (input_spec, tf.TensorSpec(shape=(None, len(classes)), dtype=tf.float32))

    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(
            file_paths, classes,
            batch_size=batch_size,
            audio_frontend=audio_frontend,
            sample_rate=sr,
            max_duration=kwargs.get('max_duration', 30),
            chunk_duration=cd,
            spec_width=spec_width,
            mixup_alpha=kwargs.get('mixup_alpha', 0.0),
            mixup_probability=kwargs.get('mixup_probability', 0.0),
            mel_bins=mel_bins,
            fft_length=fft_length,
            mag_scale=mag_scale
        ),
        output_signature=output_signature
    )
    return dataset.repeat().prefetch(tf.data.AUTOTUNE)

class AudioFrontendLayer(layers.Layer):
    """
    Audio frontend:
      - mode='precomputed': input [B, mel_bins, spec_width, 1]. Optional mag_scale (pcen/pwl).
      - mode='hybrid':      input [B, fft_bins, spec_width, 1]. Fixed mel (tf.signal weights).
      - mode='raw':         input [B, T, 1]. STFT -> power -> fixed mel.
    """
    def __init__(
        self,
        mode: str,
        mel_bins: int,
        spec_width: int,
        sample_rate: int,
        chunk_duration: int,
        fft_length: int = 512,
        use_pcen: bool = False,
        pcen_K: int = 4,
        init_mel: bool = True,
        mel_fmin: float = 150.0,
        mel_fmax: Optional[float] = None,
        mel_norm: str = "slaney",
        mel_htk: bool = False,  # kept for API compatibility (unused with tf.signal)
        mag_scale: str = "none",  # 'pcen' | 'pwl' | 'none'
        name: str = "audio_frontend",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        assert mode in ("precomputed", "hybrid", "raw")
        assert mag_scale in ("pcen", "pwl", "none")
        self.mode = mode
        self.mel_bins = int(mel_bins)
        self.spec_width = int(spec_width)
        self.sample_rate = int(sample_rate)
        self.chunk_duration = int(chunk_duration)
        self.fft_length = int(fft_length)
        self.use_pcen = bool(use_pcen)
        self.pcen_K = int(pcen_K)
        self.init_mel = bool(init_mel)
        self.mel_fmin = float(mel_fmin)
        self.mel_fmax = mel_fmax
        self.mel_norm = mel_norm
        self.mag_scale = mag_scale

        # Resolve legacy flag
        if self.use_pcen and self.mag_scale == "none":
            self.mag_scale = "pcen"

        self._T = self.sample_rate * self.chunk_duration
        self._hop = max(1, self._T // max(1, self.spec_width))
        self._hann = None  # created lazily

        # Layout helpers
        self.permute_to_C = layers.Permute((3, 2, 1), name=f"{name}_to_C_last")   # (B,H,W,C)->(B,C,W,H)
        self.permute_back = layers.Permute((3, 2, 1), name=f"{name}_from_C_last") # (B,C,W,H)->(B,H,W,C)

        # Fixed mel projection (1x1 Conv2D). We will set tf.signal mel weights and freeze it.
        self.mel_mixer = layers.Conv2D(
            filters=self.mel_bins,
            kernel_size=(1, 1),
            padding="same",
            use_bias=True,
            kernel_constraint=constraints.NonNeg(),
            kernel_regularizer=regularizers.l2(1e-6),
            name=f"{name}_mel_mixer",
        )
        # Optional stabilization
        self.mel_post_bn = layers.BatchNormalization(center=False, scale=False, name=f"{name}_mel_bn")

        # Inline PCEN/PWL sublayers (no extra Layer classes)
        self._pcen_pools = [layers.AveragePooling2D(pool_size=(1, 3), strides=(1, 1), padding="same",
                                                    name=f"{name}_pcen_ema{k}") for k in range(self.pcen_K)] if self.mag_scale == "pcen" else []
        # Depthwise 1x1 for per-channel gains/hinges (built lazily on first call)
        self._pcen_agc_dw = layers.DepthwiseConv2D((1, 1), use_bias=False,
                                                   depthwise_initializer=tf.keras.initializers.Constant(0.6),
                                                   padding="same", name=f"{name}_pcen_agc_dw") if self.mag_scale == "pcen" else None
        self._pcen_k1_dw = layers.DepthwiseConv2D((1, 1), use_bias=False,
                                                  depthwise_initializer=tf.keras.initializers.Constant(0.15),
                                                  padding="same", name=f"{name}_pcen_k1_dw") if self.mag_scale == "pcen" else None
        self._pcen_shift_dw = layers.DepthwiseConv2D((1, 1), use_bias=True,
                                                     depthwise_initializer=tf.keras.initializers.Ones(),
                                                     bias_initializer=tf.keras.initializers.Constant(-0.2),
                                                     padding="same", name=f"{name}_pcen_shift_dw") if self.mag_scale == "pcen" else None
        self._pcen_k2mk1_dw = layers.DepthwiseConv2D((1, 1), use_bias=False,
                                                     depthwise_initializer=tf.keras.initializers.Constant(0.45),
                                                     padding="same", name=f"{name}_pcen_k2mk1_dw") if self.mag_scale == "pcen" else None

        if self.mag_scale == "pwl":
            self._pwl_k0_dw = layers.DepthwiseConv2D((1, 1), use_bias=False,
                                                     depthwise_initializer=tf.keras.initializers.Constant(0.40),
                                                     padding="same", name=f"{name}_pwl_k0_dw")
            # Three hinges: t=[0.10, 0.35, 0.65], slopes=[0.25,0.15,0.08]
            self._pwl_shift_dws = [
                layers.DepthwiseConv2D((1, 1), use_bias=True,
                                       depthwise_initializer=tf.keras.initializers.Ones(),
                                       bias_initializer=tf.keras.initializers.Constant(-t),
                                       padding="same", name=f"{name}_pwl_shift{i+1}_dw")
                for i, t in enumerate((0.10, 0.35, 0.65))
            ]
            self._pwl_k_dws = [
                layers.DepthwiseConv2D((1, 1), use_bias=False,
                                       depthwise_initializer=tf.keras.initializers.Constant(k),
                                       padding="same", name=f"{name}_pwl_k{i+1}_dw")
                for i, k in enumerate((0.25, 0.15, 0.08))
            ]
        else:
            self._pwl_k0_dw = None
            self._pwl_shift_dws = []
            self._pwl_k_dws = []

    def build(self, input_shape):
        if self.mode == "raw":
            self._hann = tf.signal.hann_window(self.fft_length, dtype=tf.float32)

        # Set tf.signal mel matrix on mel_mixer and freeze it
        if self.mode in ("hybrid", "raw") and self.init_mel:
            fft_bins = self.fft_length // 2 + 1
            try:
                self.mel_mixer.build(tf.TensorShape([None, 1, None, fft_bins]))
            except Exception:
                pass
            upper = int(self.mel_fmax) if self.mel_fmax is not None else (self.sample_rate // 2)
            mel_mat = tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins=self.mel_bins,
                num_spectrogram_bins=fft_bins,
                sample_rate=float(self.sample_rate),
                lower_edge_hertz=float(self.mel_fmin),
                upper_edge_hertz=float(upper),
                dtype=tf.float32
            )  # [fft_bins, mel_bins]
            kernel = tf.expand_dims(tf.expand_dims(mel_mat, axis=0), axis=0).numpy()  # [1,1,fft_bins,mel_bins]
            bias = np.zeros((self.mel_bins,), dtype=np.float32)
            if len(self.mel_mixer.get_weights()) == 2:
                W, b = self.mel_mixer.get_weights()
                if W.shape == kernel.shape:
                    self.mel_mixer.set_weights([kernel.astype(np.float32), bias])
            else:
                self.mel_mixer.set_weights([kernel.astype(np.float32), bias])
            # Freeze mel
            self.mel_mixer.trainable = False

        super().build(input_shape)

    def _align_time(self, y):
        # y: [B,1,T,?] -> center crop/pad to spec_width
        cur_w = tf.shape(y)[2]
        def crop():
            start = (cur_w - self.spec_width) // 2
            end = cur_w - self.spec_width - start
            return layers.Cropping2D(cropping=((0, 0), (start, end)))(y)
        def pad():
            padw = self.spec_width - cur_w
            left = padw // 2
            right = padw - left
            return layers.ZeroPadding2D(padding=((0, 0), (left, right)))(y)
        y = tf.__internal__.smart_cond.smart_cond(tf.greater(cur_w, self.spec_width), crop, lambda: y)
        y = tf.__internal__.smart_cond.smart_cond(tf.less(cur_w, self.spec_width), pad, lambda: y)
        return y

    def _apply_pcen(self, x):
        # x: [B,1,T,C]
        m = x
        for pool in self._pcen_pools:
            m = pool(m)
        # AGC residual -> ReLU
        y0 = layers.Subtract(name=f"{self.name}_pcen_agc_sub")([x, self._pcen_agc_dw(m)])
        y0 = layers.ReLU(name=f"{self.name}_pcen_agc_relu")(y0)
        # k1*y0 + (k2-k1)*relu(y0 - t)
        b1 = self._pcen_k1_dw(y0)
        relu = layers.ReLU(name=f"{self.name}_pcen_relu")(self._pcen_shift_dw(y0))
        b2 = self._pcen_k2mk1_dw(relu)
        y = layers.Add(name=f"{self.name}_pcen_add")([b1, b2])
        # Clamp non-negative
        return layers.ReLU(name=f"{self.name}_pcen_out_relu")(y)

    def _apply_pwl(self, x):
        # x: [B,1,T,C]
        branches = [self._pwl_k0_dw(x)]
        for i, (shift, k_dw) in enumerate(zip(self._pwl_shift_dws, self._pwl_k_dws), start=1):
            relu = layers.ReLU(name=f"{self.name}_pwl_relu{i}")(shift(x))
            branches.append(k_dw(relu))
        return layers.Add(name=f"{self.name}_pwl_add")(branches)

    def _apply_mag(self, x):
        if self.mag_scale == "pcen":
            return self._apply_pcen(x)
        if self.mag_scale == "pwl":
            return self._apply_pwl(x)
        return x

    def call(self, inputs, training=None):
        if self.mode == "precomputed":
            # [B, mel, T, 1] -> optional mag scale
            y = inputs[:, :, :self.spec_width, :]
            if self.mag_scale != "none":
                y = self.permute_to_C(y)          # [B,1,T,mel]
                y = self._apply_mag(y)
                y = self.permute_back(y)          # [B, mel, T, 1]
            return y

        if self.mode == "hybrid":
            # inputs: [B, fft_bins, T, 1] -> [B,1,T,fft_bins]
            y = self.permute_to_C(inputs)
        else:
            # raw: [B,T,1] -> STFT power -> [B,1,frames,fft_bins]
            x = inputs
            stft = tf.signal.stft(
                signals=tf.squeeze(x, axis=-1),
                frame_length=self.fft_length,
                frame_step=self._hop,
                window_fn=lambda frame_length, dtype: self._hann,
                pad_end=True
            )
            power = tf.math.real(stft * tf.math.conj(stft))
            y = tf.expand_dims(power, axis=1)

        # Align time, fixed mel, stabilization
        y = self._align_time(y)
        y = self.mel_mixer(y)                          # fixed mel
        y = self.mel_post_bn(y, training=training)
        y = layers.ReLU(name=f"{self.name}_mel_relu")(y)

        # Optional magnitude scaling
        y = self._apply_mag(y)

        # [B, mel, T, 1]
        y = self.permute_back(y)
        return y[:, :, :self.spec_width, :]

def _make_divisible(v, divisor=8):
    v = int(v + divisor / 2) // divisor * divisor
    return max(divisor, v)

def ds_conv_block(x, out_ch, stride_f=1, stride_t=1, name="ds"):
    """
    Depthwise-separable conv block (STM32N6-friendly):
      - DepthwiseConv2D 3x3, stride=(stride_f, stride_t), padding='same'
      - BN + ReLU6
      - Pointwise Conv2D 1x1, stride 1
      - BN (+ residual if stride=(1,1) and channels match)
      - ReLU6

    Strides:
      - stride_f: along frequency (mel, height)
      - stride_t: along time (width)
      Use 1 or 2 only to stay in NPU fast path.
    """
    in_ch = x.shape[-1]
    y = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=(stride_f, stride_t),
        padding='same',
        use_bias=False,
        name=f"{name}_dw",
    )(x)
    y = layers.BatchNormalization(name=f"{name}_dw_bn")(y)
    y = layers.ReLU(max_value=6, name=f"{name}_dw_relu")(y)

    y = layers.Conv2D(
        filters=out_ch,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        use_bias=False,
        name=f"{name}_pw",
    )(y)
    y = layers.BatchNormalization(name=f"{name}_pw_bn")(y)

    if (stride_f == 1 and stride_t == 1) and (in_ch is not None and int(in_ch) == int(out_ch)):
        y = layers.Add(name=f"{name}_add")([x, y])

    y = layers.ReLU(max_value=6, name=f"{name}_pw_relu")(y)
    return y

def build_dscnn_model(num_mels, spec_width, sample_rate, chunk_duration, embeddings_size, num_classes, audio_frontend='precomputed', alpha=1.0, depth_multiplier=1, fft_length=512, mag_scale='none'):
    """
    Build DS-CNN with selectable audio frontend (mag_scale: 'pcen'|'pwl'|'none').
    """
    if audio_frontend in ('librosa', 'precomputed'):
        inputs = tf.keras.Input(shape=(num_mels, spec_width, 1), name='mel_spectrogram_input')
        x = AudioFrontendLayer(
            mode='precomputed',
            mel_bins=num_mels,
            spec_width=spec_width,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            fft_length=fft_length,
            mag_scale=mag_scale,
            name="audio_frontend",
        )(inputs)
    elif audio_frontend == 'hybrid':
        fft_bins = fft_length // 2 + 1
        inputs = tf.keras.Input(shape=(fft_bins, spec_width, 1), name='linear_spectrogram_input')
        x = AudioFrontendLayer(
            mode='hybrid',
            mel_bins=num_mels,
            spec_width=spec_width,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            fft_length=fft_length,
            mag_scale=mag_scale,
            name="audio_frontend",
        )(inputs)
    elif audio_frontend in ('tf', 'raw'):
        inputs = tf.keras.Input(shape=(chunk_duration * sample_rate, 1), name='raw_audio_input')
        x = AudioFrontendLayer(
            mode='raw',
            mel_bins=num_mels,
            spec_width=spec_width,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            fft_length=fft_length,
            mag_scale=mag_scale,
            name="audio_frontend",
        )(inputs)
    else:
        raise ValueError("Invalid audio_frontend.")

    # Stem (3x3, stride 1) to lift channels
    stem_ch = _make_divisible(int(24 * alpha), 8)
    x = layers.Conv2D(stem_ch, (3, 3), strides=(1, 1), padding='same', use_bias=False, name="stem_conv")(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(max_value=6, name="stem_relu")(x)

    # Stages: (filters, repeats, (stride_f, stride_t))
    # Use stride 2 in both axes early to reduce HxW; last stage keeps 1x1.
    base_filters = [32, 64, 128, 256]
    base_repeats = [2, 3, 3, 2]
    base_strides = [(2, 2), (2, 2), (2, 2), (2, 2)]

    for si, (bf, br, (sf, st)) in enumerate(zip(base_filters, base_repeats, base_strides), start=1):
        out_ch = _make_divisible(int(bf * alpha), 8)
        reps = max(1, int(math.ceil(br * depth_multiplier)))
        # First block in stage may downsample both frequency and time
        x = ds_conv_block(x, out_ch, stride_f=sf, stride_t=st, name=f"stage{si}_ds1")
        for bi in range(2, reps + 1):
            x = ds_conv_block(x, out_ch, stride_f=1, stride_t=1, name=f"stage{si}_ds{bi}")
            
    # Final 1x1 conv to embeddings
    emb_ch = _make_divisible(int(embeddings_size * alpha), 8)
    # check if we already have emb_ch channels
    if not (x.shape[-1] is not None and int(x.shape[-1]) == int(emb_ch)):    
        x = layers.Conv2D(emb_ch, (1, 1), strides=(1, 1), padding='same', use_bias=False, name="emb_conv")(x)
        x = layers.BatchNormalization(name="emb_bn")(x)
        x = layers.ReLU(max_value=6, name="emb_relu")(x)

    # Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation='sigmoid', name="pred")(x)
    return tf.keras.models.Model(inputs, outputs, name="dscnn_audio")

def train_model(model, train_dataset, val_dataset, epochs=50, learning_rate=0.001, batch_size=64, patience=5, checkpoint_path="checkpoints/best_model.keras", steps_per_epoch=None, val_steps=None):
    """
    Train with cosine-annealed Adam, early stopping, and ModelCheckpoint (.keras).

    Metrics:
      - ROC AUC (multi-label)

    Args:
        model: tf.keras.Model.
        train_dataset: tf.data.Dataset of (inputs, labels).
        val_dataset: tf.data.Dataset of (inputs, labels).
        epochs: Number of epochs.
        learning_rate: Initial LR for cosine schedule.
        batch_size: Unused (for API symmetry).
        patience: Early stopping patience.
        checkpoint_path: Destination .keras path.
        steps_per_epoch: Training steps per epoch (required).
        val_steps: Validation steps per epoch.
    """

    if steps_per_epoch is None or steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be > 0")
    if val_steps is None or val_steps <= 0:
        val_steps = 1

    # Ensure checkpoint dir exists
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    first_decay_steps = max(1, steps_per_epoch * max(1, epochs // 10))
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=learning_rate,
        first_decay_steps=first_decay_steps,
        t_mul=2.0,
        m_mul=1.0,
        alpha=1e-4
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(curve='ROC', multi_label=True, name="roc_auc")]
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, mode='min'),
        # Save full Keras v3 model
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min', save_weights_only=False),
    ]
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks
    )
    return history

def get_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train iNat-tiny audio classifier")
    parser.add_argument('--data_path_train', type=str, required=True, help='Path to train dataset')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples per class for training (None for all)')
    parser.add_argument('--sample_rate', type=int, default=22050, help='Audio sample rate. Default is 22050 Hz.')
    parser.add_argument('--num_mels', type=int, default=64, help='Number of mel bins for spectrogram')
    parser.add_argument('--spec_width', type=int, default=128, help='Spectrogram width')
    parser.add_argument('--fft_length', type=int, default=512, help='FFT length for STFT/linear spectrogram')
    parser.add_argument('--chunk_duration', type=int, default=3, help='Audio chunk duration (seconds)')
    parser.add_argument('--max_duration', type=int, default=60, help='Max audio duration (seconds)')
    parser.add_argument('--audio_frontend', type=str, default='librosa',
                        choices=['precomputed', 'hybrid', 'raw', 'librosa', 'tf'],
                        help='Frontend: precomputed/librosa=melspec outside; hybrid=linear->fixed mel; raw/tf=STFT->fixed mel')
    parser.add_argument('--mag_scale', type=str, default='pcen',
                        choices=['pcen', 'pwl', 'none'],
                        help='Magnitude compression in frontend: pcen | pwl | none')
    parser.add_argument('--embeddings_size', type=int, default=512, help='Size of the final embeddings layer')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha for model scaling')
    parser.add_argument('--depth_multiplier', type=int, default=1, help='Depth multiplier for model')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup alpha')
    parser.add_argument('--mixup_probability', type=float, default=0.25, help='Fraction of batch to apply mixup')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/best_model.keras', help='Path to save best model (.keras)')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Load file paths and classes
    file_paths, classes = load_file_paths_from_directory(args.data_path_train, 
                                                         max_samples=250, #args.max_samples, 
                                                         classes=get_classes_with_most_samples(args.data_path_train, 25, False) # DEBUG: Only use 25 classes for debugging
                                                         )
    
    # Perform sanity check on the dataset
    dataset_sanity_check(
        file_paths, classes,
        sample_rate=args.sample_rate,
        max_duration=args.max_duration,
        chunk_duration=args.chunk_duration,
        spec_width=args.spec_width,
        mel_bins=args.num_mels,
        audio_frontend=args.audio_frontend,
        fft_length=args.fft_length,
        mag_scale=args.mag_scale
    )

    # Split dataset into training and validation sets
    val_split = args.val_split
    split_idx = int(len(file_paths) * (1 - val_split))
    train_paths = file_paths[:split_idx]
    val_paths = file_paths[split_idx:]
    print(f"Training on {len(train_paths)} files, validating on {len(val_paths)} files.")

    # Create training dataset (with mixup)
    train_dataset = load_dataset(
        train_paths, classes,
        audio_frontend=args.audio_frontend,
        sample_rate=args.sample_rate,
        max_duration=args.max_duration,
        chunk_duration=args.chunk_duration,
        spec_width=args.spec_width,
        batch_size=args.batch_size,
        mixup_alpha=args.mixup_alpha,
        mixup_probability=args.mixup_probability,
        mel_bins=args.num_mels,
        fft_length=args.fft_length,
        mag_scale=args.mag_scale
    )

    # Create validation dataset (without mixup)
    val_dataset = load_dataset(
        val_paths, classes,
        audio_frontend=args.audio_frontend,
        sample_rate=args.sample_rate,
        max_duration=args.max_duration,
        chunk_duration=args.chunk_duration,
        spec_width=args.spec_width,
        batch_size=args.batch_size,
        mixup_alpha=0.0,
        mixup_probability=0.0,
        mel_bins=args.num_mels,
        fft_length=args.fft_length,
        mag_scale=args.mag_scale
    )

    # Update steps_per_epoch and val_steps (robust)
    steps_per_epoch = max(1, math.ceil(len(train_paths) / float(args.batch_size)))
    val_steps = max(1, math.ceil(len(val_paths) / float(args.batch_size)))

    # Build model
    print("Building model...")
    model = build_dscnn_model(
        num_mels=args.num_mels,
        spec_width=args.spec_width,
        sample_rate=args.sample_rate,
        chunk_duration=args.chunk_duration,
        audio_frontend=args.audio_frontend,
        num_classes=len(classes),
        alpha=args.alpha,
        depth_multiplier=args.depth_multiplier,
        embeddings_size=args.embeddings_size,
        fft_length=args.fft_length,
        mag_scale=args.mag_scale
    )
    
    model.summary()
    print("Model built successfully.")

    # Train model
    print("Starting training...")
    history = train_model(
        model, train_dataset, val_dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint_path,
        steps_per_epoch=steps_per_epoch,
        val_steps=val_steps
    )
    print(f"Training complete. Best model saved to '{args.checkpoint_path}'.")

    # Save model config JSON next to the checkpoint
    cfg = {
        "sample_rate": args.sample_rate,
        "num_mels": args.num_mels,
        "spec_width": args.spec_width,
        "fft_length": args.fft_length,
        "chunk_duration": args.chunk_duration,
        "audio_frontend": args.audio_frontend,
        "mag_scale": args.mag_scale,
        "embeddings_size": args.embeddings_size,
        "alpha": args.alpha,
        "depth_multiplier": args.depth_multiplier,
        "num_classes": len(classes),
        "class_names": classes,
    }
    cfg_path = os.path.splitext(args.checkpoint_path)[0] + "_model_config.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved model config to '{cfg_path}'")

    # Save labels to txt file
    labels_file = args.checkpoint_path.replace('.h5', '_labels.txt')
    with open(labels_file, 'w') as f:
        for cls in classes:
            f.write(f"{cls}\n")

    # Post-training sanity check with trained TF frontend (mag_scale/BN) if applicable
    if args.audio_frontend in ('tf', 'raw', 'hybrid') or (args.audio_frontend in ('precomputed', 'librosa') and args.mag_scale != 'none'):
        print("Post-training sanity check using trained TF audio frontend...")
        trained_frontend = model.get_layer("audio_frontend")
        dataset_sanity_check(
            val_paths, classes,
            sample_rate=args.sample_rate,
            max_duration=args.max_duration,
            chunk_duration=args.chunk_duration,
            spec_width=args.spec_width,
            mel_bins=args.num_mels,
            audio_frontend=args.audio_frontend,
            fft_length=args.fft_length,
            mag_scale=args.mag_scale,
            tf_frontend_layer=trained_frontend
        )
