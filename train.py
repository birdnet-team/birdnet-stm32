import os
import argparse
import numpy as np
from typing import Optional
import tensorflow as tf
from tensorflow.keras import layers, constraints, regularizers
import math
import json
import librosa

# Supported audio filename extensions (lowercase)
SUPPORTED_AUDIO_EXTS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')

from utils.audio import (
    load_audio_file,
    get_spectrogram_from_audio,
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
    Plot spectrograms for a quick visual inspection before/after training.

    Behavior:
      - precomputed/librosa: computes mel + mag_scale offline and plots.
      - hybrid/raw(tf): runs the TF AudioFrontendLayer and plots its output.
      - if tf_frontend_layer is provided, it is used (trained params).

    Args:
        file_paths (list[str]): Dataset audio file paths.
        classes (list[str]): Ordered class names for labels.
        sample_rate (int): Audio sampling rate (Hz).
        max_duration (int): Max seconds to read from a file.
        chunk_duration (int): Chunk length (seconds) per sample.
        spec_width (int): Target spectrogram width (frames).
        mel_bins (int): Number of mel bins.
        audio_frontend (str): 'precomputed' | 'librosa' | 'hybrid' | 'raw' | 'tf'.
        tf_frontend_layer (AudioFrontendLayer | None): Trained frontend to use for plotting.
        fft_length (int): FFT length used for linear/hybrid paths.
        mag_scale (str): 'pcen' | 'pwl' | 'db' | 'none' magnitude scaling.

    Returns:
        None. Saves/plots example spectrograms to the samples/ folder.
    """
    os.makedirs("samples", exist_ok=True)
    np.random.shuffle(file_paths)

    tf_frontend = None
    if audio_frontend in ('hybrid', 'tf', 'raw'):
        # Use trained frontend if provided, otherwise build one configured with requested mag_scale
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
        fft_bins = fft_length // 2 + 1
        dummy_shape = (1, chunk_duration * sample_rate, 1) if audio_frontend in ('tf', 'raw') else (1, fft_bins, spec_width, 1)
        dummy = tf.zeros(dummy_shape, dtype=tf.float32)
        _ = tf_frontend(dummy, training=False)

    for i, path in enumerate(file_paths[:5]):
        audio_chunks = load_audio_file(path, sample_rate, max_duration, chunk_duration)
        if len(audio_chunks) == 0: continue

        if audio_frontend in ('librosa', 'precomputed'):
            # Precompute mel + requested mag_scale for visualization
            specs = [get_spectrogram_from_audio(chunk, sample_rate, n_fft=fft_length, mel_bins=mel_bins, spec_width=spec_width, mag_scale=mag_scale) for chunk in audio_chunks]
            pool = sort_by_s2n(specs, threshold=0.5) or specs
            if len(pool) == 0: continue
            spec = pick_random_samples(pool, num_samples=1)
            spec = spec[0] if isinstance(spec, list) else spec

        elif audio_frontend == 'hybrid':
            # Feed linear power to TF frontend (mel + mag_scale applied in TF layer)
            specs = [get_spectrogram_from_audio(chunk, sample_rate, n_fft=fft_length, mel_bins=-1, spec_width=spec_width) for chunk in audio_chunks]
            pool = sort_by_s2n(specs, threshold=0.5) or specs
            if len(pool) == 0: continue
            spec_in = pick_random_samples(pool, num_samples=1)
            spec_in = spec_in[0] if isinstance(spec_in, list) else spec_in  # [fft_bins, spec_width]
            inp = spec_in[np.newaxis, :, :, np.newaxis].astype(np.float32)   # [B,fft_bins,T,1]
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

        plot_spectrogram(spec, title=f"{audio_frontend}_{mag_scale}_{os.path.basename(path)}")
        
def get_classes_with_most_samples(directory, n_classes=25, include_noise=False, exts: tuple = SUPPORTED_AUDIO_EXTS):
    """
    Collect the most frequent class labels from a dataset root.

    Args:
        directory (str): Root dataset directory (class-subfolders).
        n_classes (int): Number of top classes to return.
        include_noise (bool): Include noise-like labels (noise/silence/background/other).
        exts (tuple[str, ...]): Accepted audio file extensions (case-insensitive).

    Returns:
        list[str]: Class names, sorted by descending sample count (length ≤ n_classes).
    """
    classes = {}  # 'class_name': num_samples
    noise_classes = {'noise', 'silence', 'background', 'other'}

    for root, dirs, files in os.walk(directory):
        for fname in files:
            if not fname.lower().endswith(exts):
                continue
            class_name = os.path.basename(root)
            if not include_noise and class_name.lower() in noise_classes:
                continue
            classes[class_name] = classes.get(class_name, 0) + 1

    sorted_classes = sorted(classes.items(), key=lambda x: x[1], reverse=True)
    return [cls for cls, _ in sorted_classes[:n_classes]]    

def load_file_paths_from_directory(directory, classes=None, max_samples=None, exts: tuple = SUPPORTED_AUDIO_EXTS):
    """
    Recursively gather audio files from a class-structured directory.

    Structure:
        root/
          class_a/*.(wav|mp3|flac|ogg|...)
          class_b/*.(wav|mp3|flac|ogg|...)
          ...

    Args:
        directory (str): Dataset root directory.
        classes (list[str] | None): If given, restrict to these class names.
        max_samples (int | None): Cap the number of files per class.
        exts (tuple[str, ...]): Accepted audio file extensions (case-insensitive).

    Returns:
        tuple[list[str], list[str]]:
            - file_paths: Shuffled list of absolute audio paths (capped per class if requested).
            - classes_sorted: Sorted class names found (noise-like names excluded).
    """
    per_class = {}

    # Walk and bucket files by their immediate parent directory (class)
    for root, _, files in tf.io.gfile.walk(directory):
        for fname in files:
            if not fname.lower().endswith(exts):
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

def upsample_minority_classes(file_paths, classes, ratio=0.25):
    
    """
    Upsample minority classes to match the largest class size through random repetition.
    Args:
        file_paths (list[str]): List of audio file paths.
        classes (list[str]): Ordered class names.
        ratio (float): Fraction of the largest class size to upsample to (0 < ratio ≤ 1).
    Returns:
        list[str]: Augmented list of audio file paths with upsampled minority classes.
    """
    
    assert 0 < ratio <= 1, "Ratio must be in (0, 1]."
    class_to_paths = {cls: [] for cls in classes}
    
    # Bucket file paths by class
    for path in file_paths:
        class_name = os.path.basename(os.path.dirname(path))
        if class_name in class_to_paths:
            class_to_paths[class_name].append(path)
    
    # Determine target size based on the largest class
    max_size = max(len(paths) for paths in class_to_paths.values())
    target_size = int(max_size * ratio)
    
    augmented_paths = []
    for cls, paths in class_to_paths.items():
        current_size = len(paths)
        if current_size < target_size:
            # Upsample by random repetition
            num_to_add = target_size - current_size
            additional_paths = np.random.choice(paths, size=num_to_add, replace=True).tolist()
            paths.extend(additional_paths)
        augmented_paths.extend(paths)
    
    np.random.shuffle(augmented_paths)
    return augmented_paths

def data_generator(file_paths, classes, batch_size=32, audio_frontend='librosa', sample_rate=22050, max_duration=30, chunk_duration=3, spec_width=128, mixup_alpha=0.2, mixup_probability=0.25, mel_bins=48, fft_length=512, mag_scale='none'):
    """
    Yield batches of (inputs, one_hot_labels) for training/validation.

    Frontends:
        - precomputed/librosa: mel spectrogram (mel_bins, spec_width, 1)
        - hybrid: linear magnitude STFT (fft_bins, spec_width, 1)
        - raw/tf: raw waveform (T, 1) where T = sample_rate*chunk_duration

    Selection:
        - Loads multiple chunks per file and picks one by SNR ranking.
        - Optional mixup applied to a subset of the batch.

    Args:
        file_paths (list[str]): Audio file paths.
        classes (list[str]): Ordered class names for one-hot encoding.
        batch_size (int): Batch size.
        audio_frontend (str): 'precomputed' | 'librosa' | 'hybrid' | 'raw' | 'tf'.
        sample_rate (int): Sampling rate (Hz).
        max_duration (int): Max duration to read per file (seconds).
        chunk_duration (int): Chunk duration (seconds).
        spec_width (int): Target spectrogram width (frames).
        mixup_alpha (float): Beta distribution alpha for mixup.
        mixup_probability (float): Fraction of batch to apply mixup to.
        mel_bins (int): Number of mel bins for mel spectrograms.
        fft_length (int): FFT size used by librosa/hybrid paths.
        mag_scale (str): 'pcen' | 'pwl' | 'db' | 'none' magnitude scaling.

    Yields:
        tuple[np.ndarray, np.ndarray]: (inputs, labels) for a batch.
    """
    T = sample_rate * chunk_duration
    while True:
        idxs = np.random.permutation(len(file_paths))
        for batch_start in range(0, len(idxs), batch_size):
            batch_idxs = idxs[batch_start:batch_start + batch_size]
            batch_samples, batch_labels = [], []
            for idx in batch_idxs:
                path = file_paths[idx]
                label_str = path.split('/')[-2]
                audio_chunks = load_audio_file(path, sample_rate, max_duration, chunk_duration, random_offset=True)
                if len(audio_chunks) == 0:
                    # Use random noise of chunk_duration if file loading failed
                    audio_chunks = [np.random.uniform(-1.0, 1.0, size=(T,)).astype(np.float32)]
                    label_str = 'noise'

                if audio_frontend in ('librosa', 'precomputed'):
                    specs = [get_spectrogram_from_audio(chunk, sample_rate, n_fft=fft_length, mel_bins=mel_bins, spec_width=spec_width, mag_scale=mag_scale) for chunk in audio_chunks]
                    pool = sort_by_s2n(specs, threshold=0.5) or specs
                    if len(pool) == 0: continue
                    sample = pick_random_samples(pool, num_samples=1)
                    sample = sample[0] if isinstance(sample, list) else sample   # [mel, T]
                    need_ch_last = True

                elif audio_frontend == 'hybrid':
                    specs = [get_spectrogram_from_audio(chunk, sample_rate, n_fft=fft_length, mel_bins=-1, spec_width=spec_width) for chunk in audio_chunks]
                    pool = sort_by_s2n(specs, threshold=0.5) or specs
                    if len(pool) == 0: continue
                    sample = pick_random_samples(pool, num_samples=1)
                    sample = sample[0] if isinstance(sample, list) else sample   # [fft_bins, T]
                    need_ch_last = True

                elif audio_frontend in ('tf', 'raw'):
                    pool = sort_by_s2n(audio_chunks, threshold=0.5) or audio_chunks
                    if len(pool) == 0: continue
                    sample = pick_random_samples(pool, num_samples=1)
                    x = sample[0] if isinstance(sample, list) else sample
                    x = x[:T]
                    if x.shape[0] < T: x = np.pad(x, (0, T - x.shape[0]))
                    sample = x                                           # [T]
                    need_ch_last = True
                else:
                    raise ValueError("Invalid audio frontend. Choose 'precomputed', 'hybrid', or 'raw'.")

                # Make one-hot label; noise-like labels get all-zero vector
                if label_str.lower() in ['noise', 'silence', 'background', 'other']:
                    one_hot_label = np.zeros(len(classes), dtype=np.float32)
                else:
                    if label_str not in classes: continue
                    one_hot_label = tf.one_hot(classes.index(label_str), depth=len(classes)).numpy()

                if need_ch_last:
                    # Add trailing channel dim
                    sample = np.expand_dims(sample, axis=-1)

                batch_samples.append(sample.astype(np.float32))
                batch_labels.append(one_hot_label.astype(np.float32))

            if len(batch_samples) == 0: continue
            batch_samples = np.stack(batch_samples)
            batch_labels = np.stack(batch_labels)

            # Mixup
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
    Wrap a Python generator as a tf.data.Dataset with static shapes.

    Args:
        file_paths (list[str]): Audio file paths.
        classes (list[str]): Ordered class names for one-hot encoding.
        audio_frontend (str): 'precomputed' | 'librosa' | 'hybrid' | 'raw' | 'tf'.
        batch_size (int): Batch size.
        spec_width (int): Target spectrogram width (frames).
        mel_bins (int): Number of mel bins.
        **kwargs: sample_rate (int), chunk_duration (int), fft_length (int),
                  mag_scale (str: 'pcen' | 'pwl' | 'db' | 'none'), max_duration (int),
                  mixup_alpha (float), mixup_probability (float).

    Returns:
        tf.data.Dataset: Infinite dataset of (inputs, labels), prefetching enabled.
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
        # [B, fft_bins, T, 1]
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
    Audio frontend with interchangeable input modes.

    Modes:
        - precomputed: Input mel spectrogram [B, mel_bins, spec_width, 1].
        - hybrid:      Linear STFT bins [B, fft_bins, spec_width, 1] -> mel mixer.
        - raw:         Waveform [B, T, 1] -> fold -> window -> DFT (1x1 convs) -> mel.

    Magnitude scaling:
        - 'none': Pass-through.
        - 'pwl':  Piecewise-linear compression (1x1 DW convs, ReLU, Add).
        - 'pcen': PCEN-like compression (pool/conv/ReLU/Add), linear magnitude domain.
        - 'db':   Log compression (10·log10 by default) after mel (be aware: does not quantize well).

    Notes:
        - Slaney mel basis from librosa is used to seed mel_mixer for parity.
        - Hybrid/raw paths feed magnitude to mel to match librosa (power=1.0).
        - Layer is NPU-friendly (STM32N6) by construction.
    """
    def __init__(
        self,
        mode: str,
        mel_bins: int,
        spec_width: int,
        sample_rate: int,
        chunk_duration: int,
        fft_length: int = 512,
        pcen_K: int = 4,
        init_mel: bool = True,
        mel_fmin: float = 150.0,
        mel_fmax: Optional[float] = None,
        mel_norm: str = "slaney",
        mag_scale: str = "none",
        name: str = "audio_frontend",
        is_trainable: bool = False,
        **kwargs):
        super().__init__(name=name, **kwargs)
        assert mode in ("precomputed", "hybrid", "raw")
        assert mag_scale in ("pcen", "pwl", "db", "none")
        self.mode = mode
        self.mel_bins = int(mel_bins)
        self.spec_width = int(spec_width)
        self.sample_rate = int(sample_rate)
        self.chunk_duration = int(chunk_duration)
        self.fft_length = int(fft_length)
        self.pcen_K = int(pcen_K)
        self.init_mel = bool(init_mel)
        self.mel_fmin = float(mel_fmin)
        self.mel_fmax = mel_fmax
        self.mel_norm = mel_norm
        self.mag_scale = mag_scale
        self.is_trainable = bool(is_trainable)

        # DB params
        self.db_eps = 1e-6
        self.db_ref = 1.0

        # stride targeting spec_width (no external pad ops)
        self._T = int(self.sample_rate) * int(self.chunk_duration)
        self._frame_step = max(1, int(math.ceil(self._T / float(self.spec_width))))

        # Shared mel mixer (weights depend on mode-specific Cin; set in build)
        self.mel_mixer = layers.Conv2D(
            filters=int(self.mel_bins),
            kernel_size=(1, 1),
            padding="same",
            use_bias=False,
            kernel_constraint=constraints.NonNeg(),
            name=f"{name}_mel_mixer",
            trainable=False,#self.is_trainable,
        )

        # RAW: fold waveform -> window -> DFT via 1x1 convs -> mel_mixer
        if self.mode == "raw":
            self.frame_len = max(8, self._T // max(1, self.spec_width))
            self.fft_bins_raw = self.frame_len // 2 + 1
            self._frame_pad_in = (8 - (self.frame_len % 8)) % 8

            self.v2d_window_dw = layers.DepthwiseConv2D(
                kernel_size=(1, 1),
                depth_multiplier=1,
                padding="same",
                use_bias=False,
                name=f"{name}_v2d_window_dw",
                trainable=self.is_trainable,
            )
            # DFT across channelized frame samples
            self.v2d_dft_real = layers.Conv2D(
                filters=int(self.fft_bins_raw),
                kernel_size=(1, 1),
                padding="same",
                use_bias=False,
                name=f"{name}_v2d_dft_real",
                trainable=self.is_trainable,
            )
            self.v2d_dft_imag = layers.Conv2D(
                filters=int(self.fft_bins_raw),
                kernel_size=(1, 1),
                padding="same",
                use_bias=False,
                name=f"{name}_v2d_dft_imag",
                trainable=self.is_trainable,
            )
        else:
            self.frame_len = None
            self.fft_bins_raw = None
            self._frame_pad_in = 0
            self.v2d_window_dw = None
            self.v2d_dft_real = None
            self.v2d_dft_imag = None

        # PCEN/PWL sublayers (respect is_trainable)
        if self.mag_scale == "pcen":
            self._pcen_pools = [layers.AveragePooling2D(pool_size=(1, 3), strides=(1, 1), padding="same",
                                                        name=f"{name}_pcen_ema{k}") for k in range(self.pcen_K)]
            self._pcen_agc_dw = layers.DepthwiseConv2D((1, 1), use_bias=False,
                                                    depthwise_initializer=tf.keras.initializers.Constant(0.6),
                                                    padding="same", name=f"{name}_pcen_agc_dw", trainable=self.is_trainable)
            self._pcen_k1_dw = layers.DepthwiseConv2D((1, 1), use_bias=False,
                                                    depthwise_initializer=tf.keras.initializers.Constant(0.15),
                                                    padding="same", name=f"{name}_pcen_k1_dw", trainable=self.is_trainable)
            self._pcen_shift_dw = layers.DepthwiseConv2D((1, 1), use_bias=True,
                                                        depthwise_initializer=tf.keras.initializers.Ones(),
                                                        bias_initializer=tf.keras.initializers.Constant(-0.2),
                                                        padding="same", name=f"{name}_pcen_shift_dw", trainable=self.is_trainable)
            self._pcen_k2mk1_dw = layers.DepthwiseConv2D((1, 1), use_bias=False,
                                                        depthwise_initializer=tf.keras.initializers.Constant(0.45),
                                                        padding="same", name=f"{name}_pcen_k2mk1_dw", trainable=self.is_trainable)

        if self.mag_scale == "pwl":
            self._pwl_k0_dw = layers.DepthwiseConv2D((1, 1), use_bias=False,
                                                     depthwise_initializer=tf.keras.initializers.Constant(0.40),
                                                     padding="same", name=f"{name}_pwl_k0_dw", trainable=self.is_trainable)
            self._pwl_shift_dws = [
                layers.DepthwiseConv2D((1, 1), use_bias=True,
                                       depthwise_initializer=tf.keras.initializers.Ones(),
                                       bias_initializer=tf.keras.initializers.Constant(-t),
                                       padding="same", name=f"{name}_pwl_shift{i+1}_dw", trainable=self.is_trainable)
                for i, t in enumerate((0.10, 0.35, 0.65))
            ]
            self._pwl_k_dws = [
                layers.DepthwiseConv2D((1, 1), use_bias=False,
                                       depthwise_initializer=tf.keras.initializers.Constant(k),
                                       padding="same", name=f"{name}_pwl_k{i+1}_dw", trainable=self.is_trainable)
                for i, k in enumerate((0.25, 0.15, 0.08))
            ]
        else:
            self._pwl_k0_dw = None
            self._pwl_shift_dws = []
            self._pwl_k_dws = []

    def build(self, input_shape):
        """
        Build sub-layers and set fixed kernels (mel basis, window, DFT).

        Args:
            input_shape (tf.TensorShape): Keras-provided input shape.
        """
        # Hybrid: mel mixer with n_fft=self.fft_length
        if self.mode == "hybrid":
            fft_bins = self.fft_length // 2 + 1
            self._build_and_set_mel_mixer(n_fft=self.fft_length, cin=fft_bins)

        # Raw: window/DFT for frame_len, then mel mixer with n_fft=frame_len
        elif self.mode == "raw":
            in_ch = self.frame_len + self._frame_pad_in
            self._build_raw_window_and_dft(in_ch=in_ch)
            self._build_and_set_mel_mixer(n_fft=self.frame_len, cin=int(self.fft_bins_raw))

        # Build mag-scale layers as needed
        self._build_mag_layers()

        super().build(input_shape)

    # Helper: build mel mixer with a Slaney mel basis for a given n_fft and input channels
    def _build_and_set_mel_mixer(self, n_fft: int, cin: int):
        """
        Initialize mel_mixer with a Slaney mel basis for a given n_fft.

        Args:
            n_fft (int): FFT length used to compute the mel basis.
            cin (int): Number of input channels (fft_bins = n_fft//2 + 1).
        """
        upper = int(self.mel_fmax) if self.mel_fmax is not None else (self.sample_rate // 2)
        mel_mat = librosa.filters.mel(
            sr=int(self.sample_rate), n_fft=int(n_fft),
            n_mels=int(self.mel_bins), fmin=float(self.mel_fmin), fmax=float(upper),
            htk=False, norm='slaney'
        ).T.astype(np.float32)  # [cin, mel] where cin = n_fft//2+1
        pad = (8 - (cin % 8)) % 8
        if pad:
            mel_mat = np.pad(mel_mat, ((0, pad), (0, 0)), mode='constant')
        mel_kernel = mel_mat[None, None, :, :]  # [1,1,cin(+pad),mel]
        if not self.mel_mixer.built:
            self.mel_mixer.build(tf.TensorShape([None, 1, None, cin + pad]))
        self.mel_mixer.set_weights([mel_kernel])
        # Remember pad for call() path
        self._pad_ch_in = pad

    # Helper: build and init raw window and DFT convs (length=self.frame_len)
    def _build_raw_window_and_dft(self, in_ch: int):
        """
        Initialize raw frontend window (Hann) and DFT 1x1 conv kernels.

        Args:
            in_ch (int): Channelized frame length including padding.
        """
        # Window DWConv
        if not self.v2d_window_dw.built:
            self.v2d_window_dw.build(tf.TensorShape([None, 1, None, in_ch]))
        n = np.arange(self.frame_len, dtype=np.float32)
        hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / max(1, (self.frame_len - 1)))
        if self._frame_pad_in:
            hann = np.concatenate([hann, np.zeros((self._frame_pad_in,), dtype=np.float32)], axis=0)
        self.v2d_window_dw.set_weights([hann.reshape(1, 1, in_ch, 1).astype(np.float32)])

        # DFT convs sized for frame_len
        if not self.v2d_dft_real.built:
            self.v2d_dft_real.build(tf.TensorShape([None, 1, None, in_ch]))
        if not self.v2d_dft_imag.built:
            self.v2d_dft_imag.build(tf.TensorShape([None, 1, None, in_ch]))
        K = int(self.fft_bins_raw)
        n_idx = np.arange(in_ch, dtype=np.float32)[:, None]  # [in_ch,1]
        k_idx = np.arange(K, dtype=np.float32)[None, :]      # [1,K]
        ang = 2.0 * np.pi * (n_idx / max(1.0, float(self.frame_len))) @ k_idx  # [in_ch,K]
        valid = (n_idx < self.frame_len).astype(np.float32)
        real_kernel = (np.cos(ang) * valid)[None, None, :, :].astype(np.float32)
        imag_kernel = (-np.sin(ang) * valid)[None, None, :, :].astype(np.float32)
        self.v2d_dft_real.set_weights([real_kernel])
        self.v2d_dft_imag.set_weights([imag_kernel])

    # Helper: build mag-scale layers (PCEN/PWL)
    def _build_mag_layers(self):
        """
        Build magnitude scaling sub-layers as needed (PCEN/PWL/DB).

        Returns:
            None.
        """
        post_mel_shape = tf.TensorShape([None, 1, None, int(self.mel_bins)])
        if self.mag_scale == "pcen":
            for pool in self._pcen_pools:
                if not pool.built:
                    pool.build(post_mel_shape)
            for dw in (self._pcen_agc_dw, self._pcen_k1_dw, self._pcen_shift_dw, self._pcen_k2mk1_dw):
                if dw is not None and not dw.built:
                    dw.build(post_mel_shape)
        if self.mag_scale == "pwl":
            if self._pwl_k0_dw is not None and not self._pwl_k0_dw.built:
                self._pwl_k0_dw.build(post_mel_shape)
            for s in self._pwl_shift_dws:
                if not s.built:
                    s.build(post_mel_shape)
            for k in self._pwl_k_dws:
                if not k.built:
                    k.build(post_mel_shape)

    # Magnitude scaling implementations (N6-friendly: only conv/pool/relu/add ops)
    def _apply_pcen(self, x):
        """
        PCEN-like compression using only pool/conv/relu/add ops.
        x: [B, 1, T, C]
        """
        if not self._pcen_pools:
            return x
        m = x
        for pool in self._pcen_pools:
            m = pool(m)
        agc = self._pcen_agc_dw(m) if self._pcen_agc_dw is not None else m
        y0 = tf.nn.relu(x - agc)
        b1 = self._pcen_k1_dw(y0) if self._pcen_k1_dw is not None else y0
        y_shift = self._pcen_shift_dw(y0) if self._pcen_shift_dw is not None else y0
        relu = tf.nn.relu(y_shift)
        b2 = self._pcen_k2mk1_dw(relu) if self._pcen_k2mk1_dw is not None else relu
        y = b1 + b2
        return tf.nn.relu(y)

    # per-sample min-max normalization helper (to [0,1])
    def _minmax_norm(self, x):
        """
        Per-sample min–max normalization to [0, 1].

        Args:
            x (tf.Tensor): Input tensor [B, 1, T, C].

        Returns:
            tf.Tensor: Normalized tensor [B, 1, T, C].
        """
        # x is [B,1,T,C]
        x_min = tf.reduce_min(x, axis=[1, 2, 3], keepdims=True)
        x_max = tf.reduce_max(x, axis=[1, 2, 3], keepdims=True)
        return (x - x_min) / (x_max - x_min + 1e-6)

    def _apply_pwl(self, x):
        """
        Apply piecewise-linear compression via 1x1 DW conv branches.

        Args:
            x (tf.Tensor): Power-like input [B, 1, T, C].

        Returns:
            tf.Tensor: Compressed output [B, 1, T, C].
        """
        
        branches = []
        if self._pwl_k0_dw is not None:
            branches.append(self._pwl_k0_dw(x))
        for i, (shift_dw, k_dw) in enumerate(zip(self._pwl_shift_dws, self._pwl_k_dws), start=1):
            relu = tf.nn.relu(shift_dw(x))
            branches.append(k_dw(relu))
        if not branches:
            return x
        y = branches[0]
        for j, b in enumerate(branches[1:], start=1):
            y = tf.add(y, b, name=f"{self.name}_pwl_add_{j}")
        return y

    def _apply_db(self, x: tf.Tensor) -> tf.Tensor:
        """
        Apply dB log compression.

        Args:
            x: [B, 1, T, C] linear input.

        Returns:
            [B, 1, T, C] dB-scaled output (not normalized).
        """
        eps = tf.cast(self.db_eps, x.dtype)
        ref = tf.cast(self.db_ref, x.dtype)
        log10 = tf.math.log(tf.cast(10.0, x.dtype))
        safe = tf.maximum(x, eps)
        y = tf.math.log(safe / ref) / log10
        scale = tf.cast(10.0, x.dtype)
        return scale * y

    def _apply_mag(self, x):
        """
        Dispatch to the selected magnitude scaling ('pcen' | 'pwl' | 'db' | 'none').

        Args:
            x (tf.Tensor): Power-like input [B, 1, T, C].

        Returns:
            tf.Tensor: Output after the selected magnitude transform.
        """
        if self.mag_scale == "pcen":
            return self._apply_pcen(x)
        if self.mag_scale == "pwl":
            return self._apply_pwl(x)
        if self.mag_scale == "db":
            return self._apply_db(x)
        return x

    def call(self, inputs, training=None):
        """
        Run the selected frontend path and return a fixed-size spectrogram.

        Shapes:
            - precomputed: in [B, mel_bins, T, 1] -> out [B, mel_bins, spec_width, 1]
            - hybrid:      in [B, fft_bins, T, 1] -> out [B, mel_bins, spec_width, 1]
            - raw:         in [B, T, 1]           -> out [B, mel_bins, spec_width, 1]

        Args:
            inputs (tf.Tensor): Frontend input as per mode.
            training (bool | None): Keras training flag (unused).

        Returns:
            tf.Tensor: [B, mel_bins, spec_width, 1] spectrogram-like tensor.
        """
        # precomputed: [B, mel, T, 1]
        if self.mode == "precomputed":
            y = inputs[:, :, :self.spec_width, :]
            # Nothing to do because we assume precomputed has mel and mag_scale applied
            return y

        # hybrid: [B, fft_bins, T, 1] -> [B,1,T,fft_bins]
        if self.mode == "hybrid":
            fft_bins = self.fft_length // 2 + 1
            if inputs.shape.rank != 4 or (inputs.shape[1] is not None and int(inputs.shape[1]) != fft_bins):
                raise ValueError(f"Hybrid expects [B,{fft_bins},T,1], got {inputs.shape}")
            y = tf.transpose(inputs, [0, 3, 2, 1])  # [B,1,T,fft_bins]
            y = y[:, :, :self.spec_width, :]
            if self._pad_ch_in:
                b = tf.shape(y)[0]; t = tf.shape(y)[2]
                z = tf.zeros([b, 1, t, self._pad_ch_in], dtype=y.dtype)
                y = tf.concat([y, z], axis=-1)
            y = self.mel_mixer(y)                   # [B,1,T,mel]
            y = tf.nn.relu(y)
            y = self._apply_mag(y)
            #y = self._minmax_norm(y)
            y = tf.transpose(y, [0, 3, 2, 1])       # [B,mel,T,1]
            return y[:, :, :self.spec_width, :]

        # raw (DFT -> magnitude -> mel)
        x = inputs  # [B,T,1]
        B = tf.shape(x)[0]
        L = self.frame_len * self.spec_width
        x = x[:, :L, :]                                              # [B,L,1]
        y = tf.reshape(x, [B, self.spec_width, self.frame_len, 1])   # [B,T,C,1]
        y = tf.transpose(y, [0, 3, 1, 2])                            # [B,1,T,C]
        if self._frame_pad_in:
            t = tf.shape(y)[2]
            z = tf.zeros([B, 1, t, self._frame_pad_in], dtype=y.dtype)
            y = tf.concat([y, z], axis=-1)                           # [B,1,T,C+pad]
        y = self.v2d_window_dw(y)
        real = self.v2d_dft_real(y)                                  # [B,1,T,fft_bins_raw]
        imag = self.v2d_dft_imag(y)                                  # [B,1,T,fft_bins_raw]
        sqr = tf.add(tf.multiply(real, real), tf.multiply(imag, imag), name=f"{self.name}_raw_power")
        feat = tf.sqrt(tf.maximum(sqr, 0.0) + 1e-8)
        if self._pad_ch_in:
            t = tf.shape(feat)[2]
            z = tf.zeros([B, 1, t, self._pad_ch_in], dtype=feat.dtype)
            feat = tf.concat([feat, z], axis=-1)                     # [B,1,T,fft_bins_raw+pad]
        y = self.mel_mixer(feat)                                     # [B,1,T,mel]
        y = tf.nn.relu(y)
        y = self._apply_mag(y)
        #y = self._minmax_norm(y)
        y = tf.transpose(y, [0, 3, 2, 1])                            # [B,mel,T,1]
        return y[:, :, :self.spec_width, :]

    def compute_output_shape(self, input_shape):
        """
        Keras shape inference for the layer.

        Args:
            input_shape (tuple): Input shape tuple.

        Returns:
            tuple: (batch, mel_bins, spec_width, 1)
        """
        return (input_shape[0], int(self.mel_bins), int(self.spec_width), 1)
    
    def get_config(self):
        """
        Return the layer config for serialization.

        Returns:
            dict: Serializable configuration of the frontend layer.
        """
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

def _make_divisible(v, divisor=8):
    v = int(v + divisor / 2) // divisor * divisor
    return max(divisor, v)

def ds_conv_block(x, out_ch, stride_f=1, stride_t=1, name="ds", weight_decay=1e-4, drop_rate=0.1):
    """
    Depthwise-separable block (3x3 DW + 1x1 PW) with optional residual.

    Args:
        x (tf.Tensor): Input tensor [B, H, W, C].
        out_ch (int): Output channels for pointwise conv.
        stride_f (int): Stride along frequency axis (height).
        stride_t (int): Stride along time axis (width).
        name (str): Base name for layers.
        weight_decay (float): L2 kernel regularization (weight decay).
        drop_rate (float): Spatial dropout rate applied to the PW output (0..1).

    Returns:
        tf.Tensor: Output tensor after DW/PW + BN/ReLU (+ residual if aligned).
    """
    reg = regularizers.l2(weight_decay) if weight_decay and weight_decay > 0 else None
    in_ch = x.shape[-1]
    y = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=(stride_f, stride_t),
        padding='same',
        use_bias=False,
        depthwise_regularizer=reg,
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
        kernel_regularizer=reg,
        name=f"{name}_pw",
    )(y)
    y = layers.BatchNormalization(name=f"{name}_pw_bn")(y)

    # Optional spatial dropout
    if drop_rate and drop_rate > 0:
        y = layers.SpatialDropout2D(drop_rate, name=f"{name}_drop")(y)

    if (stride_f == 1 and stride_t == 1) and (in_ch is not None and int(in_ch) == int(out_ch)):
        y = layers.Add(name=f"{name}_add")([x, y])

    y = layers.ReLU(max_value=6, name=f"{name}_pw_relu")(y)
    return y

def build_dscnn_model(num_mels, spec_width, sample_rate, chunk_duration, embeddings_size, num_classes,
                      audio_frontend='precomputed', alpha=1.0, depth_multiplier=1, fft_length=512,
                      mag_scale='none', frontend_trainable=False):
    """
    Build a DS-CNN model with a selectable audio frontend.

    Args:
        num_mels (int): Number of mel bins in the spectrogram.
        spec_width (int): Spectrogram width (frames).
        sample_rate (int): Sampling rate (Hz).
        chunk_duration (int): Chunk duration (seconds).
        embeddings_size (int): Channels in the embeddings layer.
        num_classes (int): Number of output classes.
        audio_frontend (str): 'precomputed' | 'librosa' | 'hybrid' | 'raw' | 'tf'.
        alpha (float): Width multiplier for the backbone.
        depth_multiplier (int): Repeats multiplier for DS blocks.
        fft_length (int): FFT size for hybrid/librosa paths.
        mag_scale (str): 'pcen' | 'pwl' | 'db' | 'none' magnitude scaling.
        frontend_trainable (bool): Trainability of frontend sub-layers.

    Returns:
        tf.keras.Model: Compiled DS-CNN model.
    """
    # Enforce STM32N6 constraint if building a raw/tf frontend
    if audio_frontend in ('tf', 'raw'):
        T = int(sample_rate) * int(chunk_duration)
        if T >= (1 << 16):
            raise ValueError(
                f"STM32N6 constraint: raw input length (sample_rate*chunk_duration={T}) must be < 65536.\n"
                f"Use one of:\n"
                f"  - --sample_rate 16000 (3s => 48000)\n"
                f"  - --chunk_duration 2 (2*22050=44100)\n"
                f"  - or --audio_frontend hybrid/precomputed for deployment."
            )

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
            is_trainable=frontend_trainable,
            name="audio_frontend",
        )(inputs)
    elif audio_frontend == 'hybrid':
        fft_bins = fft_length // 2 + 1
        # [B, fft_bins, T, 1]
        inputs = tf.keras.Input(shape=(fft_bins, spec_width, 1), name='linear_spectrogram_input')
        x = AudioFrontendLayer(
            mode='hybrid',
            mel_bins=num_mels,
            spec_width=spec_width,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            fft_length=fft_length,
            mag_scale=mag_scale,
            is_trainable=frontend_trainable,
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
            is_trainable=frontend_trainable,
            name="audio_frontend",
        )(inputs)
    else:
        raise ValueError("Invalid audio_frontend.")

    # Stem (3x3, stride 1) to lift channels
    stem_ch = _make_divisible(int(16 * alpha), 8)
    x = layers.Conv2D(stem_ch, (3, 3), strides=(1, 2), padding='same', use_bias=False, name="stem_conv")(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(max_value=6, name="stem_relu")(x)

    # Stages: (filters, repeats, (stride_f, stride_t))
    base_filters = [32, 64, 128, 256]
    base_repeats = [2, 3, 4, 2]
    base_strides = [(2, 2), (2, 2), (2, 2), (2, 2)]

    for si, (bf, br, (sf, st)) in enumerate(zip(base_filters, base_repeats, base_strides), start=1):
        out_ch = _make_divisible(int(bf * alpha), 8)
        reps = max(1, int(math.ceil(br * depth_multiplier)))
        # First block in stage downsamples both frequency and time
        x = ds_conv_block(x, out_ch, stride_f=sf, stride_t=st, name=f"stage{si}_ds1")
        for bi in range(2, reps + 1):
            x = ds_conv_block(x, out_ch, stride_f=1, stride_t=1, name=f"stage{si}_ds{bi}")
            
    # Final 1x1 conv to embeddings
    emb_ch = _make_divisible(int(embeddings_size), 8)
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

def train_model(model, train_dataset, val_dataset, epochs=50, learning_rate=0.001, batch_size=64, patience=10, checkpoint_path="checkpoints/best_model.keras", steps_per_epoch=None, val_steps=None):
    """
    Train the model with cosine decay, early stopping, and checkpointing.

    Metrics:
        - ROC AUC (multi-label)

    Args:
        model (tf.keras.Model): Model to train.
        train_dataset (tf.data.Dataset): Training dataset (infinite).
        val_dataset (tf.data.Dataset): Validation dataset (infinite).
        epochs (int): Number of epochs.
        learning_rate (float): Initial learning rate for cosine schedule.
        batch_size (int): Unused; kept for API symmetry.
        patience (int): Early stopping patience (epochs).
        checkpoint_path (str): Path to save the best .keras model.
        steps_per_epoch (int): Training steps per epoch (> 0 required).
        val_steps (int | None): Validation steps per epoch.

    Returns:
        tf.keras.callbacks.History: Keras training history.
    """

    if steps_per_epoch is None or steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be > 0")
    if val_steps is None or val_steps <= 0:
        val_steps = 1

    # Ensure checkpoint dir exists
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=epochs * steps_per_epoch,
        alpha=0.0
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

def compute_hop_length(sample_rate: int, chunk_duration: int, spec_width: int) -> int:
    """
    Compute hop length to produce spec_width frames from a chunk.

    Args:
        sample_rate (int): Sampling rate (Hz).
        chunk_duration (int): Chunk duration (seconds).
        spec_width (int): Desired number of frames.

    Returns:
        int: hop_length in samples (floor(T/spec_width), at least 1).
    """
    T = int(sample_rate) * int(chunk_duration)
    return max(1, T // int(spec_width))

def get_args():
    """
    Parse command-line arguments for training.

    Returns:
        argparse.Namespace: Parsed arguments (see --help for details).
    """
    parser = argparse.ArgumentParser(description="Train STM32N6 audio classifier")
    parser.add_argument('--data_path_train', type=str, required=True, help='Path to train dataset')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples per class for training (None for all)')
    parser.add_argument('--upsample_ratio', type=float, default=0.25, help='Upsample ratio for minority classes')
    parser.add_argument('--sample_rate', type=int, default=22050, help='Audio sample rate. Default is 22050 Hz.')
    parser.add_argument('--num_mels', type=int, default=64, help='Number of mel bins for spectrogram')
    parser.add_argument('--spec_width', type=int, default=256, help='Spectrogram width')
    parser.add_argument('--fft_length', type=int, default=512, help='FFT length for STFT/linear spectrogram')
    parser.add_argument('--chunk_duration', type=int, default=3, help='Audio chunk duration (seconds)')
    parser.add_argument('--max_duration', type=int, default=30, help='Max audio duration (seconds)')
    parser.add_argument('--audio_frontend', type=str, default='hybrid',
                        choices=['precomputed', 'hybrid', 'raw', 'librosa', 'tf'],
                        help='Frontend: precomputed/librosa=melspec outside; hybrid=linear->fixed mel; raw/tf=STFT->fixed mel')
    parser.add_argument('--mag_scale', type=str, default='pwl',
                        choices=['pcen', 'pwl', 'db', 'none'],
                        help='Magnitude compression in frontend: pcen | pwl | db | none')
    parser.add_argument('--embeddings_size', type=int, default=256, help='Size of the final embeddings layer')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha for model scaling')
    parser.add_argument('--depth_multiplier', type=int, default=1, help='Depth multiplier for model')
    parser.add_argument('--frontend_trainable', action='store_true', default=True, help='If set, make audio frontend trainable (mel_mixer/raw mixer/PCEN/PWL).')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup alpha')
    parser.add_argument('--mixup_probability', type=float, default=0.25, help='Fraction of batch to apply mixup')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/best_model.keras', help='Path to save best model (.keras)')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    # Early warning for STM32N6 when using raw/tf frontend
    if args.audio_frontend in ('tf', 'raw'):
        T = int(args.sample_rate) * int(args.chunk_duration)
        if T >= (1 << 16):
            print(f"[WARN] STM32N6 compile will fail: raw input length {T} >= 65536.")
            print("       Use --sample_rate 16000 or --chunk_duration 2, or switch to --audio_frontend hybrid/precomputed.")
    
    # Compute hop length once for config/diagnostics
    hop_length = compute_hop_length(args.sample_rate, args.chunk_duration, args.spec_width)

    # Load file paths and classes
    file_paths, classes = load_file_paths_from_directory(args.data_path_train, 
                                                         max_samples=args.max_samples, 
                                                         #classes=get_classes_with_most_samples(args.data_path_train, 25, False) # DEBUG: Only use 25 classes for debugging
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
    
    # Upsample classes if requested
    if args.upsample_ratio and args.upsample_ratio > 0 and args.upsample_ratio < 1.0:
        train_paths = upsample_minority_classes(train_paths, classes, args.upsample_ratio)
        print(f"After upsampling, training on {len(train_paths)} files.")

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

    # Update steps_per_epoch and val_steps
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
        mag_scale=args.mag_scale,
        frontend_trainable=args.frontend_trainable,
    )
    
    model.summary()
    print("Model built successfully.")
    
    # Save model config JSON next to the checkpoint
    cfg = {
        "sample_rate": args.sample_rate,
        "num_mels": args.num_mels,
        "spec_width": args.spec_width,
        "fft_length": args.fft_length,
        "chunk_duration": args.chunk_duration,
        "hop_length": hop_length, 
        "audio_frontend": args.audio_frontend,
        "mag_scale": args.mag_scale,
        "embeddings_size": args.embeddings_size,
        "alpha": args.alpha,
        "depth_multiplier": args.depth_multiplier,
        "num_classes": len(classes),
        "class_names": classes,
        "frontend_trainable": args.frontend_trainable,
    }
    cfg_path = os.path.splitext(args.checkpoint_path)[0] + "_model_config.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved model config to '{cfg_path}'")

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

    # Save labels to txt file
    labels_file = args.checkpoint_path.replace('.keras', '_labels.txt')
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
