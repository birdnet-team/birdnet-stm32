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
    sort_by_activity,
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
    
def dataset_sanity_check(file_paths, 
                         classes, 
                         sample_rate=22050, 
                         max_duration=30, 
                         chunk_duration=3, 
                         spec_width=128, 
                         mel_bins=64, 
                         audio_frontend='precomputed', 
                         tf_frontend_layer=None, 
                         fft_length=512, 
                         mag_scale='pwl',
                         snr_threshold=0.05,
                         prefix='pre'):
    """
    Plot spectrograms for a quick visual inspection before/after training.

    Behavior:
      - precomputed/librosa: computes mel + mag_scale offline and plots.
      - hybrid/raw(tf): runs the TF AudioFrontendLayer and plots its output.
      - If tf_frontend_layer is provided, it is used (e.g., trained frontend).

    Args:
        file_paths (list[str]): Dataset audio file paths.
        classes (list[str]): Ordered class names for labels (unused here, kept for API parity).
        sample_rate (int): Audio sampling rate (Hz).
        max_duration (int): Max seconds to read from a file.
        chunk_duration (int): Chunk length (seconds) per sample.
        spec_width (int): Target spectrogram width (frames).
        mel_bins (int): Number of mel bins.
        audio_frontend (str): 'precomputed' | 'librosa' | 'hybrid' | 'raw' | 'tf'.
        tf_frontend_layer (AudioFrontendLayer | None): Trained frontend to use for plotting.
        fft_length (int): FFT length used for linear/hybrid paths.
        mag_scale (str): 'pcen' | 'pwl' | 'db' | 'none' magnitude scaling (for visualized output).
        snr_threshold (float): Minimum activity threshold for chunk selection.
        prefix (str): Prefix for saved plot filenames in the samples/ folder.

    Returns:
        None. Saves up to 10 example spectrogram images under samples/.
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
        dummy_shape = (1, int(chunk_duration * sample_rate), 1) if audio_frontend in ('tf', 'raw') else (1, fft_bins, spec_width, 1)
        dummy = tf.zeros(dummy_shape, dtype=tf.float32)
        _ = tf_frontend(dummy, training=False)

    for i, path in enumerate(file_paths[:10]):
        audio_chunks = load_audio_file(path, sample_rate, max_duration, chunk_duration)
        if len(audio_chunks) == 0: 
            print(f"Skipping file {path} (failed to load)")
            continue

        if audio_frontend in ('librosa', 'precomputed'):
            # Precompute mel + requested mag_scale for visualization
            specs = [get_spectrogram_from_audio(chunk, sample_rate, n_fft=fft_length, mel_bins=mel_bins, spec_width=spec_width, mag_scale=mag_scale) for chunk in audio_chunks]
            pool = sort_by_activity(specs, threshold=snr_threshold) or specs
            if len(pool) == 0: continue
            spec = pick_random_samples(pool, num_samples=1, pick_first=True)
            spec = spec[0] if isinstance(spec, list) else spec

        elif audio_frontend == 'hybrid':
            # Feed linear power to TF frontend (mel + mag_scale applied in TF layer)
            specs = [get_spectrogram_from_audio(chunk, sample_rate, n_fft=fft_length, mel_bins=-1, spec_width=spec_width) for chunk in audio_chunks]
            print(f"File {i+1}: {os.path.basename(path)} - {len(audio_chunks)} chunks, example spec shape: {specs[0].shape}")
            pool = sort_by_activity(specs, threshold=snr_threshold) or specs
            print(f"  Selected {len(pool)} chunks after activity-based filtering.")
            if len(pool) == 0: continue
            spec_in = pick_random_samples(pool, num_samples=1, pick_first=True)
            spec_in = spec_in[0] if isinstance(spec_in, list) else spec_in  # [fft_bins, spec_width]
            inp = spec_in[np.newaxis, :, :, np.newaxis].astype(np.float32)   # [B,fft_bins,T,1]
            spec = tf_frontend(inp, training=False).numpy()[0, :, :, 0]

        else:  # raw/tf
            pool = sort_by_activity(audio_chunks, threshold=snr_threshold) or audio_chunks
            if len(pool) == 0: continue
            chunk = pick_random_samples(pool, num_samples=1, pick_first=True)
            chunk = (chunk[0] if isinstance(chunk, list) else chunk)[:int(sample_rate * chunk_duration)]
            if len(chunk) < sample_rate * chunk_duration:
                chunk = np.pad(chunk, (0, int(sample_rate * chunk_duration) - len(chunk)))
            chunk = chunk / (np.max(np.abs(chunk)) + 1e-6)
            inp = chunk[np.newaxis, ..., np.newaxis].astype(np.float32)
            spec = tf_frontend(inp, training=False).numpy()[0, :, :, 0]

        plot_spectrogram(spec, title=f"{audio_frontend}_{mag_scale}_{prefix}_{os.path.basename(path)}")
        
def get_classes_with_most_samples(directory, n_classes=25, include_noise=False, exts: tuple = SUPPORTED_AUDIO_EXTS):
    """
    Collect the most frequent class labels from a dataset root.

    Args:
        directory (str): Root dataset directory (class-subfolders).
        n_classes (int): Number of top classes to return (upper bound).
        include_noise (bool): If False, exclude noise-like labels (noise/silence/background/other).
        exts (tuple[str, ...]): Accepted audio file extensions (case-insensitive).

    Returns:
        list[str]: Up to n_classes class names, sorted by descending sample count.
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

    Expected layout:
        root/
          class_a/*.(wav|mp3|flac|ogg|m4a)
          class_b/*.(wav|mp3|flac|ogg|m4a)
          ...

    Args:
        directory (str): Dataset root directory.
        classes (list[str] | None): If given, restrict to these class names only.
        max_samples (int | None): Cap the number of files per class (uniform random).
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
    Upsample minority classes to approach the largest class size via repetition.

    Args:
        file_paths (list[str]): List of audio file paths.
        classes (list[str]): Ordered class names.
        ratio (float): Target fraction of the largest class size (0 < ratio ≤ 1).

    Returns:
        list[str]: Augmented list of audio file paths with upsampled minority classes.

    Notes:
        - Sampling is with replacement; randomness follows numpy’s global RNG state.
        - This does not rebalance perfectly; it raises all classes up to ratio * max_size.
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

def data_generator(file_paths, 
                   classes, batch_size=32, 
                   audio_frontend='librosa', 
                   sample_rate=22050, 
                   max_duration=30, 
                   chunk_duration=3, 
                   spec_width=128, 
                   mixup_alpha=0.2, 
                   mixup_probability=0.25, 
                   mel_bins=48, 
                   fft_length=512, 
                   mag_scale='none', 
                   random_offset=False,
                   snr_threshold=0.5):
    """
    Yield batches of (inputs, one_hot_labels) for training/validation.

    Frontends and input shapes:
        - precomputed/librosa: mel spectrogram -> [B, mel_bins, spec_width, 1]
        - hybrid: linear STFT magnitude -> [B, fft_bins(=n_fft//2+1), spec_width, 1]
        - raw/tf: waveform -> [B, T, 1], where T = sample_rate * chunk_duration (peak-normalized to [-1, 1])

    Selection and augmentation:
        - Loads multiple chunks per file and selects one by activity (SNR) ranking.
        - Optional mixup applied to a random subset of the batch (inputs mixed, labels merged by OR).

    Args:
        file_paths (list[str]): Audio file paths.
        classes (list[str]): Ordered class names for one-hot encoding.
        batch_size (int): Batch size.
        audio_frontend (str): 'precomputed' | 'librosa' | 'hybrid' | 'raw' | 'tf'.
        sample_rate (int): Sampling rate (Hz).
        max_duration (int): Max duration to read per file (seconds).
        chunk_duration (int): Chunk duration (seconds).
        spec_width (int): Target spectrogram width (frames).
        mixup_alpha (float): Mixup strength parameter (uniform here).
        mixup_probability (float): Fraction of the batch to apply mixup to.
        mel_bins (int): Number of mel bins for mel spectrograms.
        fft_length (int): FFT size used by librosa/hybrid paths.
        mag_scale (str): 'pcen' | 'pwl' | 'db' | 'none' magnitude scaling inside the frontend.
        random_offset (bool): If True, randomly offset chunk start within file.
        snr_threshold (float): Minimum activity threshold for chunk selection.

    Yields:
        tuple[np.ndarray, np.ndarray]: (inputs, labels) for a batch. Infinite generator.

    Notes:
        - Files with labels not present in 'classes' are skipped.
        - Noise-like labels produce all-zero target vectors (treated as background).
    """
    T = int(sample_rate * chunk_duration)
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
                    pool = sort_by_activity(specs, threshold=snr_threshold) or specs
                    if len(pool) == 0: continue
                    sample = pick_random_samples(pool, num_samples=1, pick_first=random_offset)
                    sample = sample[0] if isinstance(sample, list) else sample   # [mel, T]
                    need_ch_last = True

                elif audio_frontend == 'hybrid':
                    specs = [get_spectrogram_from_audio(chunk, sample_rate, n_fft=fft_length, mel_bins=-1, spec_width=spec_width) for chunk in audio_chunks]
                    pool = sort_by_activity(specs, threshold=snr_threshold) or specs
                    if len(pool) == 0: continue
                    sample = pick_random_samples(pool, num_samples=1, pick_first=random_offset)
                    sample = sample[0] if isinstance(sample, list) else sample   # [fft_bins, T]
                    need_ch_last = True

                elif audio_frontend in ('tf', 'raw'):
                    pool = sort_by_activity(audio_chunks, threshold=snr_threshold) or audio_chunks
                    if len(pool) == 0: continue
                    sample = pick_random_samples(pool, num_samples=1, pick_first=random_offset)
                    x = sample[0] if isinstance(sample, list) else sample
                    x = x[:T]
                    if x.shape[0] < T: x = np.pad(x, (0, T - x.shape[0]))
                    x = x / (np.max(np.abs(x)) + 1e-6)
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
                    #lam = np.random.beta(mixup_alpha, mixup_alpha, size=(num_mix,)) # beta dist for mixup
                    lam = np.random.uniform(mixup_alpha, 1 - mixup_alpha, size=(num_mix,)) # uniform dist for mixup
                    lam_inp = lam.reshape((num_mix,) + (1,) * (batch_samples.ndim - 1))
                    # Audio: weighted mix
                    batch_samples[mix_indices] = (
                        lam_inp * batch_samples[mix_indices] +
                        (1 - lam_inp) * batch_samples[permuted_indices[mix_indices]]
                    )
                    # Labels: elementwise OR (multi-label union)
                    batch_labels[mix_indices] = np.maximum(
                        batch_labels[mix_indices],
                        batch_labels[permuted_indices[mix_indices]]
                    )

            yield batch_samples, batch_labels

def load_dataset(file_paths, classes, audio_frontend='precomputed', batch_size=32, spec_width=128, mel_bins=48, **kwargs):
    """
    Wrap the Python generator as a tf.data.Dataset with static shapes.

    Args:
        file_paths (list[str]): Audio file paths.
        classes (list[str]): Ordered class names for one-hot encoding.
        audio_frontend (str): 'precomputed' | 'librosa' | 'hybrid' | 'raw' | 'tf'.
        batch_size (int): Batch size.
        spec_width (int): Target spectrogram width (frames).
        mel_bins (int): Number of mel bins.
        **kwargs: sample_rate (int), chunk_duration (int), fft_length (int),
                  mag_scale (str), max_duration (int),
                  mixup_alpha (float), mixup_probability (float),
                  random_offset (bool), snr_threshold (float).

    Returns:
        tf.data.Dataset: Infinite dataset of (inputs, labels), with prefetching enabled.

    Shapes:
        - Inputs follow the selected frontend (see data_generator).
        - Labels are [B, len(classes)] float32 one-hot/multi-hot vectors.
    """
    sr = kwargs.get('sample_rate', 16000)
    cd = kwargs.get('chunk_duration', 3)
    fft_length = kwargs.get('fft_length', 512)
    mag_scale = kwargs.get('mag_scale', 'none')
    random_offset = kwargs.get('random_offset', False)
    snr_threshold = kwargs.get('snr_threshold', 0.5)
    chunk_len = int(sr * cd)

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
            mag_scale=mag_scale,
            random_offset=random_offset
        ),
        output_signature=output_signature
    )
    return dataset.repeat().prefetch(tf.data.AUTOTUNE)

class AudioFrontendLayer(layers.Layer):
    """
    Audio frontend with interchangeable input modes and optional magnitude scaling.

    Modes:
      - precomputed: Input mel spectrogram [B, mel_bins, spec_width, 1] -> slice to spec_width.
      - hybrid:      Linear STFT bins [B, fft_bins, spec_width, 1] -> 1x1 mel mixer (optionally train mel spacing).
      - raw:         Waveform [B, T, 1] -> explicit symmetric pad -> VALID Conv2D(1,k) stride s -> BN -> ReLU6
                     -> magnitude scaling -> transpose -> [B, mel_bins, spec_width, 1].

    Magnitude scaling:
      - 'none': Pass-through.
      - 'pwl':  Piecewise-linear compression (DW 1x1 branches + ReLU + Add).
      - 'pcen': PCEN-like compression (pool/conv/ReLU/Add), linear magnitude domain.
      - 'db':   Log compression (10·log10) after mel (often unfriendly to PTQ).

    Trainability:
      - is_trainable=True enables training of mag-scale sublayers (PCEN/PWL).
      - In hybrid mode, mel filter spacing (breakpoints) is currently tied to is_trainable
        and trained via a matmul path during training; at inference, a single 1x1 Conv2D is used.

    Notes:
      - Slaney mel basis (librosa) seeds mel_mixer for parity in hybrid mode.
      - Raw branch uses explicit VALID padding to avoid TF vs. TFLite SAME-padding differences.
      - Channel padding to multiples of 8 may be applied internally for NPU-friendly tensors.
    """
    def __init__(self, 
                 mode: str,
                 mel_bins: int,
                 spec_width: int,
                 sample_rate: int,
                 chunk_duration: int,
                 fft_length: int = 512,
                 pcen_K: int = 8,
                 init_mel: bool = True,
                 mel_fmin: float = 150.0,
                 mel_fmax: Optional[float] = None,
                 mel_norm: str = "slaney",
                 mag_scale: str = "none",
                 name: str = "audio_frontend",
                 is_trainable: bool = False,
                 train_mel_scale: bool = False,   # NEW
                 **kwargs):
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
        self.train_mel_scale = bool(is_trainable)

        # DB params
        self.db_eps = 1e-6
        self.db_ref = 1.0

        # Fixed input samples for one chunk
        self._T = int(self.sample_rate * self.chunk_duration)
        self._pad_ch_in = 0

        # Hybrid 1x1 mel mixer (weights will be updated when training mel spacing)
        self.mel_mixer = layers.Conv2D(
            filters=int(self.mel_bins),
            kernel_size=(1, 1),
            padding="same",
            use_bias=False,
            kernel_constraint=constraints.NonNeg(),
            name=f"{name}_mel_mixer",
            trainable=False,  # keep kernel non-trainable; we overwrite it from learned breakpoints
        )

        # Placeholders for learnable mel (only used in hybrid)
        self._bins_mel = None
        self._mel_fmin = None
        self._mel_fmax = None
        self._mel_range = None
        self._mel_seg_logits = None  # [mel_bins + 1], intervals between breakpoints

        # RAW: single Conv2D with explicit VALID padding to guarantee TF/TFLite parity
        if self.mode == "raw":
            T = int(self.sample_rate * self.chunk_duration)   # input samples
            W = int(self.spec_width)                          # target frames
            self._k_t = 16                                    # temporal kernel
            self._stride_t = int(math.ceil(T / float(W)))     # stride s = ceil(T/W)

            # For VALID conv: out = floor((L_in + pad_total - k)/s) + 1  == W
            pad_total = max(0, self._stride_t * (W - 1) + self._k_t - T)
            self._pad_left = pad_total // 2
            self._pad_right = pad_total - self._pad_left

            self.fb2d = layers.Conv2D(
                filters=int(self.mel_bins),
                kernel_size=(1, self._k_t),
                strides=(1, self._stride_t),
                padding='valid',                 # we pad ourselves to make width static
                use_bias=False,
                name=f"{name}_raw_fb2d",
                trainable=self.is_trainable,
            )
            # Quantization-friendly normalization + bounded activation
            self.fb_bn = layers.BatchNormalization(momentum=0.99, epsilon=1e-3,
                                                   name=f"{name}_raw_fb2d_bn",
                                                   trainable=self.is_trainable)
            self.fb_relu = layers.ReLU(max_value=6, name=f"{name}_raw_fb2d_relu")
        else:
            self._pad_left = 0
            self._pad_right = 0
            self._stride_t = 1
            self.fb2d = None
            self.fb_bn = None
            self.fb_relu = None

        # Build PWL/PCEN helpers as before
        if self.mag_scale == "pcen":
            self._pcen_pools = [layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding="same",
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
        if self.mode == "hybrid":
            # Setup fixed mel mixer (and record any input-channel pad)
            fft_bins = self.fft_length // 2 + 1
            self._build_and_set_mel_mixer(n_fft=self.fft_length, cin=fft_bins)

            # Learnable mel spacing setup
            if self.train_mel_scale:
                sr = int(self.sample_rate)
                fmax = int(self.mel_fmax) if self.mel_fmax is not None else (sr // 2)
                freqs = np.linspace(0.0, float(sr) / 2.0, fft_bins, dtype=np.float32)
                bins_mel = librosa.hz_to_mel(freqs)
                self._bins_mel = tf.constant(bins_mel.astype(np.float32), dtype=tf.float32)   # [F]
                self._mel_fmin = float(librosa.hz_to_mel(float(self.mel_fmin)))
                self._mel_fmax = float(librosa.hz_to_mel(float(fmax)))
                self._mel_range = float(self._mel_fmax - self._mel_fmin)
                # Intervals logits (M+1), init to equal spacing
                init_logits = np.zeros((self.mel_bins + 1,), dtype=np.float32)
                self._mel_seg_logits = self.add_weight(
                    name=f"{self.name}_mel_seg_logits",
                    shape=(self.mel_bins + 1,),
                    initializer=tf.keras.initializers.Constant(init_logits),
                    trainable=self.is_trainable,  # controlled by is_trainable
                )

        elif self.mode in ("raw",):
            # Build Conv2D and BN on static shapes to eliminate dynamic width
            T = int(self.sample_rate * self.chunk_duration)
            static_w = int(self.spec_width)  # guaranteed by VALID + our padding
            in_w = T + int(self._pad_left) + int(self._pad_right)
            self.fb2d.build(tf.TensorShape([None, 1, in_w, 1]))
            if self.fb_bn is not None:
                # fb2d output: [B, 1, static_w, mel_bins]
                self.fb_bn.build(tf.TensorShape([None, 1, static_w, int(self.mel_bins)]))

        self._build_mag_layers()
        super().build(input_shape)

    def _compute_tri_matrix(self) -> tf.Tensor:
        """
        Build a triangular mel weight matrix from learnable breakpoints.

        Returns:
            tf.Tensor: [F, M] triangle weights, where
                F = fft_bins = n_fft//2 + 1, M = mel_bins.
        Notes:
            - Breakpoints are parameterized as positive intervals via softplus and normalized
              to the [mel_fmin, mel_fmax] range.
            - Used only during training in hybrid mode to obtain gradients w.r.t. breakpoints.
        """
        eps = tf.constant(1e-6, tf.float32)
        F = tf.shape(self._bins_mel)[0]
        M = int(self.mel_bins)

        # Positive intervals via softplus, normalized to full mel range
        seg = tf.nn.softplus(self._mel_seg_logits) + 1e-3           # [M+1]
        seg = seg / (tf.reduce_sum(seg) + eps) * tf.constant(self._mel_range, tf.float32)
        cs = tf.cumsum(seg)                                         # [M+1]
        p_full = tf.concat([
            tf.constant([self._mel_fmin], tf.float32),
            tf.constant([self._mel_fmin], tf.float32) + cs
        ], axis=0)                                                  # [M+2]

        left   = p_full[0:M]          # [M]
        center = p_full[1:M+1]        # [M]
        right  = p_full[2:M+2]        # [M]

        bm = self._bins_mel                                             # [F]

        denom_l = tf.maximum(center - left, eps)
        denom_r = tf.maximum(right - center, eps)
        up   = (bm[:, None] - left[None, :]) / denom_l[None, :]
        down = (right[None, :] - bm[:, None]) / denom_r[None, :]

        tri = tf.maximum(tf.minimum(up, down), 0.0)                     # [F, M]
        # Normalize per filter for stability (sum over F)
        tri = tri / (tf.reduce_sum(tri, axis=0, keepdims=True) + eps)
        return tri  # [F, M]

    def _assign_mel_kernel_from_tri(self, tri: tf.Tensor):
        """
        Mirror the [F, M] triangle matrix into the 1x1 Conv2D mel_mixer kernel.

        This keeps the inference path NPU-friendly (Conv2D-only). Any channel padding
        added in _build_and_set_mel_mixer is honored.

        Args:
            tri (tf.Tensor): [F, M] triangle weights (nonnegative).
        """
        if self.mel_mixer is None or not hasattr(self.mel_mixer, "kernel"):
            return
        # Account for any input channel padding added in _build_and_set_mel_mixer
        if getattr(self, "_pad_ch_in", 0):
            pad = self._pad_ch_in
            zeros = tf.zeros([pad, int(self.mel_bins)], dtype=tri.dtype)
            tri = tf.concat([tri, zeros], axis=0)  # [F+pad, M]
        # Conv2D kernel shape is [1, 1, C_in, C_out]
        k = tf.reshape(tri, [1, 1, tf.shape(tri)[0], int(self.mel_bins)])
        # Overwrite kernel (non-trainable) so inference/export sees a fixed Conv2D
        self.mel_mixer.kernel.assign(k)

    # Helper: build mel mixer with a Slaney mel basis for a given n_fft and input channels
    def _build_and_set_mel_mixer(self, n_fft: int, cin: int):
        """
        Initialize mel_mixer from a Slaney mel basis and pad input channels if needed.

        Args:
            n_fft (int): FFT length used to compute the mel basis.
            cin (int): Number of input channels (fft_bins = n_fft//2 + 1).

        Side effects:
            - Builds mel_mixer on input shape [B, 1, T, cin(+pad)].
            - Sets mel_mixer weights to [1, 1, cin(+pad), mel_bins].
            - Stores channel padding in self._pad_ch_in.
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

    # Helper: build mag-scale layers (PCEN/PWL)
    def _build_mag_layers(self):
        """
        Ensure magnitude scaling sub-layers are built (PCEN/PWL), if selected.

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
        PCEN-like compression using only pool/conv/ReLU/Add ops.

        Args:
            x (tf.Tensor): Power-like input [B, 1, T, C].

        Returns:
            tf.Tensor: Compressed output [B, 1, T, C].
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

    def _apply_pwl(self, x):
        """
        Apply piecewise-linear compression via 1x1 depthwise branches.

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
        Apply dB log compression (10 * log10).

        Args:
            x (tf.Tensor): Linear input [B, 1, T, C].

        Returns:
            tf.Tensor: dB-scaled output [B, 1, T, C].
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
                • If is_trainable: training uses a matmul path (for gradients),
                  inference uses a single 1x1 Conv2D (mel_mixer).
            - raw:         in [B, T, 1]           -> out [B, mel_bins, spec_width, 1]

        Args:
            inputs (tf.Tensor): See shapes above.
            training (bool | tf.Tensor | None): Keras training flag.

        Returns:
            tf.Tensor: Frontend output [B, mel_bins, spec_width, 1].
        """
        # precomputed
        if self.mode == "precomputed":
            return inputs[:, :, :self.spec_width, :]

        if self.mode == "hybrid":
            fft_bins = self.fft_length // 2 + 1
            if inputs.shape.rank != 4 or (inputs.shape[1] is not None and int(inputs.shape[1]) != fft_bins):
                raise ValueError(f"Hybrid expects [B,{fft_bins},T,1], got {inputs.shape}")
            y = tf.transpose(inputs, [0, 3, 2, 1])  # [B,1,T,fft_bins]
            y = y[:, :, :self.spec_width, :]

            if self.train_mel_scale and self.is_trainable:
                def _train_branch(y_in: tf.Tensor) -> tf.Tensor:
                    # Compute triangles, update 1x1 kernel for future inference, and use matmul for gradients
                    tri = self._compute_tri_matrix()                         # [F, M]
                    # Keep kernel in sync but stop gradients through the assignment
                    self._assign_mel_kernel_from_tri(tf.stop_gradient(tri))
                    B = tf.shape(y_in)[0]; Tt = tf.shape(y_in)[2]; F = tf.shape(y_in)[3]
                    y_flat = tf.reshape(y_in, [B * Tt, F])                   # [B*T, F]
                    y_mel = tf.matmul(y_flat, tri)                           # [B*T, M]
                    return tf.reshape(y_mel, [B, 1, Tt, int(self.mel_bins)]) # [B,1,T,M]

                def _infer_branch(y_in: tf.Tensor) -> tf.Tensor:
                    # N6-friendly 1x1 Conv2D only (no assigns in inference graph)
                    if self._pad_ch_in:
                        b = tf.shape(y_in)[0]; t = tf.shape(y_in)[2]
                        z = tf.zeros([b, 1, t, self._pad_ch_in], dtype=y_in.dtype)
                        y_in = tf.concat([y_in, z], axis=-1)
                    return self.mel_mixer(y_in)

                if isinstance(training, bool):
                    y = _train_branch(y) if training else _infer_branch(y)
                else:
                    y = tf.cond(tf.cast(training, tf.bool),
                                lambda: _train_branch(y),
                                lambda: _infer_branch(y))
            else:
                # Fixed mel mixer (Conv2D only)
                if self._pad_ch_in:
                    b = tf.shape(y)[0]; t = tf.shape(y)[2]
                    z = tf.zeros([b, 1, t, self._pad_ch_in], dtype=y.dtype)
                    y = tf.concat([y, z], axis=-1)
                y = self.mel_mixer(y)

            y = tf.nn.relu(y)
            y = self._apply_mag(y)
            y = tf.transpose(y, [0, 3, 2, 1])       # [B,mel,T,1]
            return y[:, :, :self.spec_width, :]

        # raw: explicit symmetric pad -> VALID Conv2D -> BN -> ReLU6 -> mag -> transpose
        x = inputs[:, :int(self.sample_rate * self.chunk_duration), :]
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
        """
        Keras static shape inference for the layer.

        Args:
            input_shape (tuple): Input shape tuple.

        Returns:
            tuple: (batch, mel_bins, spec_width, 1)
        """
        return (input_shape[0], int(self.mel_bins), int(self.spec_width), 1)
    
    def get_config(self):
        """
        Return a serializable configuration of the frontend layer.

        Returns:
            dict: JSON-serializable config for model saving/loading.
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
    """
    Round channel counts to be divisible by a given divisor.

    Args:
        v (int | float): Target channel count.
        divisor (int): Divisor to align to (default 8).

    Returns:
        int: Max(divisor, nearest multiple of 'divisor').
    """
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
        weight_decay (float): L2 kernel regularization (weight decay) for DW/PW.
        drop_rate (float): Spatial dropout rate (applied after PW BN).

    Returns:
        tf.Tensor: Output tensor after DW/PW + BN/ReLU (+ residual when stride==1 and channels match).
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
                      mag_scale='none', frontend_trainable=False, class_activation='softmax'):
    """
    Build a DS-CNN model with a selectable audio frontend.

    Frontends:
        - precomputed/librosa: expects mel spectrogram input.
        - hybrid: expects linear STFT magnitude; applies mel mixer inside the model.
        - raw/tf: expects waveform; applies a TF frontend to produce mel features.

    Args:
        num_mels (int): Number of mel bins in the spectrogram.
        spec_width (int): Spectrogram width (frames).
        sample_rate (int): Sampling rate (Hz).
        chunk_duration (int): Chunk duration (seconds).
        embeddings_size (int): Channels in the embeddings layer (final 1x1 conv).
        num_classes (int): Number of output classes.
        audio_frontend (str): 'precomputed' | 'librosa' | 'hybrid' | 'raw' | 'tf'.
        alpha (float): Width multiplier for the backbone.
        depth_multiplier (int): Repeats multiplier for DS blocks per stage.
        fft_length (int): FFT size for hybrid/librosa paths.
        mag_scale (str): 'pcen' | 'pwl' | 'db' | 'none' magnitude scaling inside the frontend.
        frontend_trainable (bool): If True, make frontend sub-layers trainable (mel spacing in hybrid, PCEN/PWL).
        class_activation (str): 'softmax' | 'sigmoid' for the classifier head.

    Returns:
        tf.keras.Model: Uncompiled DS-CNN model ready for training.
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
        inputs = tf.keras.Input(shape=(int(chunk_duration * sample_rate), 1), name='raw_audio_input')
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
    
    # BN+ReLU after frontend
    x = layers.BatchNormalization(name="frontend_bn")(x)
    x = layers.ReLU(max_value=6, name="frontend_relu")(x)

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
    outputs = layers.Dense(num_classes, activation=class_activation, name="pred")(x)
    return tf.keras.models.Model(inputs, outputs, name="dscnn_audio")

def train_model(model, 
                train_dataset, 
                val_dataset, 
                epochs=50, 
                learning_rate=0.001, 
                batch_size=64, 
                patience=10, 
                checkpoint_path="checkpoints/best_model.keras", 
                steps_per_epoch=None, 
                val_steps=None,
                is_multilabel=False):
    """
    Train the model with cosine LR schedule, early stopping, and checkpointing.

    Monitors:
        - val_loss (min). Best model is saved as a full .keras file.

    Args:
        model (tf.keras.Model): Model to train.
        train_dataset (tf.data.Dataset): Training dataset (infinite).
        val_dataset (tf.data.Dataset): Validation dataset (infinite).
        epochs (int): Number of epochs.
        learning_rate (float): Initial learning rate for cosine schedule.
        batch_size (int): Unused here; kept for API symmetry with data loader.
        patience (int): Early stopping patience (epochs).
        checkpoint_path (str): Path to save the best .keras model.
        steps_per_epoch (int): Training steps per epoch (> 0 required).
        val_steps (int | None): Validation steps per epoch (defaults to 1 if <= 0).
        is_multilabel (bool): If True, uses binary_crossentropy; else categorical_crossentropy.

    Returns:
        tf.keras.callbacks.History: Keras training history object.
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
        loss='binary_crossentropy' if is_multilabel else 'categorical_crossentropy',
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
    Compute hop length to produce spec_width frames from an input chunk.

    Args:
        sample_rate (int): Sampling rate (Hz).
        chunk_duration (int): Chunk duration (seconds).
        spec_width (int): Desired number of frames.

    Returns:
        int: hop_length in samples (floor(T / spec_width), at least 1), where T = sample_rate*chunk_duration.
    """
    T = int(sample_rate) * int(chunk_duration)
    return max(1, T // int(spec_width))

def get_args():
    """
    Parse command-line arguments for training.

    Returns:
        argparse.Namespace: Parsed arguments. See --help for details on each flag.
    """
    parser = argparse.ArgumentParser(description="Train STM32N6 audio classifier")
    parser.add_argument('--data_path_train', type=str, required=True, help='Path to train dataset')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples per class for training (None for all)')
    parser.add_argument('--upsample_ratio', type=float, default=0.5, help='Upsample ratio for minority classes')
    parser.add_argument('--sample_rate', type=int, default=22050, help='Audio sample rate. Default is 22050 Hz.')
    parser.add_argument('--num_mels', type=int, default=64, help='Number of mel bins for spectrogram')
    parser.add_argument('--spec_width', type=int, default=256, help='Spectrogram width')
    parser.add_argument('--fft_length', type=int, default=512, help='FFT length for STFT/linear spectrogram')
    parser.add_argument('--chunk_duration', type=float, default=3, help='Audio chunk duration (seconds)')
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
    parser.add_argument('--frontend_trainable', action='store_true', default=False, help='If set, make audio frontend trainable (mel_mixer/raw mixer/PCEN/PWL).')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup alpha')
    parser.add_argument('--mixup_probability', type=float, default=0.25, help='Fraction of batch to apply mixup to.')
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
        T = int(args.sample_rate * args.chunk_duration)
        if T >= (1 << 16):
            print(f"[WARN] STM32N6 compile will fail: raw input length {T} >= 65536.")
            print("       Use --sample_rate 16000 or --chunk_duration 2, or switch to --audio_frontend hybrid/precomputed.")
    
    # Compute hop length once for config/diagnostics
    hop_length = compute_hop_length(args.sample_rate, args.chunk_duration, args.spec_width)

    # Load file paths and classes
    file_paths, classes = load_file_paths_from_directory(args.data_path_train, 
                                                         max_samples=args.max_samples, 
                                                         classes=get_classes_with_most_samples(args.data_path_train, 100, True) # DEBUG: Only use N classes for debugging
                                                         )

    # Perform sanity check on the dataset
    sanity_file_paths = file_paths[:10]
    dataset_sanity_check(
        sanity_file_paths, classes,
        sample_rate=args.sample_rate,
        max_duration=args.max_duration,
        chunk_duration=args.chunk_duration,
        spec_width=args.spec_width,
        mel_bins=args.num_mels,
        audio_frontend=args.audio_frontend,
        fft_length=args.fft_length,
        mag_scale=args.mag_scale,
        snr_threshold=0.25,
        prefix='pre'
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
        mag_scale=args.mag_scale,
        random_offset=True,
        snr_threshold=0.1
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
        mag_scale=args.mag_scale,
        random_offset=False,
        snr_threshold=0.5
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
        class_activation='sigmoid'  if args.mixup_probability > 0 else 'softmax'  # Use sigmoid for mixup (multi-label), softmax otherwise
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
        val_steps=val_steps,
        is_multilabel=(args.mixup_probability > 0)
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
            sanity_file_paths, classes,
            sample_rate=args.sample_rate,
            max_duration=args.max_duration,
            chunk_duration=args.chunk_duration,
            spec_width=args.spec_width,
            mel_bins=args.num_mels,
            audio_frontend=args.audio_frontend,
            fft_length=args.fft_length,
            mag_scale=args.mag_scale,
            tf_frontend_layer=trained_frontend,
            snr_threshold=0.25,
            prefix='post'
        )
