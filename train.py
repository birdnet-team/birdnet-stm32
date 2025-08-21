import os
import librosa
import numpy as np
from typing import Iterable, Optional, Tuple
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import math

from utils.audio import load_audio_file, get_spectrogram_from_audio, sort_by_s2n, pick_random_samples, plot_spectrogram, save_wav

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
    
def dataset_sanity_check(file_paths, classes, sample_rate=22050, max_duration=30, chunk_duration=3, spec_width=128, mel_bins=64, audio_frontend='librosa', tf_frontend_layer=None):
    """
    Run a quick sanity check and save spectrogram images for a few files.

    - If audio_frontend == 'librosa':
        Loads audio, computes mel spectrograms per chunk using librosa helpers,
        filters by SNR, then plots a random spectrogram.
    - If audio_frontend == 'tf':
        Loads raw audio chunks and passes one chunk through the TF AudioFrontendLayer.
        If tf_frontend_layer is provided (e.g., from a trained model), it is used
        so plots reflect trained parameters (e.g., mag_scale, clamp); otherwise a
        fresh layer is built with default settings.

    Args:
        file_paths: List of audio file paths.
        classes: List of class names (unused here, only for labeling).
        sample_rate: Target sample rate for loading audio.
        max_duration: Max seconds to load from each file.
        chunk_duration: Seconds per chunk.
        spec_width: Target number of time frames in the spectrogram.
        mel_bins: Number of mel bins.
        audio_frontend: 'librosa' or 'tf'.
        tf_frontend_layer: Optional, a pre-built/trained AudioFrontendLayer.

    Returns:
        None. Saves plots under samples/ via utils.plot_spectrogram().
    """
    # Prepare output dir
    out_dir = os.path.join("samples")
    os.makedirs(out_dir, exist_ok=True)

    # If using TF frontend, initialize or reuse trained layer
    tf_frontend = None
    if audio_frontend == 'tf':
        if tf_frontend_layer is not None:
            tf_frontend = tf_frontend_layer
        else:
            tf_frontend = AudioFrontendLayer(
                sample_rate=sample_rate,
                chunk_duration=chunk_duration,
                mel_bins=mel_bins,
                spec_width=spec_width,
                name="audio_frontend_sanity"
            )
            # Build layer with a dummy input to ensure variables exist
            dummy = tf.zeros([1, sample_rate * chunk_duration, 1], dtype=tf.float32)
            _ = tf_frontend(dummy, training=False)

    # shuffle file paths
    np.random.shuffle(file_paths)
    for i, path in enumerate(file_paths[:5]):  # Check first 5 files
        audio_chunks = load_audio_file(path, sample_rate, max_duration, chunk_duration)
        if len(audio_chunks) == 0:
            print(f"File {path} has no valid audio chunks.")
            continue
        print(f"File {path} has {len(audio_chunks)} chunks of duration {chunk_duration} seconds each.")
        
        if audio_frontend == 'librosa':
            spectrograms = [get_spectrogram_from_audio(chunk, sample_rate, mel_bins=mel_bins, spec_width=spec_width) for chunk in audio_chunks]
            print(f"File {path} has {len(spectrograms)} spectrograms of shape {spectrograms[0].shape}.")
            
            sorted_specs = sort_by_s2n(spectrograms, threshold=0.5)
            print(f"File {path} has {len(sorted_specs)} spectrograms after SNR filtering.")
                    
            if len(sorted_specs) == 0:
                print(f"No valid spectrograms found for file {path}.")
                continue
            
            spec = pick_random_samples(sorted_specs, num_samples=1)     
            
        elif audio_frontend == 'tf':
            sorted_chunks = sort_by_s2n(audio_chunks, threshold=0.5)
            print(f"File {path} has {len(sorted_chunks)} audio chunks after SNR filtering.")
            
            if len(sorted_chunks) == 0:
                print(f"No valid audio chunks found for file {path}.")
                continue
            
            random_chunk = pick_random_samples(sorted_chunks, num_samples=1)
            chunk = random_chunk[0] if isinstance(random_chunk, list) else random_chunk

            # Run through TF frontend (use trained layer if provided)
            inp = np.expand_dims(chunk.astype(np.float32), axis=(0, -1))  # [1, T, 1]
            spec = tf_frontend(inp, training=False).numpy()[0, :, :, 0]   # [mel_bins, spec_width]
        else:
            raise ValueError("Invalid audio frontend. Choose 'librosa' or 'tf'.")
        
        plot_spectrogram(spec, title=f"Spectrogram for {os.path.basename(path)} - Class: {path.split('/')[-2]}")

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

def data_generator(file_paths, classes, batch_size=32, audio_frontend='librosa', sample_rate=22050, max_duration=30, chunk_duration=3, spec_width=128, mixup_alpha=0.2, mixup_probability=0.25, mel_bins=48):
    """
    Yield batches of model-ready samples and one-hot labels with optional mixup.

    For each selected file:
      - Loads up to max_duration seconds at sample_rate.
      - Splits into chunk_duration-second chunks.
      - Filters by SNR and picks one random chunk/spectrogram.

    Output shapes per frontend:
      - audio_frontend='librosa': sample shape (mel_bins, spec_width, 1)
      - audio_frontend='tf':      sample shape (sample_rate*chunk_duration, 1)

    Mixup:
      - If enabled, applies mixup to a random subset of the batch on both inputs
        and one-hot labels.

    Args:
        file_paths: List of .wav file paths.
        classes: List of class names; used to build one-hot labels.
        batch_size: Number of samples per batch.
        audio_frontend: 'librosa' or 'tf' (raw-audio frontend).
        sample_rate: Audio sample rate for loading.
        max_duration: Max seconds per file to load.
        chunk_duration: Seconds per training chunk.
        spec_width: Target number of frames for spectrograms (librosa path).
        mixup_alpha: Beta distribution parameter; 0 disables mixup.
        mixup_probability: Fraction of batch to mix.
        mel_bins: Number of mel bins (librosa path).

    Yields:
        (batch_samples, batch_labels)
        - batch_samples: float32 array of shape
            (B, mel_bins, spec_width, 1) for 'librosa'
            (B, sample_rate*chunk_duration, 1) for 'tf'
        - batch_labels: float32 array of shape (B, num_classes)
    """
    T = sample_rate * chunk_duration
    while True:
        idxs = np.random.permutation(len(file_paths))
        for batch_start in range(0, len(idxs), batch_size):
            batch_idxs = idxs[batch_start:batch_start + batch_size]
            batch_samples = []
            batch_labels = []
            for idx in batch_idxs:
                path = file_paths[idx]
                audio_chunks = load_audio_file(path, sample_rate, max_duration, chunk_duration)
                if len(audio_chunks) == 0:
                    continue

                if audio_frontend == 'librosa':
                    spectrograms = [get_spectrogram_from_audio(chunk, sample_rate, mel_bins=mel_bins, spec_width=spec_width) for chunk in audio_chunks]
                    sorted_specs = sort_by_s2n(spectrograms, threshold=0.5)
                    pool = sorted_specs if len(sorted_specs) > 0 else spectrograms
                    if len(pool) == 0:
                        continue
                    sample = pick_random_samples(pool, num_samples=1)
                    sample = sample[0] if isinstance(sample, list) else sample
                elif audio_frontend == 'tf':
                    sorted_chunks = sort_by_s2n(audio_chunks, threshold=0.5)
                    pool = sorted_chunks if len(sorted_chunks) > 0 else audio_chunks
                    if len(pool) == 0:
                        continue
                    sample = pick_random_samples(pool, num_samples=1)
                    x = sample[0] if isinstance(sample, list) else sample
                    # Pad/truncate to fixed length
                    x = x[:T]
                    if x.shape[0] < T:
                        x = np.pad(x, (0, T - x.shape[0]))
                    sample = x
                else:
                    raise ValueError("Invalid audio frontend. Choose 'librosa' or 'tf'.")

                label_str = path.split('/')[-2]
                if label_str.lower() in ['noise', 'silence', 'background', 'other']:
                    one_hot_label = np.zeros(len(classes), dtype=np.float32)
                else:
                    if label_str not in classes:
                        continue
                    one_hot_label = tf.one_hot(classes.index(label_str), depth=len(classes)).numpy()

                sample = np.expand_dims(sample, axis=-1)
                batch_samples.append(sample.astype(np.float32))
                batch_labels.append(one_hot_label.astype(np.float32))

            if len(batch_samples) == 0:
                continue
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

def load_dataset(file_paths, classes, audio_frontend='librosa', batch_size=32, spec_width=128, mel_bins=48, **kwargs):
    """
    Wrap the Python generator into a tf.data.Dataset with fixed signatures.

    Input spec depends on the chosen frontend:
      - 'librosa': (None, mel_bins, spec_width, 1)
      - 'tf':      (None, sample_rate*chunk_duration, 1)

    Args:
        file_paths: List of .wav file paths.
        classes: List of class names.
        audio_frontend: 'librosa' or 'tf'.
        batch_size: Batch size.
        spec_width: Number of time frames for spectrograms.
        mel_bins: Number of mel bins for spectrograms.
        **kwargs:
            sample_rate (int): Audio sample rate (default 16000).
            chunk_duration (int): Seconds per training chunk (default 3).
            max_duration (int): Max seconds to read per file (default 30).
            mixup_alpha (float): Mixup Beta parameter.
            mixup_probability (float): Fraction of samples to mix.

    Returns:
        A tf.data.Dataset yielding (inputs, labels) with the shapes listed above.
    """
    sr = kwargs.get('sample_rate', 16000)
    cd = kwargs.get('chunk_duration', 3)
    chunk_len = sr * cd

    if audio_frontend == 'librosa':
        input_spec = tf.TensorSpec(shape=(None, mel_bins, spec_width, 1), dtype=tf.float32)
    elif audio_frontend == 'tf':
        # Fixed-length raw audio matching model input
        input_spec = tf.TensorSpec(shape=(None, chunk_len, 1), dtype=tf.float32)
    else:
        raise ValueError("Invalid audio frontend. Choose 'librosa' or 'tf'.")

    output_signature = (
        input_spec,
        tf.TensorSpec(shape=(None, len(classes)), dtype=tf.float32),
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(
            file_paths, classes,
            batch_size=batch_size,
            audio_frontend=audio_frontend,   # forward the frontend
            sample_rate=sr,
            max_duration=kwargs.get('max_duration', 30),
            chunk_duration=cd,
            spec_width=spec_width,
            mixup_alpha=kwargs.get('mixup_alpha', 0.0),
            mixup_probability=kwargs.get('mixup_probability', 0.0),
            mel_bins=mel_bins,
        ),
        output_signature=output_signature
    )
    dataset = dataset.repeat().prefetch(tf.data.AUTOTUNE)
    return dataset

class N6ConvFrontend(layers.Layer):
    """Like V2, but anti-alias uses AveragePooling (GPU+NPU friendly).

    anti_alias_mode: "avg" (default), "dw" (depthwise 1x3), or "none".
    """
    def __init__(
        self,
        n_bins: int = 64,
        channels: Iterable[int] = (24, 32, 48, 64, 64, 64, 64),
        stride_plan: Iterable[int] | None = (4, 2, 2, 2, 2, 2, 2),
        target_frames: Optional[int] = None,
        input_samples: int = 48_000,
        anti_alias_mode: str = "avg",
        use_preemphasis: bool = False,
        compression: str = "tanh",
        smoothing_pool: int = 3,
        out_layout: str = "2d",
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if smoothing_pool not in (1, 2, 3):
            raise ValueError("smoothing_pool must be 1, 2, or 3")
        if compression not in ("hard_swish", "tanh", "none"):
            raise ValueError("compression must be 'hard_swish', 'tanh', or 'none'.")
        if out_layout not in ("2d", "1d"):
            raise ValueError("out_layout must be '2d' or '1d'")
        if anti_alias_mode not in ("avg", "dw", "none"):
            raise ValueError("anti_alias_mode must be 'avg', 'dw', or 'none'")
        if target_frames is not None and target_frames <= 0:
            raise ValueError("target_frames must be positive when provided")
        if input_samples <= 0:
            raise ValueError("input_samples must be positive")

        self.n_bins = int(n_bins)
        self.channels = tuple(int(c) for c in channels)
        self.anti_alias_mode = anti_alias_mode
        self.use_preemphasis = bool(use_preemphasis)
        self.compression = compression
        self.smoothing_pool = int(smoothing_pool)
        self.out_layout = out_layout
        self.input_samples = int(input_samples)
        self.target_frames = int(target_frames) if target_frames is not None else None

        # Stride plan (same as V2)
        if self.target_frames is not None:
            self.stride_plan = N6ConvFrontendV2._make_stride_plan(self.input_samples, self.target_frames, len(self.channels))
        else:
            sp = tuple(int(s) for s in stride_plan)
            if not all(s in (1, 2, 4) for s in sp):
                raise ValueError("stride_plan values must be in {1,2,4}")
            if len(sp) != len(self.channels):
                raise ValueError("channels and stride_plan must have same length")
            self.stride_plan = sp

        # Shared pieces
        self._expand_h = layers.Lambda(lambda x: tf.expand_dims(x, axis=1), name="expand_h")
        self._relu = layers.Activation("relu")
        self._square = layers.Multiply(name="square")
        self._pool = layers.AveragePooling2D(pool_size=(1, self.smoothing_pool), strides=(1, 1), padding="same")

        # Optional pre-emphasis
        self.pre_depthwise = None
        if self.use_preemphasis:
            self.pre_depthwise = layers.DepthwiseConv2D((1,3), strides=(1,1), padding="same", use_bias=False, name="pre_dw1x3")

        # Build DS stacks
        self.lpf = []       # either avgpool or dw1x3
        self.ds_dw = []
        self.ds_pw = []
        for i, (ch, s) in enumerate(zip(self.channels, self.stride_plan)):
            if self.anti_alias_mode == "avg" and s in (2,4):
                self.lpf.append(layers.AveragePooling2D(pool_size=(1,3), strides=(1,1), padding="same", name=f"ds{i+1}_lpf_avg1x3"))
            elif self.anti_alias_mode == "dw" and s in (2,4):
                self.lpf.append(layers.DepthwiseConv2D((1,3), strides=(1,1), padding="same", use_bias=False, name=f"ds{i+1}_lpf1x3"))
            else:
                self.lpf.append(None)
            self.ds_dw.append(layers.DepthwiseConv2D((1,3), strides=(s,s), padding="same", use_bias=False, name=f"ds{i+1}_dw1x3_s{s}"))
            self.ds_pw.append(layers.Conv2D(ch, (1,1), padding="same", use_bias=True, name=f"ds{i+1}_pw1x1"))

        if self.compression == "hard_swish":
            self._compress = layers.Activation("hard_swish")
        elif self.compression == "tanh":
            self._compress = layers.Activation("tanh")
        else:
            self._compress = None

        self._bin_mixer = layers.Conv2D(self.n_bins, (1,1), padding="same", use_bias=True, name="bins_1x1")
        self._to_2d = layers.Permute((3,2,1), name="to_F_T_1")

    def call(self, inputs, training=None):
        x = inputs
        if x.shape.rank == 2:
            x = tf.expand_dims(x, axis=-1)
        elif x.shape.rank != 3:
            raise ValueError("Input must be (B,T) or (B,T,1)")
        x = self._expand_h(x)

        if self.pre_depthwise is not None:
            x = self.pre_depthwise(x); x = tf.nn.relu(x)

        for lpf, dw, pw in zip(self.lpf, self.ds_dw, self.ds_pw):
            if lpf is not None: x = lpf(x)
            x = dw(x); x = pw(x); x = tf.nn.relu(x)

        x = tf.nn.relu(x); x = self._square([x, x]); x = self._pool(x)
        if self._compress is not None: x = self._compress(x)
        x = self._bin_mixer(x)
        if self.out_layout == "2d":
            x = self._to_2d(x)
        else:
            x = tf.squeeze(x, axis=1)
        return x

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 2: batch, t = input_shape
        else: batch, t, _ = input_shape
        def ceil_div(a,b): return None if a is None else (a+b-1)//b
        T = t
        for s in self.stride_plan: T = ceil_div(T, s)
        return (batch, self.n_bins, T, 1) if self.out_layout=="2d" else (batch, T, self.n_bins)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "n_bins": self.n_bins,
            "channels": self.channels,
            "stride_plan": self.stride_plan,
            "target_frames": self.target_frames,
            "input_samples": self.input_samples,
            "anti_alias_mode": self.anti_alias_mode,
            "use_preemphasis": self.use_preemphasis,
            "compression": self.compression,
            "smoothing_pool": self.smoothing_pool,
            "out_layout": self.out_layout,
        })
        return cfg

class AudioFrontendLayer(layers.Layer):
    """
    Learned, NPU-friendly audio frontend.

    Pipeline:
      1) Reshape raw audio [B, T, 1] into a raster [B, frames_src, subband_width, 1]
         where frames_src = ceil(T / subband_width).
      2) Two 3x3 Conv2D (+BN+ReLU), stride 1, filters=mel_bins.
      3) Collapse subband width to 1 via AveragePooling2D using kernels <= 3.
      4) Energy detection (square).
      5) Fixed temporal smoothing: DepthwiseConv2D 3x1 with [1,2,1]/4.
      6) Per-mel affine (Depthwise 1x1 with bias) to learn gain/bias per channel.
      7) Piecewise-linear compression y = k1*x + (k2-k1)*relu(x - t), clamped to [0,1].
      8) Downsample time via a static chain of AveragePooling2D(2x1) until near spec_width.
      9) Transpose to [B, mel_bins, spec_width, 1] and statically slice.

    All ops are Conv2D/DepthwiseConv2D/AvgPool/ReLU/Mul/Transpose — INT8 friendly and map to NPU.
    """
    def __init__(
        self,
        sample_rate: int,
        chunk_duration: int,
        mel_bins: int,
        spec_width: int,
        subband_width: int = 6,
        k1: float = 0.6,
        k2: float = 0.15,
        t: float = 0.2,
        name: str = "audio_frontend",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.sample_rate = int(sample_rate)
        self.chunk_duration = int(chunk_duration)
        self.mel_bins = int(mel_bins)
        self.spec_width = int(spec_width)
        self.subband_width = int(subband_width)

        # Layers
        self.conv_fb = None
        self.bn_fb1 = None
        self.conv_fb2 = None
        self.bn_fb2 = None
        self.pool_width_a = None
        self.pool_width_b = None
        self.pool_time_layers = []
        self.smooth_dw = None
        self.mel_affine = None

        # PWL params
        self._k1_init, self._k2_init, self._t_init = float(k1), float(k2), float(t)
        self.w_k1 = self.w_k2 = self.w_t = None

        # Static shapes
        self._T = self.sample_rate * self.chunk_duration
        self._pad = 0
        self._frames_src = None

    def build(self, input_shape):
        # Raster geometry
        w = self.subband_width
        self._pad = (w - (self._T % w)) % w
        self._frames_src = int((self._T + self._pad) // w)

        # Conv stack (3x3, stride 1)
        self.conv_fb = layers.Conv2D(
            filters=self.mel_bins, kernel_size=(3, 3), strides=(1, 1),
            padding='same', use_bias=False, name='fb_3x3_a'
        )
        self.bn_fb1 = layers.BatchNormalization(name='fb_3x3_a_bn')
        self.conv_fb2 = layers.Conv2D(
            filters=self.mel_bins, kernel_size=(3, 3), strides=(1, 1),
            padding='same', use_bias=False, name='fb_3x3_b'
        )
        self.bn_fb2 = layers.BatchNormalization(name='fb_3x3_b_bn')

        # Collapse subband width to 1
        if (self.subband_width % 3) == 0:
            self.pool_width_a = layers.AveragePooling2D(
                pool_size=(1, 3), strides=(1, 3), padding='valid', name='pool_width_a'
            )
            self.pool_width_b = layers.AveragePooling2D(
                pool_size=(1, self.subband_width // 3),
                strides=(1, self.subband_width // 3),
                padding='valid', name='pool_width_b'
            )
        else:
            self.pool_width_a = layers.AveragePooling2D(
                pool_size=(1, self.subband_width), strides=(1, self.subband_width),
                padding='valid', name='pool_width'
            )
            self.pool_width_b = None

        # Fixed temporal smoothing (DW 3x1 with [1,2,1]/4)
        self.smooth_dw = layers.DepthwiseConv2D(
            kernel_size=(3, 1), strides=(1, 1), padding='same',
            use_bias=False, name='smooth_dw_3x1', trainable=False
        )
        self.smooth_dw.build(tf.TensorShape([None, None, 1, self.mel_bins]))
        k = np.array([1.0, 2.0, 1.0], dtype=np.float32) / 4.0
        kernel = np.zeros((3, 1, self.mel_bins, 1), dtype=np.float32)
        for c in range(self.mel_bins):
            kernel[:, 0, c, 0] = k
        self.smooth_dw.set_weights([kernel])

        # Per-mel affine (depthwise 1x1 with bias)
        self.mel_affine = layers.DepthwiseConv2D(
            kernel_size=(1, 1), depth_multiplier=1, use_bias=True,
            padding='same', name='mel_affine'
        )

        # PWL parameters (Keras 3-safe add_weight usage)
        self.w_k1 = self.add_weight(
            name="k1",
            shape=(),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(self._k1_init),
            trainable=True,
        )
        self.w_k2 = self.add_weight(
            name="k2",
            shape=(),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(self._k2_init),
            trainable=True,
        )
        self.w_t = self.add_weight(
            name="t",
            shape=(),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(self._t_init),
            trainable=True,
        )

        # Time downsampling via powers-of-two 2x1 pools
        frames_after = self._frames_src
        ratio = frames_after // self.spec_width if self.spec_width > 0 else 1
        self.pool_time_layers = []
        pow2 = 1
        while (pow2 * 2) <= ratio:
            pow2 *= 2
        steps = int(math.log2(pow2)) if pow2 > 1 else 0
        for i in range(steps):
            self.pool_time_layers.append(
                layers.AveragePooling2D(
                    pool_size=(2, 1), strides=(2, 1), padding='valid', name=f'pool_time_2x1_{i+1}'
                )
            )

        super().build(input_shape)

    def call(self, inputs, training=None):
        # [B, T, 1] -> pad -> [B, frames_src, subband_width, 1]
        x = inputs
        if self._pad:
            x = tf.pad(x, [[0, 0], [0, self._pad], [0, 0]])
        B = tf.shape(x)[0]
        x = tf.reshape(x, [B, self._frames_src, self.subband_width, 1])

        # 3x3 conv stack
        y = self.conv_fb(x)
        y = self.bn_fb1(y, training=training)
        y = tf.nn.relu(y)
        y = self.conv_fb2(y)
        y = self.bn_fb2(y, training=training)
        y = tf.nn.relu(y)

        # collapse width to 1
        y = self.pool_width_a(y)
        if self.pool_width_b is not None:
            y = self.pool_width_b(y)

        # energy -> smoothing -> affine -> PWL
        y = y * y
        y = self.smooth_dw(y)
        y = self.mel_affine(y)

        k1 = tf.clip_by_value(self.w_k1, 0.0, 2.0)
        k2 = tf.clip_by_value(self.w_k2, 0.0, 2.0)
        t  = tf.clip_by_value(self.w_t,  0.0, 1.0)
        y = k1 * y + (k2 - k1) * tf.nn.relu(y - t)
        y = tf.maximum(0.0, tf.minimum(1.0, y))

        # temporal downsampling
        for pool in self.pool_time_layers:
            y = pool(y)

        # [B, mel_bins, spec_width, 1]
        y = tf.transpose(y, [0, 3, 1, 2])
        y = y[:, :, :self.spec_width, :]
        return y

    def get_config(self):
        return {
            "sample_rate": self.sample_rate,
            "chunk_duration": self.chunk_duration,
            "mel_bins": self.mel_bins,
            "spec_width": self.spec_width,
            "subband_width": self.subband_width,
            "name": self.name,
        }

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

def build_dscnn_model(num_mels, spec_width, sample_rate, chunk_duration, embeddings_size, num_classes, audio_frontend='librosa', alpha=1.0, depth_multiplier=1):
    """
    Depthwise-separable CNN adhering to STM32N6 constraints.
    - 3x3 depthwise with strides in both frequency (height) and time (width) where needed.
    - 1x1 pointwise to set channel count (multiples of 8).
    - Repeats scale with depth_multiplier; channels scale with alpha.
    """
    # Input + frontend
    if audio_frontend == 'tf':
        inputs = tf.keras.Input(shape=(chunk_duration * sample_rate, 1), name='raw_audio_input')
        #x = AudioFrontendLayer(sample_rate=sample_rate,
        #                       chunk_duration=chunk_duration,
        #                       mel_bins=num_mels,
        #                       spec_width=spec_width,
        #                       subband_width=6,
        #                       name="audio_frontend")(inputs)
        x = N6ConvFrontend(input_samples=chunk_duration * sample_rate,
                           n_bins=num_mels,
                           name="audio_frontend")(inputs)
    else:
        inputs = tf.keras.Input(shape=(num_mels, spec_width, 1), name='spectrogram_input')
        x = inputs
    # Stem (3x3, stride 1) to lift channels
    stem_ch = _make_divisible(int(24 * alpha), 8)
    x = layers.Conv2D(stem_ch, (3, 3), strides=(1, 2), padding='same', use_bias=False, name="stem_conv")(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(max_value=6, name="stem_relu")(x)

    # Stages: (filters, repeats, (stride_f, stride_t))
    # Use stride 2 in both axes early to reduce HxW; last stage keeps 1x1.
    base_filters = [24, 48, 96, 128]
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
    x = layers.Conv2D(emb_ch, (1, 1), strides=(1, 1), padding='same', use_bias=False, name="emb_conv")(x)
    x = layers.BatchNormalization(name="emb_bn")(x)
    x = layers.ReLU(max_value=6, name="emb_relu")(x)

    # Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation='sigmoid', name="pred")(x)
    return tf.keras.models.Model(inputs, outputs, name="dscnn_audio")

def train_model(model, train_dataset, val_dataset, epochs=50, learning_rate=0.001, batch_size=64, patience=5, checkpoint_path="best_model.h5", steps_per_epoch=None, val_steps=None):
    """
    Train the model with cosine-annealed learning rate, early stopping, and checkpointing.

    Uses Adam optimizer with a CosineDecayRestarts schedule; no ReduceLROnPlateau
    is used because the optimizer's learning rate is a schedule (not settable).

    Metrics:
      - ROC AUC (multi-label) monitored on the validation set for early stopping
        and model checkpointing (monitor='val_roc_auc', mode='max').

    Args:
        model: A tf.keras.Model.
        train_dataset: tf.data.Dataset yielding (inputs, labels) for training.
        val_dataset: tf.data.Dataset yielding (inputs, labels) for validation.
        epochs: Number of epochs to train.
        learning_rate: Initial learning rate for the cosine schedule.
        batch_size: Unused inside (kept for API consistency).
        patience: Early stopping patience (epochs).
        checkpoint_path: Where to save the best model.
        steps_per_epoch: Number of training steps per epoch (required by LR schedule).
        val_steps: Number of validation steps per epoch.

    Returns:
        tf.keras.callbacks.History: Training history.
    """

    if steps_per_epoch is None or steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be > 0")
    if val_steps is None or val_steps <= 0:
        val_steps = 1

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
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min'),
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

import argparse

def get_args():
    """
    Parse command-line arguments for training configuration.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train iNat-tiny audio classifier")
    parser.add_argument('--data_path_train', type=str, required=True, help='Path to train dataset')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples per class for training (None for all)')
    parser.add_argument('--sample_rate', type=int, default=22050, help='Audio sample rate. Default is 22050 Hz.')
    parser.add_argument('--num_mels', type=int, default=64, help='Number of mel bins for spectrogram')
    parser.add_argument('--spec_width', type=int, default=188, help='Spectrogram width')
    parser.add_argument('--chunk_duration', type=int, default=3, help='Audio chunk duration (seconds)')
    parser.add_argument('--max_duration', type=int, default=60, help='Max audio duration (seconds)')
    parser.add_argument('--audio_frontend', type=str, default='librosa', choices=['librosa', 'tf'], help='Audio frontend to use. Options: "librosa" or "tf", when using "tf" the fft will be part of the model.')
    parser.add_argument('--embeddings_size', type=int, default=256, help='Size of the final embeddings layer')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha for model scaling')
    parser.add_argument('--depth_multiplier', type=int, default=1, help='Depth multiplier for model')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup alpha')
    parser.add_argument('--mixup_probability', type=float, default=0.25, help='Fraction of batch to apply mixup')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/best_model.h5', help='Path to save best model')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Load file paths and classes
    file_paths, classes = load_file_paths_from_directory(args.data_path_train, 
                                                         max_samples=250, #args.max_samples, 
                                                         classes=get_classes_with_most_samples(args.data_path_train, 25, False) # DEBUG: Only use 25 classes for debugging
                                                         )
    
    # Perform sanity check on the dataset
    dataset_sanity_check(file_paths, classes, sample_rate=args.sample_rate, max_duration=args.max_duration, chunk_duration=args.chunk_duration, spec_width=args.spec_width, mel_bins=args.num_mels, audio_frontend=args.audio_frontend)

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
        mel_bins=args.num_mels
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
        mel_bins=args.num_mels
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
        embeddings_size=args.embeddings_size
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
    print(f"Training complete. Model saved to '{args.checkpoint_path}'.")
    
    # Save labels to txt file
    labels_file = args.checkpoint_path.replace('.h5', '_labels.txt')
    with open(labels_file, 'w') as f:
        for cls in classes:
            f.write(f"{cls}\n")

    # Post-training sanity check with trained TF frontend (mag_scale/BN) if applicable
    if args.audio_frontend == 'tf':
        print("Post-training sanity check using trained TF audio frontend...")
        trained_frontend = model.get_layer("audio_frontend")
        dataset_sanity_check(
            file_paths, classes,
            sample_rate=args.sample_rate,
            max_duration=args.max_duration,
            chunk_duration=args.chunk_duration,
            spec_width=args.spec_width,
            mel_bins=args.num_mels,
            audio_frontend='tf',
            tf_frontend_layer=trained_frontend
        )