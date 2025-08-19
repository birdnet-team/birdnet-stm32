import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
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
    
def dataset_sanity_check(file_paths, classes, sample_rate=16000, max_duration=30, chunk_duration=3, spec_width=128, mel_bins=64, audio_frontend='librosa', tf_frontend_layer=None):
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

def data_generator(file_paths, classes, batch_size=32, audio_frontend='librosa', sample_rate=16000, max_duration=30, chunk_duration=3, spec_width=128, mixup_alpha=0.2, mixup_probability=0.25, mel_bins=48):
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
                    print(f"File {path} has no valid audio chunks.")
                    continue
                
                if audio_frontend == 'librosa':
                    spectrograms = [get_spectrogram_from_audio(chunk, sample_rate, mel_bins=mel_bins, spec_width=spec_width) for chunk in audio_chunks]
                    sorted_specs = sort_by_s2n(spectrograms, threshold=0.5)                
                    random_spec = pick_random_samples(sorted_specs, num_samples=1)
                    sample = random_spec[0] if isinstance(random_spec, list) else random_spec
                elif audio_frontend == 'tf':
                    sorted_chunks = sort_by_s2n(audio_chunks, threshold=0.5)
                    random_chunk = pick_random_samples(sorted_chunks, num_samples=1)
                    sample = random_chunk[0] if isinstance(random_chunk, list) else random_chunk
                else:
                    raise ValueError("Invalid audio frontend. Choose 'librosa' or 'tf'.")

                label_str = path.split('/')[-2]
                if label_str.lower() in ['noise', 'silence', 'background', 'other']:
                    # Noise classes get all-zero one-hot labels
                    one_hot_label = np.zeros(len(classes), dtype=np.float32)
                else:
                    one_hot_label = tf.one_hot(classes.index(label_str), depth=len(classes)).numpy()
                
                sample = np.expand_dims(sample, axis=-1)
                batch_samples.append(sample.astype(np.float32))
                batch_labels.append(one_hot_label.astype(np.float32))

            batch_samples = np.stack(batch_samples)
            batch_labels = np.stack(batch_labels)

            # Mixup for both shapes (3D raw or 4D spectrogram)
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

class AudioFrontendLayer(tf.keras.layers.Layer):
    """
    Quantization-friendly audio frontend producing mel spectrogram tensors.

    Processing:
      1) STFT with Hann window: complex STFT -> magnitude.
      2) Normalize by Hann window L1 norm (amplitude-correct).
      3) Global compression on magnitudes:
           y = max(mag, 1e-4) ** (1 / (1 + exp(mag_scale)))    [scalar mag_scale]
      4) Linear mel projection using a 1x1 Conv1D initialized with
         tf.signal.linear_to_mel_weight_matrix.
      5) Optional affine clamp for display/range alignment:
           y = clip(out_gain * y + out_bias, 0, 1)
         (out_gain/out_bias are non-trainable and default to identity.)
      6) Pad/crop frames to spec_width and transpose to NHWC-like:
         Output shape [B, mel_bins, spec_width, 1].

    Trainable parameters:
      - mag_scale (scalar), if trainable_mag_scale=True.
      - mel projection weights, if trainable_mel=True.
      - out_gain/out_bias are non-trainable.

    Args:
        sample_rate: Audio sample rate.
        chunk_duration: Seconds per input chunk.
        mel_bins: Number of mel bins.
        spec_width: Number of output time frames.
        frame_length: STFT frame length.
        fft_length: FFT length (defines spectrogram bins).
        mag_scale_init: Initial value for mag_scale scalar.
        trainable_mag_scale: Whether mag_scale is learnable.
        trainable_mel: Whether mel projection weights are learnable.
        name: Layer name.

    Returns:
        Tensor of shape [B, mel_bins, spec_width, 1] suitable for CNN backbones.
    """
    def __init__(
        self,
        sample_rate,
        chunk_duration,
        mel_bins,
        spec_width,
        frame_length=1024,
        fft_length=1024,
        mag_scale_init=1.2,
        trainable_mag_scale=True,
        trainable_mel=True,
        name="audio_frontend",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.mel_bins = mel_bins
        self.spec_width = spec_width
        self.frame_length = frame_length
        self.fft_length = fft_length
        self.frame_step = (sample_rate * chunk_duration) // spec_width
        self.trainable_mel = trainable_mel

        self.mel_conv = tf.keras.layers.Conv1D(
            filters=self.mel_bins,
            kernel_size=1,
            use_bias=False,
            padding='valid',
            name="mel_conv",
            trainable=self.trainable_mel,
        )
        self.mag_scale_init = float(mag_scale_init)
        self.trainable_mag_scale = bool(trainable_mag_scale)
        self.mag_scale = None  # created in build

        # Affine clamp params (non-trainable; set by calibration)
        self.out_gain = self.add_weight(
            name="out_gain", shape=(), dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(1.0), trainable=False
        )
        self.out_bias = self.add_weight(
            name="out_bias", shape=(), dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(0.0), trainable=False
        )

        # Precompute Hann window L1 norm for normalization
        self._hann_l1 = None

    def build(self, input_shape):
        num_spec_bins = self.fft_length // 2 + 1

        self.mel_conv.build(tf.TensorShape([None, None, num_spec_bins]))
        mel_w = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.mel_bins,
            num_spectrogram_bins=num_spec_bins,
            sample_rate=self.sample_rate,
            lower_edge_hertz=150.0,
            upper_edge_hertz=self.sample_rate // 2,
        ).numpy()
        kernel = mel_w.reshape((1, num_spec_bins, self.mel_bins)).astype(np.float32)
        self.mel_conv.set_weights([kernel])

        self.mag_scale = self.add_weight(
            name="mag_scale",
            shape=(),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(self.mag_scale_init),
            trainable=self.trainable_mag_scale,
        )

        # Hann L1 norm (sum of window) as a constant scalar
        hann = tf.signal.hann_window(self.frame_length)
        self._hann_l1 = tf.reduce_sum(hann)
        super().build(input_shape)

    def call(self, inputs, training=None):
        # inputs: [B, T, 1]
        x1d = tf.squeeze(inputs, axis=-1)  # [B, T]

        stft = tf.signal.stft(
            x1d,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.fft_length,
            window_fn=tf.signal.hann_window,
            pad_end=True,
        )  # [B, frames, fft_bins]
        mag = tf.abs(stft)

        # Normalize by Hann L1 norm (amplitude-correct)
        mag = mag / tf.cast(self._hann_l1, tf.float32)

        # Scalar power compression
        alpha = tf.math.reciprocal(1.0 + tf.math.exp(self.mag_scale))  # scalar
        mag = tf.math.pow(tf.maximum(mag, 1e-4), alpha)

        # FFT -> mel (linear)
        x = self.mel_conv(mag)  # [B, frames, mel_bins]

        # Optional affine clamp (defaults to identity)
        x = self.out_gain * x + self.out_bias
        x = tf.clip_by_value(x, 0.0, 1.0)

        # Pad/crop frames to spec_width
        frames = tf.shape(x)[1]
        pad = tf.maximum(0, self.spec_width - frames)
        x = tf.pad(x, [[0, 0], [0, pad], [0, 0]])
        x = x[:, : self.spec_width, :]

        x = tf.expand_dims(x, axis=-1)        # [B, spec_width, mel_bins, 1]
        x = tf.transpose(x, [0, 2, 1, 3])     # [B, mel_bins, spec_width, 1]
        return x

    def get_config(self):
        return {
            "sample_rate": self.sample_rate,
            "chunk_duration": self.chunk_duration,
            "mel_bins": self.mel_bins,
            "spec_width": self.spec_width,
            "frame_length": self.frame_length,
            "fft_length": self.fft_length,
            "mag_scale_init": self.mag_scale_init,
            "trainable_mag_scale": self.trainable_mag_scale,
            "trainable_mel": self.trainable_mel,
            "name": self.name,
        }

def _make_divisible(v, divisor=8):
    """
    Round channel/width values to be divisible by 'divisor' (>= divisor).

    This is useful for TFLite/int8 kernels that prefer channel counts
    aligned to 8 or 16 for better performance.
    """
    v = int(v + divisor / 2) // divisor * divisor
    return max(divisor, v)

def build_tiny_model(
    num_mels, spec_width, sample_rate, chunk_duration, num_classes,
    audio_frontend='librosa',
    alpha=0.75,
    depth_multiplier=2,
    embeddings_size=192,
    use_residual=True,
    factorized_dw=True,
    conv_head=True,
    stem_channels=16,
    bottleneck_ratio=0.25,
    round_to=8,
):
    """
    Build a compact CNN for audio classification with fewer parameters.

    Frontends:
      - 'librosa': model input is a mel spectrogram [mel_bins, spec_width, 1].
      - 'tf':      model input is raw audio [T, 1]; AudioFrontendLayer produces mel.

    Architecture:
      - Lightweight stem Conv2D with BatchNorm + ReLU6.
      - Repeated bottlenecked depthwise-separable blocks:
          1x1 reduce -> Depthwise (factorized 1x3 + 3x1 or 3x3) -> 1x1 project.
      - Optional residual connections on non-downsampling blocks.
      - Final block to reach desired embeddings_size.
      - Head:
          conv_head=True  -> 1x1 conv to classes, GlobalAveragePooling2D, sigmoid.
          conv_head=False -> GAP, Dropout, Dense to classes, sigmoid.

    Args:
        num_mels: Number of mel bins.
        spec_width: Number of time frames.
        sample_rate: Sample rate for raw-audio input (tf frontend).
        chunk_duration: Seconds per raw-audio input (tf frontend).
        num_classes: Number of output classes.
        audio_frontend: 'librosa' or 'tf'.
        alpha: Width multiplier for channels.
        depth_multiplier: Number of block stages (depth).
        embeddings_size: Channel size before the head.
        use_residual: Enable residual in non-strided blocks if shapes match.
        factorized_dw: Use (1x3)+(3x1) depthwise factorization.
        conv_head: Use fully-convolutional classification head.
        stem_channels: Base channels of the stem before applying alpha/rounding.
        bottleneck_ratio: Ratio for 1x1 reduction inside blocks.
        round_to: Round channel counts to multiples of this value.

    Returns:
        A compiled tf.keras.Model ready for training.
    """
    def ds_bottleneck_block(x, out_filters, residual=False, strides=(1, 1), factorized=True, bn_ratio=0.25):
        in_c = int(x.shape[-1])
        out_c = _make_divisible(out_filters, round_to)
        mid_c = _make_divisible(max(8, int(min(in_c, out_c) * bn_ratio)), round_to)

        # 1x1 reduce
        y = tf.keras.layers.Conv2D(mid_c, (1, 1), padding='same', use_bias=False)(x)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.ReLU(max_value=6)(y)

        # Depthwise (optionally factorized)
        if factorized:
            y = tf.keras.layers.DepthwiseConv2D((1, 3), padding='same', use_bias=False, strides=strides)(y)
            y = tf.keras.layers.BatchNormalization()(y)
            y = tf.keras.layers.ReLU(max_value=6)(y)
            y = tf.keras.layers.DepthwiseConv2D((3, 1), padding='same', use_bias=False)(y)
            y = tf.keras.layers.BatchNormalization()(y)
            y = tf.keras.layers.ReLU(max_value=6)(y)
        else:
            y = tf.keras.layers.DepthwiseConv2D((3, 3), padding='same', use_bias=False, strides=strides)(y)
            y = tf.keras.layers.BatchNormalization()(y)
            y = tf.keras.layers.ReLU(max_value=6)(y)

        # 1x1 project
        y = tf.keras.layers.Conv2D(out_c, (1, 1), padding='same', use_bias=False)(y)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.ReLU(max_value=6)(y)

        if residual and strides == (1, 1) and in_c == out_c:
            y = tf.keras.layers.Add()([y, x])
        return y

    # Inputs and frontend
    if audio_frontend == 'tf':
        inputs = tf.keras.Input(shape=(chunk_duration * sample_rate, 1), name='raw_audio_input')
        x = AudioFrontendLayer(sample_rate, chunk_duration, num_mels, spec_width, name="audio_frontend")(inputs)
    elif audio_frontend == 'librosa':
        inputs = tf.keras.Input(shape=(num_mels, spec_width, 1), name='spectrogram_input')
        x = inputs
    else:
        raise ValueError("Invalid audio frontend. Choose 'librosa' or 'tf'.")

    # Stem
    stem_c = _make_divisible(stem_channels * alpha, round_to)
    x = tf.keras.layers.Conv2D(stem_c, (3, 3), strides=2, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6)(x)

    # Stages
    c = _make_divisible(48 * alpha, round_to)  # start lower than 64
    for i in range(depth_multiplier):
        # Downsample block
        x = ds_bottleneck_block(x, c, residual=False, strides=(2, 2), factorized=factorized_dw, bn_ratio=bottleneck_ratio)
        # Non-downsampling blocks
        x = ds_bottleneck_block(x, c, residual=use_residual, strides=(1, 1), factorized=factorized_dw, bn_ratio=bottleneck_ratio)
        x = ds_bottleneck_block(x, c, residual=use_residual, strides=(1, 1), factorized=factorized_dw, bn_ratio=bottleneck_ratio)
        c = _make_divisible(min(c * 2, 256 * alpha), round_to)

    # Final block
    x = ds_bottleneck_block(x, embeddings_size, residual=False, strides=(2, 2), factorized=factorized_dw, bn_ratio=bottleneck_ratio)

    # Head
    if conv_head:
        x = tf.keras.layers.Conv2D(embeddings_size, (1, 1), padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU(max_value=6)(x)
        logits = tf.keras.layers.Conv2D(num_classes, (1, 1), padding='same', use_bias=True)(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(logits)
        outputs = tf.keras.layers.Activation('sigmoid')(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)

    return tf.keras.models.Model(inputs, outputs)

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

    # Cosine annealing learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=learning_rate,
        first_decay_steps=steps_per_epoch * epochs // 4,
        t_mul=2.0,
        m_mul=1.0,
        alpha=1e-4
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='binary_crossentropy',
        metrics=[
            # tf.keras.metrics.AUC(curve='PR', multi_label=True, name="pr_auc"),
            tf.keras.metrics.AUC(curve='ROC', multi_label=True, name="roc_auc"),
        ]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_roc_auc', patience=patience, restore_best_weights=True, mode='max'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor='val_roc_auc', save_best_only=True, mode='max'
        ),
        # Removed ReduceLROnPlateau because optimizer uses a LearningRateSchedule
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
    parser.add_argument('--num_mels', type=int, default=64, help='Number of mel bins for spectrogram')
    parser.add_argument('--spec_width', type=int, default=128, help='Spectrogram width')
    parser.add_argument('--chunk_duration', type=int, default=3, help='Audio chunk duration (seconds)')
    parser.add_argument('--max_duration', type=int, default=30, help='Max audio duration (seconds)')
    parser.add_argument('--audio_frontend', type=str, default='librosa', choices=['librosa', 'tf'], help='Audio frontend to use. Options: "librosa" or "tf", when using "tf" the fft will be part of the model.')
    parser.add_argument('--embeddings_size', type=int, default=256, help='Size of the final embeddings layer')
    parser.add_argument('--alpha', type=float, default=0.75, help='Alpha for model scaling')
    parser.add_argument('--depth_multiplier', type=int, default=2, help='Depth multiplier for model')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup alpha')
    parser.add_argument('--mixup_probability', type=float, default=0.25, help='Fraction of batch to apply mixup')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/best_model.h5', help='Path to save best model')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Load file paths and classes
    file_paths, classes = load_file_paths_from_directory(args.data_path_train, max_samples=args.max_samples)
    
    # Perform sanity check on the dataset
    dataset_sanity_check(file_paths, classes, sample_rate=16000, max_duration=args.max_duration, chunk_duration=args.chunk_duration, spec_width=args.spec_width, mel_bins=args.num_mels, audio_frontend=args.audio_frontend)

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
        sample_rate=16000,
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
        sample_rate=16000,
        max_duration=args.max_duration,
        chunk_duration=args.chunk_duration,
        spec_width=args.spec_width,
        batch_size=args.batch_size,
        mixup_alpha=0.0,
        mixup_probability=0.0,
        mel_bins=args.num_mels
    )

    # Update steps_per_epoch and val_steps
    steps_per_epoch = len(train_paths) // args.batch_size
    val_steps = len(val_paths) // args.batch_size

    # Build model
    print("Building model...")
    model = build_tiny_model(
        num_mels=args.num_mels,
        spec_width=args.spec_width,
        sample_rate=16000,
        chunk_duration=args.chunk_duration,
        audio_frontend=args.audio_frontend,
        num_classes=len(classes),
        alpha=args.alpha,
        depth_multiplier=args.depth_multiplier,
        use_residual=True,
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
        patience=args.patience,
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
            sample_rate=16000,
            max_duration=args.max_duration,
            chunk_duration=args.chunk_duration,
            spec_width=args.spec_width,
            mel_bins=args.num_mels,
            audio_frontend='tf',
            tf_frontend_layer=trained_frontend
        )