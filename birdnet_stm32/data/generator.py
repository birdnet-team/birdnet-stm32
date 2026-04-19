"""Batch generator and tf.data.Dataset wrapper for training and validation.

Provides a Python generator that yields (inputs, one_hot_labels) batches,
and a tf.data.Dataset wrapper with static shape signatures.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf

from birdnet_stm32.audio.activity import pick_random_samples, sort_by_activity
from birdnet_stm32.audio.augmentation import apply_spec_augment
from birdnet_stm32.audio.io import load_audio_file
from birdnet_stm32.audio.spectrogram import get_spectrogram_from_audio
from birdnet_stm32.models.frontend import normalize_frontend_name

# Thread pool for parallel audio I/O within batches
_io_pool = ThreadPoolExecutor(max_workers=4)


def data_generator(
    file_paths: list[str],
    classes: list[str],
    batch_size: int = 32,
    audio_frontend: str = "librosa",
    sample_rate: int = 24000,
    max_duration: int = 30,
    chunk_duration: float = 3,
    spec_width: int = 128,
    mixup_alpha: float = 0.2,
    mixup_probability: float = 0.25,
    mel_bins: int = 48,
    fft_length: int = 512,
    mag_scale: str = "none",
    random_offset: bool = False,
    snr_threshold: float = 0.5,
    spec_augment: bool = False,
    freq_mask_max: int = 8,
    time_mask_max: int = 25,
    n_mfcc: int = 20,
):
    """Yield batches of (inputs, one_hot_labels) for training/validation.

    Frontends and input shapes:
        - precomputed/librosa: mel spectrogram -> [B, mel_bins, spec_width, 1]
        - hybrid: linear STFT magnitude -> [B, fft_bins, spec_width, 1]
        - raw/tf: waveform -> [B, T, 1], peak-normalized to [-1, 1]

    Args:
        file_paths: Audio file paths.
        classes: Ordered class names for one-hot encoding.
        batch_size: Batch size.
        audio_frontend: 'librosa' | 'hybrid' | 'raw' (deprecated: 'precomputed', 'tf').
        sample_rate: Sampling rate (Hz).
        max_duration: Max duration to read per file (seconds).
        chunk_duration: Chunk duration (seconds).
        spec_width: Target spectrogram width (frames).
        mixup_alpha: Mixup strength parameter.
        mixup_probability: Fraction of the batch to apply mixup to.
        mel_bins: Number of mel bins.
        fft_length: FFT size.
        mag_scale: 'pcen' | 'pwl' | 'db' | 'none'.
        random_offset: Randomly offset chunk start within file.
        snr_threshold: Minimum activity threshold for chunk selection.
        spec_augment: Apply SpecAugment (freq/time masking) to spectrograms.
        freq_mask_max: Maximum frequency mask width (bins) for SpecAugment.
        time_mask_max: Maximum time mask width (frames) for SpecAugment.

    Yields:
        Tuple of (inputs, labels) for a batch. Infinite generator.
    """
    audio_frontend = normalize_frontend_name(audio_frontend)
    T = int(sample_rate * chunk_duration)

    def _load_one(path):
        """Load and preprocess one audio file. Returns (sample, label_str) or None."""
        label_str = path.split("/")[-2]
        audio_chunks = load_audio_file(path, sample_rate, max_duration, chunk_duration, random_offset=True)
        if len(audio_chunks) == 0:
            audio_chunks = [np.random.uniform(-1.0, 1.0, size=(T,)).astype(np.float32)]
            label_str = "noise"

        if audio_frontend in ("mfcc", "log_mel"):
            specs = [
                get_spectrogram_from_audio(
                    chunk,
                    sample_rate,
                    n_fft=fft_length,
                    mel_bins=mel_bins,
                    spec_width=spec_width,
                    mag_scale="none",
                    mode=audio_frontend,
                    n_mfcc=n_mfcc,
                )
                for chunk in audio_chunks
            ]
            pool = sort_by_activity(specs, threshold=snr_threshold) or specs
            if len(pool) == 0:
                return None
            sample = pick_random_samples(pool, num_samples=1, pick_first=random_offset)
            sample = sample[0] if isinstance(sample, list) else sample

        elif audio_frontend == "librosa":
            specs = [
                get_spectrogram_from_audio(
                    chunk,
                    sample_rate,
                    n_fft=fft_length,
                    mel_bins=mel_bins,
                    spec_width=spec_width,
                    mag_scale=mag_scale,
                )
                for chunk in audio_chunks
            ]
            pool = sort_by_activity(specs, threshold=snr_threshold) or specs
            if len(pool) == 0:
                return None
            sample = pick_random_samples(pool, num_samples=1, pick_first=random_offset)
            sample = sample[0] if isinstance(sample, list) else sample

        elif audio_frontend == "hybrid":
            specs = [
                get_spectrogram_from_audio(chunk, sample_rate, n_fft=fft_length, mel_bins=-1, spec_width=spec_width)
                for chunk in audio_chunks
            ]
            pool = sort_by_activity(specs, threshold=snr_threshold) or specs
            if len(pool) == 0:
                return None
            sample = pick_random_samples(pool, num_samples=1, pick_first=random_offset)
            sample = sample[0] if isinstance(sample, list) else sample

        elif audio_frontend == "raw":
            pool = sort_by_activity(audio_chunks, threshold=snr_threshold) or audio_chunks
            if len(pool) == 0:
                return None
            sample = pick_random_samples(pool, num_samples=1, pick_first=random_offset)
            x = sample[0] if isinstance(sample, list) else sample
            x = x[:T]
            if x.shape[0] < T:
                x = np.pad(x, (0, T - x.shape[0]))
            x = x / (np.max(np.abs(x)) + 1e-6)
            sample = x
        else:
            raise ValueError(f"Invalid audio frontend: {audio_frontend}")

        return sample, label_str

    while True:
        idxs = np.random.permutation(len(file_paths))
        for batch_start in range(0, len(idxs), batch_size):
            batch_idxs = idxs[batch_start : batch_start + batch_size]
            batch_samples, batch_labels = [], []

            # Parallel audio loading
            paths = [file_paths[idx] for idx in batch_idxs]
            results = list(_io_pool.map(_load_one, paths))

            for result in results:
                if result is None:
                    continue
                sample, label_str = result

                # One-hot label; noise-like labels get all-zero vector
                if label_str.lower() in ("noise", "silence", "background", "other"):
                    one_hot_label = np.zeros(len(classes), dtype=np.float32)
                else:
                    if label_str not in classes:
                        continue
                    one_hot_label = tf.one_hot(classes.index(label_str), depth=len(classes)).numpy()

                # SpecAugment (only for spectrogram-based frontends)
                if spec_augment and audio_frontend in ("librosa", "hybrid", "mfcc", "log_mel"):
                    sample = apply_spec_augment(
                        sample,
                        freq_mask_max=freq_mask_max,
                        time_mask_max=time_mask_max,
                    )

                sample = np.expand_dims(sample, axis=-1)
                batch_samples.append(sample.astype(np.float32))
                batch_labels.append(one_hot_label.astype(np.float32))

            if len(batch_samples) == 0:
                continue
            batch_samples = np.stack(batch_samples)
            batch_labels = np.stack(batch_labels)

            # Mixup
            if mixup_alpha > 0 and mixup_probability > 0:
                num_mix = int(batch_samples.shape[0] * mixup_probability)
                if num_mix > 0:
                    mix_indices = np.random.choice(batch_samples.shape[0], size=num_mix, replace=False)
                    permuted_indices = np.random.permutation(batch_samples.shape[0])
                    lam = np.random.uniform(mixup_alpha, 1 - mixup_alpha, size=(num_mix,))
                    lam_inp = lam.reshape((num_mix,) + (1,) * (batch_samples.ndim - 1))
                    batch_samples[mix_indices] = (
                        lam_inp * batch_samples[mix_indices]
                        + (1 - lam_inp) * batch_samples[permuted_indices[mix_indices]]
                    )
                    batch_labels[mix_indices] = np.maximum(
                        batch_labels[mix_indices], batch_labels[permuted_indices[mix_indices]]
                    )

            yield batch_samples, batch_labels


def load_dataset(
    file_paths: list[str],
    classes: list[str],
    audio_frontend: str = "precomputed",
    batch_size: int = 32,
    spec_width: int = 128,
    mel_bins: int = 48,
    **kwargs,
) -> tf.data.Dataset:
    """Wrap the Python generator as a tf.data.Dataset with static shapes.

    Args:
        file_paths: Audio file paths.
        classes: Ordered class names.
        audio_frontend: 'librosa' | 'hybrid' | 'raw' (deprecated: 'precomputed', 'tf').
        batch_size: Batch size.
        spec_width: Target spectrogram width.
        mel_bins: Number of mel bins.
        **kwargs: Forwarded to data_generator (sample_rate, chunk_duration, etc.).

    Returns:
        Infinite tf.data.Dataset of (inputs, labels) with prefetching.
    """
    audio_frontend = normalize_frontend_name(audio_frontend)
    sr = kwargs.get("sample_rate", 16000)
    cd = kwargs.get("chunk_duration", 3)
    fft_length = kwargs.get("fft_length", 512)
    chunk_len = int(sr * cd)

    n_mfcc = kwargs.get("n_mfcc", 20)

    if audio_frontend == "mfcc":
        input_spec = tf.TensorSpec(shape=(None, n_mfcc, spec_width, 1), dtype=tf.float32)
    elif audio_frontend in ("librosa", "log_mel"):
        input_spec = tf.TensorSpec(shape=(None, mel_bins, spec_width, 1), dtype=tf.float32)
    elif audio_frontend == "hybrid":
        fft_bins = fft_length // 2 + 1
        input_spec = tf.TensorSpec(shape=(None, fft_bins, spec_width, 1), dtype=tf.float32)
    elif audio_frontend == "raw":
        input_spec = tf.TensorSpec(shape=(None, chunk_len, 1), dtype=tf.float32)
    else:
        raise ValueError(f"Invalid audio frontend: {audio_frontend}")

    output_signature = (input_spec, tf.TensorSpec(shape=(None, len(classes)), dtype=tf.float32))

    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(
            file_paths,
            classes,
            batch_size=batch_size,
            audio_frontend=audio_frontend,
            sample_rate=sr,
            max_duration=kwargs.get("max_duration", 30),
            chunk_duration=cd,
            spec_width=spec_width,
            mixup_alpha=kwargs.get("mixup_alpha", 0.0),
            mixup_probability=kwargs.get("mixup_probability", 0.0),
            mel_bins=mel_bins,
            fft_length=fft_length,
            n_mfcc=n_mfcc,
            mag_scale=kwargs.get("mag_scale", "none"),
            random_offset=kwargs.get("random_offset", False),
            spec_augment=kwargs.get("spec_augment", False),
            freq_mask_max=kwargs.get("freq_mask_max", 8),
            time_mask_max=kwargs.get("time_mask_max", 25),
        ),
        output_signature=output_signature,
    )
    return dataset.repeat().prefetch(tf.data.AUTOTUNE)
