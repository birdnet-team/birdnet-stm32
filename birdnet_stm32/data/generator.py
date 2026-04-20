"""Batch generator and tf.data.Dataset wrapper for training and validation.

Uses ``multiprocessing.Pool`` for true parallel audio loading and
preprocessing, bypassing the GIL so FLAC decode, resampling, smart-crop,
and spectrogram computation run concurrently across CPU cores.
"""

import multiprocessing as mp
import random

import numpy as np
import tensorflow as tf

from birdnet_stm32.audio.activity import pick_random_samples, smart_crop, sort_by_activity
from birdnet_stm32.audio.augmentation import apply_mixup, apply_spec_augment
from birdnet_stm32.audio.io import load_audio_file
from birdnet_stm32.audio.spectrogram import get_spectrogram_from_audio
from birdnet_stm32.models.frontend import normalize_frontend_name

# ---------------------------------------------------------------------------
# Multiprocessing worker — module-level for pickling
# ---------------------------------------------------------------------------

_worker_cfg: dict = {}


def _init_worker(cfg: dict) -> None:
    """Initializer called once per worker process."""
    global _worker_cfg  # noqa: PLW0603
    _worker_cfg = cfg


def _process_file(path: str):
    """Load and preprocess one audio file in a worker process.

    Returns ``(sample, label)`` or ``None`` on failure / unknown class.
    """
    cfg = _worker_cfg
    label_str = path.split("/")[-2]

    # --- label ---
    noise_labels = cfg["noise_labels"]
    class_to_idx = cfg["class_to_idx"]
    num_classes = cfg["num_classes"]

    if label_str.lower() in noise_labels:
        label = np.zeros(num_classes, dtype=np.float32)
    elif label_str in class_to_idx:
        label = np.zeros(num_classes, dtype=np.float32)
        label[class_to_idx[label_str]] = 1.0
    else:
        return None  # unknown class

    sr = cfg["sr"]
    cd = cfg["cd"]
    T = cfg["T"]
    fft_length = cfg["fft_length"]
    mel_bins = cfg["mel_bins"]
    spec_width = cfg["spec_width"]
    mag_scale = cfg["mag_scale"]
    n_mfcc = cfg["n_mfcc"]
    max_duration = cfg["max_duration"]
    snr_threshold = cfg["snr_threshold"]
    random_offset = cfg["random_offset"]
    spec_augment = cfg["spec_augment"]
    freq_mask_max = cfg["freq_mask_max"]
    time_mask_max = cfg["time_mask_max"]
    audio_frontend = cfg["audio_frontend"]

    try:
        audio_chunks = load_audio_file(path, sr, max_duration, cd, random_offset=random_offset)
    except Exception:
        return None

    if len(audio_chunks) == 0:
        audio_chunks = [np.random.uniform(-1.0, 1.0, size=(T,)).astype(np.float32)]
        label = np.zeros(num_classes, dtype=np.float32)

    # Smart crop for long recordings
    if len(audio_chunks) > 2:
        full_audio = np.concatenate(audio_chunks)
        audio_chunks = smart_crop(full_audio, sr, cd, max_chunks=5)

    if audio_frontend in ("mfcc", "log_mel"):
        specs = [
            get_spectrogram_from_audio(
                chunk,
                sr,
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
        if not pool:
            return None
        sample = pick_random_samples(pool, num_samples=1, pick_first=random_offset)
        sample = sample[0] if isinstance(sample, list) else sample

    elif audio_frontend == "librosa":
        specs = [
            get_spectrogram_from_audio(
                chunk,
                sr,
                n_fft=fft_length,
                mel_bins=mel_bins,
                spec_width=spec_width,
                mag_scale=mag_scale,
            )
            for chunk in audio_chunks
        ]
        pool = sort_by_activity(specs, threshold=snr_threshold) or specs
        if not pool:
            return None
        sample = pick_random_samples(pool, num_samples=1, pick_first=random_offset)
        sample = sample[0] if isinstance(sample, list) else sample

    elif audio_frontend == "hybrid":
        specs = [
            get_spectrogram_from_audio(chunk, sr, n_fft=fft_length, mel_bins=-1, spec_width=spec_width)
            for chunk in audio_chunks
        ]
        pool = sort_by_activity(specs, threshold=snr_threshold) or specs
        if not pool:
            return None
        sample = pick_random_samples(pool, num_samples=1, pick_first=random_offset)
        sample = sample[0] if isinstance(sample, list) else sample

    elif audio_frontend == "raw":
        pool = sort_by_activity(audio_chunks, threshold=snr_threshold) or audio_chunks
        if not pool:
            return None
        sample = pick_random_samples(pool, num_samples=1, pick_first=random_offset)
        x = sample[0] if isinstance(sample, list) else sample
        x = x[:T]
        if x.shape[0] < T:
            x = np.pad(x, (0, T - x.shape[0]))
        sample = x / (np.max(np.abs(x)) + 1e-6)
    else:
        raise ValueError(f"Invalid audio frontend: {audio_frontend}")

    # SpecAugment
    if spec_augment and audio_frontend in ("librosa", "hybrid", "mfcc", "log_mel"):
        sample = apply_spec_augment(sample, freq_mask_max=freq_mask_max, time_mask_max=time_mask_max)

    sample = np.expand_dims(sample, axis=-1).astype(np.float32)
    return sample, label


def load_dataset(
    file_paths: list[str],
    classes: list[str],
    audio_frontend: str = "hybrid",
    batch_size: int = 32,
    spec_width: int = 256,
    mel_bins: int = 64,
    num_workers: int = 8,
    **kwargs,
) -> tf.data.Dataset:
    """Build a high-throughput tf.data pipeline with multiprocessing workers.

    Uses ``multiprocessing.Pool`` so FLAC decode, resampling, smart-crop,
    and spectrogram computation run in **separate processes**, bypassing the
    GIL entirely.  Results stream into ``tf.data.from_generator`` for
    batching, mixup, and GPU prefetch.

    Args:
        file_paths: Audio file paths.
        classes: Ordered class names.
        audio_frontend: 'librosa' | 'hybrid' | 'raw' | 'mfcc' | 'log_mel'.
        batch_size: Batch size.
        spec_width: Target spectrogram width.
        mel_bins: Number of mel bins.
        num_workers: Number of worker processes (0 = single-process fallback).
        **kwargs: Forwarded to loading logic (sample_rate, chunk_duration, etc.).

    Returns:
        Infinite tf.data.Dataset of (inputs, labels) with prefetching.
    """
    audio_frontend = normalize_frontend_name(audio_frontend)
    sr = kwargs.get("sample_rate", 24000)
    cd = kwargs.get("chunk_duration", 3)
    fft_length = kwargs.get("fft_length", 512)
    chunk_len = int(sr * cd)
    n_mfcc = kwargs.get("n_mfcc", 20)
    mag_scale = kwargs.get("mag_scale", "pwl")
    max_duration = kwargs.get("max_duration", 60)
    snr_threshold = kwargs.get("snr_threshold", 0.5)
    random_offset = kwargs.get("random_offset", False)
    spec_augment = kwargs.get("spec_augment", False)
    freq_mask_max = kwargs.get("freq_mask_max", 8)
    time_mask_max = kwargs.get("time_mask_max", 25)
    mixup_alpha = kwargs.get("mixup_alpha", 0.2)
    mixup_probability = kwargs.get("mixup_probability", 0.25)

    num_classes = len(classes)

    # Determine output shapes
    if audio_frontend == "mfcc":
        sample_shape = (n_mfcc, spec_width, 1)
    elif audio_frontend in ("librosa", "log_mel"):
        sample_shape = (mel_bins, spec_width, 1)
    elif audio_frontend == "hybrid":
        sample_shape = (fft_length // 2 + 1, spec_width, 1)
    elif audio_frontend == "raw":
        sample_shape = (chunk_len, 1)
    else:
        raise ValueError(f"Invalid audio frontend: {audio_frontend}")

    # Worker config (picklable dict — no closures)
    worker_cfg = {
        "audio_frontend": audio_frontend,
        "sr": sr,
        "cd": cd,
        "T": chunk_len,
        "fft_length": fft_length,
        "mel_bins": mel_bins,
        "spec_width": spec_width,
        "mag_scale": mag_scale,
        "n_mfcc": n_mfcc,
        "max_duration": max_duration,
        "snr_threshold": snr_threshold,
        "random_offset": random_offset,
        "spec_augment": spec_augment,
        "freq_mask_max": freq_mask_max,
        "time_mask_max": time_mask_max,
        "noise_labels": ("noise", "silence", "background", "other"),
        "class_to_idx": {c: i for i, c in enumerate(classes)},
        "num_classes": num_classes,
    }

    # Choose chunksize: enough to amortize IPC, small enough for fairness
    chunksize = max(1, min(64, len(file_paths) // max(num_workers, 1) // 4))

    use_mp = num_workers > 0

    def _generator():
        """Infinite generator streaming results from the worker pool."""
        pool = None
        if use_mp:
            pool = mp.Pool(num_workers, initializer=_init_worker, initargs=(worker_cfg,))
        else:
            _init_worker(worker_cfg)

        try:
            while True:
                shuffled = list(file_paths)
                random.shuffle(shuffled)

                if pool is not None:
                    results = pool.imap_unordered(_process_file, shuffled, chunksize=chunksize)
                else:
                    results = map(_process_file, shuffled)

                for result in results:
                    if result is not None:
                        yield result
        finally:
            if pool is not None:
                pool.terminate()
                pool.join()

    output_sig = (
        tf.TensorSpec(shape=sample_shape, dtype=tf.float32),
        tf.TensorSpec(shape=(num_classes,), dtype=tf.float32),
    )

    dataset = tf.data.Dataset.from_generator(_generator, output_signature=output_sig)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Mixup on batches
    if mixup_alpha > 0 and mixup_probability > 0:

        def _apply_batch_mixup(samples, labels):
            mixed_s, mixed_l = tf.py_function(
                lambda s, lb: apply_mixup(s.numpy(), lb.numpy(), alpha=mixup_alpha, probability=mixup_probability),
                [samples, labels],
                [tf.float32, tf.float32],
            )
            mixed_s.set_shape(samples.shape)
            mixed_l.set_shape(labels.shape)
            return mixed_s, mixed_l

        dataset = dataset.map(_apply_batch_mixup, num_parallel_calls=1)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
