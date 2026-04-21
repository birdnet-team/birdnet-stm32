"""Batch generator and tf.data.Dataset wrapper for training and validation.

Uses ``multiprocessing.Pool`` for true parallel audio loading and
preprocessing, bypassing the GIL so FLAC decode, resampling, smart-crop,
and spectrogram computation run concurrently across CPU cores.

Long files yield multiple salient chunks per open, stored in a shuffled
in-memory reservoir to maximize I/O reuse and batch diversity.
"""

import contextlib
import multiprocessing as mp
import random
import signal

import numpy as np
import tensorflow as tf

from birdnet_stm32.audio.activity import smart_crop, sort_by_activity
from birdnet_stm32.audio.augmentation import apply_mixup, apply_spec_augment
from birdnet_stm32.audio.io import load_audio_file
from birdnet_stm32.audio.spectrogram import get_spectrogram_from_audio
from birdnet_stm32.models.frontend import normalize_frontend_name

# ---------------------------------------------------------------------------
# Multiprocessing worker — module-level for pickling
# ---------------------------------------------------------------------------

_worker_cfg: dict = {}


def _init_worker(cfg: dict) -> None:
    """Initializer called once per worker process.

    Ignores SIGINT so only the main process handles Ctrl+C, preventing
    ``BrokenPipeError`` when workers try to write after the pool is torn down.
    """
    # Only set signal handler in actual child processes (not the main thread
    # fallback used by num_workers=0).
    import threading

    if threading.current_thread() is threading.main_thread():
        with contextlib.suppress(ValueError):
            signal.signal(signal.SIGINT, signal.SIG_IGN)
    global _worker_cfg  # noqa: PLW0603
    _worker_cfg = cfg


def _process_file(path: str):
    """Load and preprocess one audio file in a worker process.

    Returns a **list** of ``(sample, label)`` tuples (one per salient chunk),
    or ``None`` on failure / unknown class.  The number of chunks per file is
    controlled by ``max_chunks_per_file`` in the worker config.
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
    max_chunks = cfg["max_chunks_per_file"]

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
        audio_chunks = smart_crop(full_audio, sr, cd, max_chunks=max(5, max_chunks))

    # --- Compute spectrograms / raw features for all chunks ---
    if audio_frontend in ("mfcc", "log_mel"):
        features = [
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
    elif audio_frontend == "librosa":
        features = [
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
    elif audio_frontend == "hybrid":
        features = [
            get_spectrogram_from_audio(chunk, sr, n_fft=fft_length, mel_bins=-1, spec_width=spec_width)
            for chunk in audio_chunks
        ]
    elif audio_frontend == "raw":
        features = audio_chunks
    else:
        raise ValueError(f"Invalid audio frontend: {audio_frontend}")

    # Activity-sort: most salient first
    pool = sort_by_activity(features, threshold=snr_threshold) or features
    if not pool:
        return None

    # Take up to max_chunks salient items
    selected = pool[:max_chunks]

    results = []
    for item in selected:
        if audio_frontend == "raw":
            x = item[:T]
            if x.shape[0] < T:
                x = np.pad(x, (0, T - x.shape[0]))
            sample = x / (np.max(np.abs(x)) + 1e-6)
        else:
            sample = item

        if spec_augment and audio_frontend in ("librosa", "hybrid", "mfcc", "log_mel"):
            sample = apply_spec_augment(sample, freq_mask_max=freq_mask_max, time_mask_max=time_mask_max)

        sample = np.expand_dims(sample, axis=-1).astype(np.float32)
        results.append((sample, label))

    return results if results else None


def estimate_samples_per_epoch(n_files: int, max_chunks_per_file: int = 1) -> int:
    """Estimate the number of samples produced per full pass over the files.

    Short files produce 1 chunk, longer files up to ``max_chunks_per_file``.
    On average we estimate ``(1 + max_chunks_per_file) / 2`` samples per file.
    """
    avg = (1 + max_chunks_per_file) / 2.0
    return max(1, int(n_files * avg))


# Default reservoir capacity — number of ready samples to buffer.
_RESERVOIR_HIGH = 512
_RESERVOIR_LOW = 128


def load_dataset(
    file_paths: list[str],
    classes: list[str],
    audio_frontend: str = "hybrid",
    batch_size: int = 32,
    spec_width: int = 256,
    mel_bins: int = 64,
    num_workers: int = 8,
    max_chunks_per_file: int = 1,
    **kwargs,
) -> tf.data.Dataset:
    """Build a high-throughput tf.data pipeline with multiprocessing workers.

    Uses ``multiprocessing.Pool`` so FLAC decode, resampling, smart-crop,
    and spectrogram computation run in **separate processes**, bypassing the
    GIL entirely.

    When ``max_chunks_per_file > 1``, each file open extracts up to that many
    salient chunks, which are buffered in a shuffled in-memory reservoir.
    This dramatically reduces redundant I/O for long recordings (e.g. a 60 s
    file decoded once yields 3 usable chunks instead of 1).

    Args:
        file_paths: Audio file paths.
        classes: Ordered class names.
        audio_frontend: 'librosa' | 'hybrid' | 'raw' | 'mfcc' | 'log_mel'.
        batch_size: Batch size.
        spec_width: Target spectrogram width.
        mel_bins: Number of mel bins.
        num_workers: Number of worker processes (0 = single-process fallback).
        max_chunks_per_file: Max salient chunks to extract per file open.
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
    # Keep prefetch bounded to avoid RAM spikes with large raw batches.
    prefetch_batches = int(kwargs.get("prefetch_batches", 2))
    # Bound in-flight multiprocessing tasks so result queues cannot grow
    # unbounded during long epochs.
    max_inflight_files = int(kwargs.get("max_inflight_files", max(256, num_workers * 64)))
    loader_control = kwargs.get("loader_control")

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
        "max_chunks_per_file": max_chunks_per_file,
    }

    # Keep multiprocessing chunks small to avoid huge serialized payloads.
    # Large chunks can buffer many samples in worker->parent pipes and blow RAM.
    chunksize = 1 if audio_frontend == "raw" else 2

    use_mp = num_workers > 0

    reservoir_high = max(_RESERVOIR_HIGH, batch_size * 8)
    reservoir_low = max(_RESERVOIR_LOW, batch_size * 2)

    def _generator():
        """Infinite generator with reservoir for multi-chunk file reuse."""
        pool = None
        if use_mp:
            pool = mp.Pool(
                num_workers,
                initializer=_init_worker,
                initargs=(worker_cfg,),
                maxtasksperchild=100,
            )
        else:
            _init_worker(worker_cfg)

        try:
            while True:
                shuffled = list(file_paths)
                random.shuffle(shuffled)

                reservoir: list[tuple[np.ndarray, np.ndarray]] = []

                current_inflight = max_inflight_files
                if isinstance(loader_control, dict):
                    current_inflight = int(loader_control.get("max_inflight_files", max_inflight_files))
                current_inflight = max(32, current_inflight)

                for start in range(0, len(shuffled), current_inflight):
                    window = shuffled[start : start + current_inflight]
                    if pool is not None:
                        results_iter = pool.imap_unordered(_process_file, window, chunksize=chunksize)
                    else:
                        results_iter = map(_process_file, window)

                    for result in results_iter:
                        if result is None:
                            continue
                        reservoir.extend(result)

                        # Once reservoir is full enough, shuffle and drain.
                        if len(reservoir) >= reservoir_high:
                            random.shuffle(reservoir)
                            while len(reservoir) > reservoir_low:
                                yield reservoir.pop()

                # Drain remaining samples at end of epoch
                if reservoir:
                    random.shuffle(reservoir)
                    while reservoir:
                        yield reservoir.pop()
        except GeneratorExit:
            pass  # tf.data tearing down the generator — normal shutdown
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

    dataset = dataset.prefetch(max(1, prefetch_batches))
    return dataset
