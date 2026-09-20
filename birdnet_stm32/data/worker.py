"""Per-file loading and preprocessing, run inside the loader's worker processes.

Kept free of TensorFlow on purpose. The worker pool uses the ``forkserver``
start method, whose server preloads only this module, so workers start from a
small single-threaded process instead of a fork of the TensorFlow trainer.
"""

import contextlib
import signal

import numpy as np

from birdnet_stm32.audio.activity import smart_crop, sort_by_activity, uniform_crop
from birdnet_stm32.audio.augmentation import apply_spec_augment, apply_time_mask
from birdnet_stm32.audio.io import estimate_num_chunks, load_audio_window, split_audio_into_chunks
from birdnet_stm32.audio.spectrogram import get_spectrogram_from_audio

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
    # One BLAS thread per worker: the pool provides the parallelism. Left at the
    # default, every worker starts a full OpenBLAS pool for the mel matrix
    # product, and 8 workers x 16 threads thrash a 16-core host.
    with contextlib.suppress(ImportError):
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
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
    compression = cfg.get("input_compression", "none")
    load_duration = cfg.get("load_duration", cfg.get("max_duration"))
    snr_threshold = cfg["snr_threshold"]
    random_offset = cfg["random_offset"]
    spec_augment = cfg["spec_augment"]
    raw_time_masks = int(cfg.get("raw_time_masks", 0))
    raw_time_mask_ms = float(cfg.get("raw_time_mask_ms", 30.0))
    freq_mask_max = cfg["freq_mask_max"]
    time_mask_max = cfg["time_mask_max"]
    audio_frontend = cfg["audio_frontend"]
    max_chunks = cfg["max_chunks_per_file"]
    candidate_chunks = cfg.get("candidate_chunks_per_file", min(8, max(4, max_chunks * 2)))

    try:
        audio = load_audio_window(
            path,
            sample_rate=sr,
            max_duration=load_duration,
            chunk_duration=cd,
            random_offset=random_offset,
        )
    except Exception:
        return None

    if audio.size == 0:
        return None

    crop_policy = cfg.get("crop_policy", "energy")
    available_chunks = estimate_num_chunks(audio.shape[0], sr, cd)
    if crop_policy == "uniform":
        audio_chunks = list(uniform_crop(audio, sr, cd, max_chunks=candidate_chunks))
    elif available_chunks > candidate_chunks:
        audio_chunks = list(smart_crop(audio, sr, cd, max_chunks=candidate_chunks))
    else:
        audio_chunks = list(split_audio_into_chunks(audio, sample_rate=sr, chunk_duration=cd))

    if len(audio_chunks) == 0:
        return None

    # --- Compute spectrograms / raw features for all chunks ---
    # Chunks are ranked on the uncompressed spectrogram whatever the model input
    # is, so input compression changes the representation and not which chunks
    # a file contributes.
    if audio_frontend == "librosa":
        pairs = [
            get_spectrogram_from_audio(
                chunk,
                sr,
                n_fft=fft_length,
                mel_bins=mel_bins,
                spec_width=spec_width,
                mag_scale=mag_scale,
                compression=compression,
                with_uncompressed=True,
            )
            for chunk in audio_chunks
        ]
    elif audio_frontend == "hybrid":
        pairs = [
            get_spectrogram_from_audio(
                chunk,
                sr,
                n_fft=fft_length,
                mel_bins=-1,
                spec_width=spec_width,
                compression=compression,
                with_uncompressed=True,
            )
            for chunk in audio_chunks
        ]
    elif audio_frontend == "raw":
        pairs = [(chunk, chunk) for chunk in audio_chunks]
    else:
        raise ValueError(f"Invalid audio frontend: {audio_frontend}")
    features = [ranked for _, ranked in pairs]
    model_input = {id(ranked): item for item, ranked in pairs}

    # Activity-sort: most salient first. Under the uniform policy the ranking is
    # skipped too, since re-ranking uniformly drawn chunks by energy would put
    # the very bias back that the policy exists to remove.
    if crop_policy == "uniform":
        pool = list(features)
        np.random.shuffle(pool)
    else:
        pool = sort_by_activity(features, threshold=snr_threshold) or features
    if not pool:
        return None
    pool = [model_input[id(ranked)] for ranked in pool]

    # Take up to max_chunks salient items
    selected = pool[:max_chunks]

    results = []
    for item in selected:
        if audio_frontend == "raw":
            x = item[:T]
            if x.shape[0] < T:
                x = np.pad(x, (0, T - x.shape[0]))
            sample = x / (np.max(np.abs(x)) + 1e-6)
            # After normalization, so a mask cannot change the scaling peak.
            if raw_time_masks > 0:
                sample = apply_time_mask(sample, sr, num_masks=raw_time_masks, max_width_ms=raw_time_mask_ms)
        else:
            sample = item

        if spec_augment and audio_frontend in ("librosa", "hybrid"):
            sample = apply_spec_augment(sample, freq_mask_max=freq_mask_max, time_mask_max=time_mask_max)

        sample = np.expand_dims(sample, axis=-1).astype(np.float32)
        results.append((sample, label))

    return results if results else None
