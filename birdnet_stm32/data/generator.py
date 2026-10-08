"""Batch generator and tf.data.Dataset wrapper for training and validation.

Uses ``multiprocessing.Pool`` for true parallel audio loading and
preprocessing, bypassing the GIL so FLAC decode, resampling, smart-crop,
and spectrogram computation run concurrently across CPU cores.

Long files yield multiple salient chunks per open, stored in a shuffled
in-memory reservoir to maximize I/O reuse and batch diversity.
"""

import multiprocessing as mp
import os
import random
import time
from typing import Any

import numpy as np
import tensorflow as tf

from birdnet_stm32.data.dataset import NOISE_CLASSES
from birdnet_stm32.data.worker import _init_worker, _process_file
from birdnet_stm32.models.frontend import hybrid_fft_bins, normalize_frontend_name

# ---------------------------------------------------------------------------
# Multiprocessing worker pool
# ---------------------------------------------------------------------------

_POOL_POLL_INTERVAL_S = 0.05


def mixup_batch(samples: tf.Tensor, labels: tf.Tensor, alpha: float = 0.2, probability: float = 0.25):
    """Multi-source mixup on a batch, in TensorFlow ops (runs inside tf.data, off the Python lock).

    Emulates several birds calling at once: ``floor(B * probability)`` random samples of the batch
    are each replaced by a blend of themselves and 1 or 2 random partners (even odds), with gains
    from a Dirichlet(``alpha``) distribution (normalized Gamma draws), and their labels become the
    element-wise max over the sources so every contributing species stays a target. Every blend
    reads the original batch.

    Returns:
        ``(samples, labels, mixed)``: ``mixed`` is a float32 ``[B]`` mask of the replaced samples
        (the teacher embedding of such a chunk no longer describes its audio).
    """
    b = tf.shape(samples)[0]
    if alpha <= 0 or probability <= 0:
        return samples, labels, tf.zeros([b], tf.float32)
    num_mix = tf.cast(tf.floor(tf.cast(b, tf.float32) * probability), tf.int32)
    chosen = tf.random.shuffle(tf.range(b))[:num_mix]
    mixed = tf.scatter_nd(chosen[:, None], tf.ones([num_mix], tf.float32), [b])
    sources = tf.concat([tf.range(b)[:, None], tf.random.uniform([b, 2], maxval=b, dtype=tf.int32)], axis=1)
    third = tf.cast(tf.random.uniform([b]) < 0.5, tf.float32)
    active = tf.stack([tf.ones([b]), tf.ones([b]), third], axis=1)
    gains = tf.random.gamma([b, 3], alpha) * active + 1e-12 * active
    gains = gains / tf.reduce_sum(gains, axis=1, keepdims=True)
    extra = tf.ones([tf.rank(samples) - 1], tf.int32)
    blend = tf.reduce_sum(tf.reshape(gains, tf.concat([[b, 3], extra], 0)) * tf.gather(samples, sources), axis=1)
    merged = tf.reduce_max(tf.gather(labels, sources) * active[:, :, None], axis=1)
    m = tf.reshape(mixed, tf.concat([[b], extra], 0))
    return m * blend + (1.0 - m) * samples, mixed[:, None] * merged + (1.0 - mixed[:, None]) * labels, mixed


def _pool_context():
    """Start workers from a clean forkserver rather than by forking the trainer.

    Forking the training process copies a snapshot of dozens of TensorFlow and
    loader threads. A child forked while one of them held a lock (logging,
    malloc, BLAS) hangs on first use, and with ``maxtasksperchild`` respawning
    every worker every 100 files, some runs lost most of their pool: measured,
    the same seeded run trained at 55 or at 390 ms/step with the GPU at 6%.
    The forkserver is single-threaded and preloads only the TF-free worker
    module, so a replacement worker (after a timeout) cannot inherit a held lock.
    """
    ctx = mp.get_context("forkserver")
    ctx.set_forkserver_preload(["birdnet_stm32.data.worker"])
    return ctx


def _create_worker_pool(num_workers: int, worker_cfg: dict) -> mp.pool.Pool:
    """Create a worker pool with the project's standard settings."""
    return _pool_context().Pool(
        num_workers,
        initializer=_init_worker,
        initargs=(worker_cfg,),
        # Workers live for the whole run. Recycling them every 100 files cost more
        # than the files did (~0.5 s to re-open the teacher cache against ~5 ms a
        # file) and capped the loader at ~10 batches/s; a worker's private memory
        # stays flat (measured over 4,000 files), the growth in its RSS is the
        # shared, memory-mapped teacher cache.
        maxtasksperchild=None,
    )


def _terminate_worker_pool(pool: mp.pool.Pool | None) -> None:
    """Terminate a worker pool if it exists."""
    if pool is not None:
        pool.terminate()
        pool.join()


def estimate_samples_per_epoch(n_files: int, max_chunks_per_file: int = 1) -> int:
    """Estimate the number of samples produced per full pass over the files.

    Short files produce 1 chunk, longer files up to ``max_chunks_per_file``.
    On average we estimate ``(1 + max_chunks_per_file) / 2`` samples per file.
    """
    avg = (1 + max_chunks_per_file) / 2.0
    return max(1, int(n_files * avg))


# Default reservoir capacity — number of ready samples to buffer.
_DEFAULT_BUFFER_MB = 128.0
_MAX_RESERVOIR_SAMPLES = 1024


def _estimate_sample_bytes(sample_shape: tuple[int, ...], num_classes: int) -> int:
    """Estimate bytes per buffered sample, including labels."""
    sample_elems = int(np.prod(sample_shape, dtype=np.int64))
    return (sample_elems + int(num_classes)) * np.dtype(np.float32).itemsize


def _compute_reservoir_limits(
    sample_shape: tuple[int, ...],
    num_classes: int,
    batch_size: int,
    loader_buffer_mb: float,
) -> tuple[int, int]:
    """Derive memory-aware reservoir high/low watermarks.

    The target buffer is expressed in megabytes, then converted to a bounded
    number of ready samples based on the actual tensor size for the chosen
    frontend.
    """
    sample_bytes = max(1, _estimate_sample_bytes(sample_shape, num_classes))
    min_high = max(batch_size * 4, 32)
    target_bytes = int(max(loader_buffer_mb, 1.0) * 1024 * 1024)
    high = max(min_high, min(_MAX_RESERVOIR_SAMPLES, target_bytes // sample_bytes))
    low = max(batch_size * 2, high // 3)
    if low >= high:
        low = max(batch_size, high - batch_size)
    return int(high), int(low)


def _worker_config(
    classes: list[str],
    audio_frontend: str,
    spec_width: int,
    mel_bins: int,
    max_chunks_per_file: int,
    **kwargs: Any,
) -> tuple[dict, tuple[int, ...]]:
    """The per-file processing a loader worker runs, and the shape of one sample.

    Returns:
        ``(worker_cfg, sample_shape)``: the picklable config handed to every
        worker (see ``birdnet_stm32.data.worker``) and one sample's shape.
    """
    audio_frontend = normalize_frontend_name(audio_frontend)
    sr = kwargs.get("sample_rate", 24000)
    cd = kwargs.get("chunk_duration", 3)
    fft_length = kwargs.get("fft_length", 512)
    chunk_len = int(sr * cd)
    mag_scale = kwargs.get("mag_scale", "pwl")
    input_compression = kwargs.get("input_compression", "none")
    max_duration = kwargs.get("max_duration", 60)
    snr_threshold = kwargs.get("snr_threshold", 0.5)
    random_offset = kwargs.get("random_offset", False)
    spec_augment = kwargs.get("spec_augment", False)
    freq_mask_max = kwargs.get("freq_mask_max", 8)
    time_mask_max = kwargs.get("time_mask_max", 25)
    crop_policy = kwargs.get("crop_policy", "energy")
    teacher_cache = kwargs.get("teacher_cache")
    teacher_weight = float(kwargs.get("teacher_weight", 0.0))
    teacher_embeddings = bool(kwargs.get("teacher_embeddings", False))
    if teacher_embeddings and not teacher_cache:
        raise ValueError("teacher_embeddings needs a teacher_cache")
    candidate_chunks_per_file = int(kwargs.get("candidate_chunks_per_file", min(8, max(4, max_chunks_per_file * 2))))
    if random_offset:
        load_duration = max(cd, cd * candidate_chunks_per_file)
        if max_duration:
            load_duration = min(max_duration, load_duration)
    else:
        load_duration = max_duration

    num_classes = len(classes)

    # Determine output shapes
    if audio_frontend == "librosa":
        sample_shape: tuple[int, ...] = (mel_bins, spec_width, 1)
    elif audio_frontend == "hybrid":
        sample_shape = (hybrid_fft_bins(fft_length), spec_width, 1)
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
        "input_compression": input_compression,
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
        "candidate_chunks_per_file": candidate_chunks_per_file,
        "load_duration": load_duration,
        "crop_policy": crop_policy,
        "classes": list(classes),
        "teacher_cache": str(teacher_cache) if teacher_cache else None,
        "teacher_weight": teacher_weight,
        "teacher_embeddings": teacher_embeddings,
        "label_sidecar": kwargs.get("label_sidecar") or {},
    }

    return worker_cfg, sample_shape


def sample_chunks(
    file_paths: list[str],
    classes: list[str],
    audio_frontend: str = "hybrid",
    spec_width: int = 256,
    mel_bins: int = 64,
    num_workers: int = 8,
    max_chunks_per_file: int = 1,
    **kwargs: Any,
) -> list[tuple[np.ndarray, ...]]:
    """Process each file once, exactly as a training loader would, and return the chunks.

    A finite draw with the loader's own worker processing (crop policy, teacher
    targets and embeddings) but no ``tf.data``, reservoir or mixup: for
    one-off statistics such as fitting a training-only head before training.
    A second ``tf.data`` iterator over the infinite training loader would have to
    be torn down mid-stream, and tearing down its worker pool can hang.

    Returns:
        The worker's tuples, in file order; files that fail to load are skipped.
    """
    worker_cfg, _ = _worker_config(classes, audio_frontend, spec_width, mel_bins, max_chunks_per_file, **kwargs)
    if num_workers <= 0:
        _init_worker(worker_cfg)
        results = [_process_file(path) for path in file_paths]
    else:
        pool = _create_worker_pool(num_workers, worker_cfg)
        try:
            results = pool.map(_process_file, list(file_paths), chunksize=16)
        finally:
            _terminate_worker_pool(pool)
    return [item for result in results if result for item in result]


class ClassCappedPasses:
    """File lists for successive passes with at most ``cap`` files per class (the folder).

    A class with more files than ``cap`` contributes a different ``cap`` of them each pass: it
    walks through a shuffled order of all its files, continuing where the last pass stopped and
    reshuffling once every file has been used, so over successive passes every file is drawn
    and none is discarded. Smaller classes contribute all their files every pass. Nothing is
    repeated within a pass. Noise-like folders (all-zero negatives) are not a class and are
    never capped.

    With a ``field`` table (file stem -> (site, recording)) a class's field clips are drawn for
    diversity, not at random: at most ``field_share`` of its quota (more only when it lacks
    focal files to fill the rest), taken round-robin over its sites and, within a site, over
    recordings, with at most ``per_recording`` segments of one recording per pass (consecutive
    windows of a recording are near duplicates). Every recording rotates through its segments
    across passes. Focal files fill the remaining quota as above.
    """

    def __init__(
        self,
        file_paths: list[str],
        cap: int,
        rng: random.Random | None = None,
        field: dict[str, tuple[str, str]] | None = None,
        field_share: float = 0.5,
        per_recording: int = 2,
    ):
        self.cap = int(cap)
        self.rng = rng or random
        self.field_share = float(field_share)
        self.per_recording = max(1, int(per_recording))
        groups: dict[str, list[str]] = {}
        sites: dict[str, dict[str, dict[str, list[str]]]] = {}
        for p in file_paths:
            parts = p.replace("\\", "/").split("/")
            k = parts[-2]
            info = field.get(os.path.splitext(parts[-1])[0]) if field else None
            if info is not None and k.lower() not in NOISE_CLASSES:
                site, rec = info
                sites.setdefault(k, {}).setdefault(site, {}).setdefault(rec, []).append(p)
            else:
                groups.setdefault(k, []).append(p)
        for k in sites:
            groups.setdefault(k, [])
        self.groups = groups  # focal (or all, without a field table) files per class
        self.sites = sites  # field clips per class -> site -> recording
        self.order = {k: [] for k in groups}
        self.rec_order: dict[tuple[str, str, str], list[str]] = {}
        self.rec_start: dict[tuple[str, str], int] = {}

    def _cap(self, k: str) -> int:
        return len(self.groups[k]) if k.lower() in NOISE_CLASSES else self.cap

    def _field_plan(self, k: str) -> int:
        """How many field clips class ``k`` takes per pass (deterministic, for ``len``)."""
        recs = [r for site in self.sites.get(k, {}).values() for r in site.values()]
        if not recs:
            return 0
        eligible = sum(min(self.per_recording, len(r)) for r in recs)
        target = int(self.field_share * self.cap)
        target = max(target, self.cap - len(self.groups[k]))  # field fills what focal cannot
        return min(eligible, target, self.cap)

    def __len__(self) -> int:
        total = 0
        for k, paths in self.groups.items():
            n_field = self._field_plan(k)
            total += n_field + min(len(paths), self._cap(k) - n_field)
        return total

    def _draw_field(self, k: str, n: int) -> list[str]:
        if n <= 0:
            return []
        site_names = list(self.sites[k])
        self.rng.shuffle(site_names)
        cursors = []
        for site in site_names:
            recs = sorted(self.sites[k][site])
            start = self.rec_start.get((k, site), 0) % len(recs)
            self.rec_start[(k, site)] = start + 1  # the next pass starts at another recording
            cursors.append([recs[(start + i) % len(recs)] for i in range(len(recs))])
        used: dict[tuple[str, str], int] = {}
        take: list[str] = []
        while len(take) < n:
            progressed = False
            for site, recs in zip(site_names, cursors, strict=True):
                while recs:
                    rec = recs[0]
                    key = (site, rec)
                    clips = self.sites[k][site][rec]
                    if used.get(key, 0) >= min(self.per_recording, len(clips)):
                        recs.pop(0)
                        continue
                    order_key = (k, site, rec)
                    if not self.rec_order.get(order_key):
                        self.rec_order[order_key] = list(clips)
                        self.rng.shuffle(self.rec_order[order_key])
                    take.append(self.rec_order[order_key].pop())
                    used[key] = used.get(key, 0) + 1
                    recs.append(recs.pop(0))  # next recording of this site on the next round
                    progressed = True
                    break
                if len(take) >= n:
                    break
            if not progressed:
                break
        return take

    def next_pass(self) -> list[str]:
        out: list[str] = []
        for k, paths in self.groups.items():
            n_field = self._field_plan(k)
            out.extend(self._draw_field(k, n_field))
            quota = self._cap(k) - n_field
            if len(paths) <= quota:
                out.extend(paths)
                continue
            take: list[str] = []
            while len(take) < quota:
                if not self.order[k]:
                    self.order[k] = list(paths)
                    self.rng.shuffle(self.order[k])
                n = quota - len(take)
                take.extend(self.order[k][:n])
                self.order[k] = self.order[k][n:]
            out.extend(take)
        self.rng.shuffle(out)
        return out


def load_dataset(
    file_paths: list[str],
    classes: list[str],
    audio_frontend: str = "hybrid",
    batch_size: int = 32,
    spec_width: int = 256,
    mel_bins: int = 64,
    num_workers: int = 8,
    max_chunks_per_file: int = 1,
    **kwargs: Any,
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
        audio_frontend: 'librosa' | 'hybrid' | 'raw'.
        batch_size: Batch size.
        spec_width: Target spectrogram width.
        mel_bins: Number of mel bins.
        num_workers: Number of worker processes (0 = single-process fallback).
        max_chunks_per_file: Max salient chunks to extract per file open.
        **kwargs: Forwarded to loading logic (sample_rate, chunk_duration, etc.).

    Returns:
        Infinite tf.data.Dataset of (inputs, labels) with prefetching. With
        ``teacher_embeddings=True`` (needs a ``teacher_cache`` holding
        ``emb.npy``) the labels are ``(labels, teacher_embedding, valid)``:
        float32 ``[B, D]`` embeddings and a float32 ``[B]`` flag that is 0 for
        chunks with no usable teacher window and for chunks mixup mixed, whose
        audio the cached embedding no longer describes.
    """
    worker_cfg, sample_shape = _worker_config(
        classes, audio_frontend, spec_width, mel_bins, max_chunks_per_file, **kwargs
    )
    cd = worker_cfg["cd"]
    num_classes = worker_cfg["num_classes"]
    teacher_embeddings = worker_cfg["teacher_embeddings"]
    embedding_dim = 0
    if teacher_embeddings:
        embedding_dim = int(np.load(f"{worker_cfg['teacher_cache']}/emb.npy", mmap_mode="r").shape[1])
    mixup_alpha = kwargs.get("mixup_alpha", 0.2)
    mixup_probability = kwargs.get("mixup_probability", 0.25)
    # Keep prefetch bounded to avoid RAM spikes with large raw batches.
    prefetch_batches = int(kwargs.get("prefetch_batches", 2))
    loader_buffer_mb = float(kwargs.get("loader_buffer_mb", _DEFAULT_BUFFER_MB))
    # Bound in-flight multiprocessing tasks so result queues cannot grow
    # unbounded during long epochs.
    max_inflight_files = int(kwargs.get("max_inflight_files", max(256, num_workers * 64)))
    loader_control = kwargs.get("loader_control")
    file_task_timeout_s = float(kwargs.get("file_task_timeout_s", max(120.0, cd * 10.0)))

    use_mp = num_workers > 0

    reservoir_high, reservoir_low = _compute_reservoir_limits(
        sample_shape=sample_shape,
        num_classes=num_classes,
        batch_size=batch_size,
        loader_buffer_mb=loader_buffer_mb,
    )

    # Per-epoch class cap (train --class_cap_per_epoch): a balanced draw without repetition.
    class_cap = int(kwargs.get("class_cap_per_epoch", 0) or 0)
    capped = (
        ClassCappedPasses(
            file_paths,
            class_cap,
            field=kwargs.get("field_table"),
            field_share=float(kwargs.get("field_share", 0.5)),
            per_recording=int(kwargs.get("field_per_recording", 2)),
        )
        if class_cap > 0
        else None
    )

    def _generator():
        """Infinite generator with reservoir for multi-chunk file reuse."""
        pool = None
        if use_mp:
            pool = _create_worker_pool(num_workers, worker_cfg)
        else:
            _init_worker(worker_cfg)

        try:
            while True:
                if capped is not None:
                    shuffled = capped.next_pass()
                else:
                    shuffled = list(file_paths)
                    random.shuffle(shuffled)

                reservoir: list[tuple[np.ndarray, np.ndarray]] = []

                if pool is None:
                    for path in shuffled:
                        result = _process_file(path)
                        if result is not None:
                            reservoir.extend(result)
                        elif isinstance(loader_control, dict):
                            loader_control["last_skipped_file"] = path
                        if len(reservoir) >= reservoir_high:
                            random.shuffle(reservoir)
                            while len(reservoir) > reservoir_low:
                                yield reservoir.pop()
                else:
                    pending: list[dict[str, object]] = []
                    next_index = 0

                    while next_index < len(shuffled) or pending:
                        current_inflight = max_inflight_files
                        if isinstance(loader_control, dict):
                            current_inflight = int(loader_control.get("max_inflight_files", max_inflight_files))
                        inflight_cap = max(
                            32,
                            max(batch_size * 2, num_workers * 4, (reservoir_high // max(1, max_chunks_per_file)) * 2),
                        )
                        current_inflight = max(32, min(current_inflight, inflight_cap))

                        while next_index < len(shuffled) and len(pending) < current_inflight:
                            path = shuffled[next_index]
                            next_index += 1
                            pending.append(
                                {
                                    "path": path,
                                    "started_at": time.monotonic(),
                                    "result": pool.apply_async(_process_file, (path,)),
                                }
                            )

                        made_progress = False
                        recycle_pool = False
                        timed_out_path = None
                        now = time.monotonic()

                        for idx in range(len(pending) - 1, -1, -1):
                            job = pending[idx]
                            async_result = job["result"]
                            if async_result.ready():
                                pending.pop(idx)
                                try:
                                    result = async_result.get()
                                except Exception:
                                    result = None
                                if result is not None:
                                    reservoir.extend(result)
                                elif isinstance(loader_control, dict):
                                    loader_control["last_skipped_file"] = str(job["path"])
                                made_progress = True
                                continue

                            if file_task_timeout_s > 0 and (now - float(job["started_at"])) > file_task_timeout_s:
                                timed_out_path = str(job["path"])
                                recycle_pool = True
                                break

                        if recycle_pool:
                            if isinstance(loader_control, dict):
                                loader_control["last_loader_timeout"] = {
                                    "path": timed_out_path,
                                    "timeout_s": float(file_task_timeout_s),
                                    "pending_jobs": int(len(pending)),
                                }
                            _terminate_worker_pool(pool)
                            pool = _create_worker_pool(num_workers, worker_cfg)
                            pending.clear()
                            continue

                        if len(reservoir) >= reservoir_high:
                            random.shuffle(reservoir)
                            while len(reservoir) > reservoir_low:
                                yield reservoir.pop()
                                made_progress = True
                        elif reservoir and not made_progress:
                            yield reservoir.pop()
                            made_progress = True

                        if not made_progress and pending:
                            time.sleep(_POOL_POLL_INTERVAL_S)

                # Drain remaining samples at end of epoch
                if reservoir:
                    random.shuffle(reservoir)
                    while reservoir:
                        yield reservoir.pop()
        except GeneratorExit:
            pass  # tf.data tearing down the generator — normal shutdown
        finally:
            _terminate_worker_pool(pool)

    def _batched():
        """Stack samples into whole batches before they cross into TensorFlow.

        One conversion per batch instead of one per sample: the per-sample
        handoff held the GIL often enough that loading and the training step
        took turns instead of overlapping. The stream is unchanged; a partial
        batch carries over to the next pass, as ``batch(drop_remainder=True)``
        on the infinite generator did.
        """
        batch: list[tuple[np.ndarray, ...]] = []
        for item in _generator():
            batch.append(item)
            if len(batch) == batch_size:
                columns = [np.stack(column) for column in zip(*batch, strict=True)]
                if teacher_embeddings:
                    x, y, emb, valid = columns
                    yield x, (y, emb.astype(np.float32), valid)
                else:
                    yield columns[0], columns[1]
                batch = []

    label_sig = tf.TensorSpec(shape=(batch_size, num_classes), dtype=tf.float32)
    if teacher_embeddings:
        label_sig = (
            label_sig,
            tf.TensorSpec(shape=(batch_size, embedding_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size,), dtype=tf.float32),
        )
    output_sig = (tf.TensorSpec(shape=(batch_size, *sample_shape), dtype=tf.float32), label_sig)

    dataset = tf.data.Dataset.from_generator(_batched, output_signature=output_sig)

    # Mixup on batches
    if mixup_alpha > 0 and mixup_probability > 0:

        def _apply_batch_mixup(samples, labels):
            if teacher_embeddings:
                labels, emb, valid = labels
            mixed_s, mixed_l, mixed = mixup_batch(samples, labels, alpha=mixup_alpha, probability=mixup_probability)
            if not teacher_embeddings:
                return mixed_s, mixed_l
            # A mixed chunk no longer sounds like the window its embedding came from.
            return mixed_s, (mixed_l, emb, valid * (1.0 - mixed))

        dataset = dataset.map(_apply_batch_mixup, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.prefetch(max(1, prefetch_batches))
    return dataset
