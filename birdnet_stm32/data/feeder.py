"""Training batches assembled in a separate process and handed over through shared memory.

The training process used to do all of this in its own Python thread: send files to the
worker pool, receive and unpickle every chunk, buffer chunks in the shuffling reservoir and
stack batches. That work held the GIL often enough to keep the training step waiting
(measured: 50 ms a step against 19 ms for the step alone). Here a feeder process does it:
it owns the worker pool, draws each pass's files, keeps the reservoir and writes whole
batches into a ring of shared-memory slots. The training process copies a finished slot
into TensorFlow and hands it back.

Kept free of TensorFlow, like the worker module: the feeder starts from the forkserver.
"""

from __future__ import annotations

import atexit
import contextlib
import multiprocessing as mp
import os
import random
import signal
import sys
import time
import traceback
from collections.abc import Callable, Iterator
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from birdnet_stm32.data.species import NOISE_CLASSES
from birdnet_stm32.data.worker import _init_worker, _process_file

_POOL_POLL_INTERVAL_S = 0.05

# Batches the feeder may hold ready ahead of the training step (a raw batch of 32 is ~8 MB).
_FEEDER_SLOTS = 8


def _pool_context():
    """Start workers from a clean forkserver rather than by forking the trainer.

    Forking the training process copies a snapshot of dozens of TensorFlow and
    loader threads. A child forked while one of them held a lock (logging,
    malloc, BLAS) hangs on first use, and with ``maxtasksperchild`` respawning
    every worker every 100 files, some runs lost most of their pool: measured,
    the same seeded run trained at 55 or at 390 ms/step with the GPU at 6%.
    The forkserver is single-threaded and preloads only the TF-free worker
    and feeder modules, so a replacement worker (after a timeout) cannot inherit a held lock.
    """
    ctx = mp.get_context("forkserver")
    ctx.set_forkserver_preload(["birdnet_stm32.data.worker", "birdnet_stm32.data.feeder"])
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


def sample_stream(
    file_paths: list[str],
    worker_cfg: dict,
    num_workers: int,
    *,
    batch_size: int,
    reservoir_high: int,
    reservoir_low: int,
    max_inflight_files: int,
    file_task_timeout_s: float,
    class_cap: int = 0,
    field: dict[str, tuple[str, str]] | None = None,
    field_share: float = 0.5,
    per_recording: int = 2,
    inflight: Callable[[], int] | None = None,
    report: Callable[[str, Any], None] | None = None,
) -> Iterator[tuple[np.ndarray, ...]]:
    """Infinite stream of training chunks, shuffled through a reservoir for multi-chunk file reuse.

    Each pass covers ``file_paths`` once, or a class-capped draw of them (``class_cap``, see
    ``ClassCappedPasses``). ``inflight`` returns the current cap on files in flight (tuned
    online by the trainer); ``report`` receives loader events (skipped files, timeouts).
    """
    max_chunks_per_file = int(worker_cfg["max_chunks_per_file"])
    capped = (
        ClassCappedPasses(file_paths, class_cap, field=field, field_share=field_share, per_recording=per_recording)
        if class_cap > 0
        else None
    )
    report = report or (lambda key, value: None)
    pool = None
    if num_workers > 0:
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

            reservoir: list[tuple[np.ndarray, ...]] = []

            if pool is None:
                for path in shuffled:
                    result = _process_file(path)
                    if result is not None:
                        reservoir.extend(result)
                    else:
                        report("last_skipped_file", path)
                    if len(reservoir) >= reservoir_high:
                        random.shuffle(reservoir)
                        while len(reservoir) > reservoir_low:
                            yield reservoir.pop()
            else:
                pending: list[dict[str, object]] = []
                next_index = 0

                while next_index < len(shuffled) or pending:
                    current_inflight = inflight() if inflight is not None else max_inflight_files
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
                            else:
                                report("last_skipped_file", str(job["path"]))
                            made_progress = True
                            continue

                        if file_task_timeout_s > 0 and (now - float(job["started_at"])) > file_task_timeout_s:
                            timed_out_path = str(job["path"])
                            recycle_pool = True
                            break

                    if recycle_pool:
                        report(
                            "last_loader_timeout",
                            {
                                "path": timed_out_path,
                                "timeout_s": float(file_task_timeout_s),
                                "pending_jobs": int(len(pending)),
                            },
                        )
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
    finally:
        _terminate_worker_pool(pool)


def batch_layout(
    batch_size: int, sample_shape: tuple[int, ...], num_classes: int, embedding_dim: int = 0
) -> list[tuple[tuple[int, ...], str]]:
    """Shape and dtype of each column of a batch: samples, labels and, with teacher embeddings,
    the float16 embeddings and their valid flags (the worker's tuple, stacked)."""
    layout = [((batch_size, *sample_shape), "float32"), ((batch_size, num_classes), "float32")]
    if embedding_dim:
        layout += [((batch_size, embedding_dim), "float16"), ((batch_size,), "float32")]
    return layout


def fill_batch(stream: Iterator[tuple[np.ndarray, ...]], columns: list[np.ndarray]) -> None:
    """Write the next ``len(columns[0])`` chunks of ``stream`` into ``columns`` row by row."""
    for row in range(len(columns[0])):
        for column, value in zip(columns, next(stream), strict=True):
            column[row] = value


def _slot_columns(buf, n_slots: int, layout) -> list[np.ndarray]:
    """``[n_slots, *shape]`` arrays over ``buf``, one per column, each 64-byte aligned."""
    columns, offset = [], 0
    for shape, dtype in layout:
        nbytes = n_slots * int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        columns.append(np.ndarray((n_slots, *shape), dtype=dtype, buffer=buf, offset=offset))
        offset += -(-nbytes // 64) * 64
    return columns


def _ring_bytes(n_slots: int, layout) -> int:
    return sum(-(-(n_slots * int(np.prod(s, dtype=np.int64)) * np.dtype(d).itemsize) // 64) * 64 for s, d in layout)


def _feeder_main(stream_kwargs, seed, shm_name, n_slots, layout, free_conn, ready_conn, inflight_value) -> None:
    """Feeder process: fill each slot the trainer frees with the next batch and announce it.

    Exits when the trainer closes its end of ``free_conn`` (or dies). Any error is sent to the
    trainer, which raises it.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C is the trainer's; it stops the feeder
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # run the finally blocks: stop the pool
    random.seed(seed)
    np.random.seed(seed % 2**32)
    shm = shared_memory.SharedMemory(name=shm_name)
    columns = _slot_columns(shm.buf, n_slots, layout)
    stream = sample_stream(
        **stream_kwargs,
        inflight=lambda: int(inflight_value.value),
        report=lambda key, value: ready_conn.send(("control", key, value)),
    )
    try:
        while True:
            try:
                slot = free_conn.recv()
            except EOFError:
                return
            fill_batch(stream, [column[slot] for column in columns])
            ready_conn.send(("batch", slot))
    except (BrokenPipeError, ConnectionResetError):
        pass  # the trainer is gone
    except SystemExit:
        raise
    except BaseException:
        with contextlib.suppress(OSError):
            ready_conn.send(("error", traceback.format_exc()))
        raise
    finally:
        stream.close()
        del columns
        shm.close()


class BatchFeeder:
    """Training-process end of the feeder: start it, receive its batches, stop it.

    The ring of slots lives in shared memory created here. A slot circulates as an index:
    free (trainer to feeder), filled (feeder to trainer), copied out and freed again.
    """

    def __init__(
        self,
        stream_kwargs: dict,
        layout: list[tuple[tuple[int, ...], str]],
        seed: int,
        n_slots: int = _FEEDER_SLOTS,
    ):
        ctx = _pool_context()
        self._closed = False
        self.shm = shared_memory.SharedMemory(create=True, size=_ring_bytes(n_slots, layout))
        self.columns = _slot_columns(self.shm.buf, n_slots, layout)
        free_r, self.free = ctx.Pipe(duplex=False)
        self.ready, ready_w = ctx.Pipe(duplex=False)
        self.inflight = ctx.Value("i", int(stream_kwargs["max_inflight_files"]), lock=False)
        # Not a daemon: a daemon may not start the worker pool. close() (also run at exit,
        # before multiprocessing joins its children) stops it.
        self.proc = ctx.Process(
            target=_feeder_main,
            args=(stream_kwargs, seed, self.shm.name, n_slots, layout, free_r, ready_w, self.inflight),
            name="loader-feeder",
        )
        try:
            self.proc.start()
        except BaseException:
            self.close()
            raise
        free_r.close()  # only the feeder's copies remain: EOF reaches each side when the other exits
        ready_w.close()
        atexit.register(self.close)
        for slot in range(n_slots):
            self.free.send(slot)

    def next(self, control: dict | None = None) -> list[np.ndarray]:
        """The next batch's columns (copies); loader events go into ``control``."""
        if isinstance(control, dict):
            self.inflight.value = int(control.get("max_inflight_files", self.inflight.value))
        while True:
            try:
                message = self.ready.recv()
            except EOFError:
                raise RuntimeError(f"Loader feeder process exited (exit code {self.proc.exitcode})") from None
            kind = message[0]
            if kind == "batch":
                slot = message[1]
                batch = [column[slot].copy() for column in self.columns]
                self.free.send(slot)
                return batch
            if kind == "control":
                if isinstance(control, dict):
                    control[message[1]] = message[2]
            elif kind == "error":
                raise RuntimeError(f"Loader feeder process failed:\n{message[1]}")

    def close(self) -> None:
        """Stop the feeder and its pool and release the shared memory. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.close)
        for conn in (getattr(self, "free", None), getattr(self, "ready", None)):
            if conn is not None:
                conn.close()
        proc = getattr(self, "proc", None)
        if proc is not None and proc.pid is not None:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()  # SIGTERM: the feeder unwinds and stops its pool
                proc.join(timeout=15)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)
        self.columns = []
        self.shm.close()
        with contextlib.suppress(FileNotFoundError):
            self.shm.unlink()
