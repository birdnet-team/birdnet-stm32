"""Tests for the feeder process that assembles training batches in shared memory."""

import os

import numpy as np
import pytest
import soundfile as sf

from birdnet_stm32.data.feeder import BatchFeeder, chunk_layout
from birdnet_stm32.data.generator import _worker_config, load_dataset

SR = 8000
CD = 1.0


def _one_file_per_class(tmp_path, n_classes=6):
    classes = [f"class_{i}" for i in range(n_classes)]
    rng = np.random.default_rng(0)
    paths = []
    for cls in classes:
        (tmp_path / cls).mkdir()
        path = tmp_path / cls / "rec.wav"
        sf.write(str(path), (rng.standard_normal(int(SR * CD)) * 0.3).astype(np.float32), SR)
        paths.append(str(path))
    return paths, classes


def _feeder(paths, classes, batch_size, **overrides):
    worker_cfg, sample_shape = _worker_config(classes, "raw", 256, 64, 1, sample_rate=SR, chunk_duration=CD)
    stream_kwargs = dict(
        file_paths=paths,
        worker_cfg=worker_cfg,
        num_workers=2,
        batch_size=batch_size,
        reservoir_high=64,
        reservoir_low=32,
        max_inflight_files=64,
        file_task_timeout_s=0.0,
    )
    stream_kwargs.update(overrides)
    return BatchFeeder(stream_kwargs, chunk_layout(sample_shape, len(classes)), seed=0, batches_ahead=2)


class TestBatchFeeder:
    def test_each_batch_is_one_pass_over_the_files(self, tmp_path):
        paths, classes = _one_file_per_class(tmp_path)
        feeder = _feeder(paths, classes, batch_size=len(paths))
        try:
            for _ in range(5):  # more batches than may be out at once: cells are recycled
                x, y = feeder.next()
                assert x.shape == (len(paths), int(SR * CD), 1)
                np.testing.assert_array_equal(y.sum(axis=0), np.ones(len(classes)))
                np.testing.assert_array_equal(y.sum(axis=1), np.ones(len(paths)))
        finally:
            feeder.close()

    def test_close_stops_the_process_and_frees_shared_memory(self, tmp_path):
        paths, classes = _one_file_per_class(tmp_path)
        feeder = _feeder(paths, classes, batch_size=4)
        feeder.next()
        name = feeder.shm.name
        assert os.path.exists(f"/dev/shm/{name}")
        feeder.close()
        feeder.close()  # idempotent
        assert not feeder.proc.is_alive()
        assert not os.path.exists(f"/dev/shm/{name}")

    def test_feeder_errors_reach_the_trainer(self, tmp_path):
        paths, classes = _one_file_per_class(tmp_path)
        feeder = _feeder(paths, classes, batch_size=4, file_task_timeout_s="broken")  # breaks the stream
        try:
            with pytest.raises(RuntimeError, match="feeder process failed"):
                feeder.next()
        finally:
            feeder.close()

    def test_a_killed_feeder_raises_instead_of_hanging(self, tmp_path):
        paths, classes = _one_file_per_class(tmp_path)
        feeder = _feeder(paths, classes, batch_size=4)
        try:
            feeder.next()
            feeder.proc.kill()
            feeder.proc.join(5)
            with pytest.raises(RuntimeError, match="exited"):
                for _ in range(10):  # slots already filled are still delivered
                    feeder.next()
        finally:
            feeder.close()

    def test_loader_events_reach_the_control_dict(self, tmp_path):
        paths, classes = _one_file_per_class(tmp_path)
        bad = tmp_path / "class_0" / "broken.wav"
        bad.write_bytes(b"not audio")
        feeder = _feeder([str(bad), *paths], classes, batch_size=len(paths))
        control = {"max_inflight_files": 40}
        try:
            for _ in range(3):
                feeder.next(control)
            assert control["last_skipped_file"] == str(bad)
            assert feeder.inflight.value == 40
        finally:
            feeder.close()


def test_load_dataset_matches_in_process_batches(tmp_path):
    """The feeder path delivers the same batch structure and label contract as the serial one."""
    paths, classes = _one_file_per_class(tmp_path)
    common = dict(
        audio_frontend="raw",
        batch_size=len(paths),
        sample_rate=SR,
        chunk_duration=CD,
        mixup_alpha=0.0,
        mixup_probability=0.0,
    )
    for workers in (0, 2):
        it = iter(load_dataset(paths, classes, num_workers=workers, **common))
        for _ in range(3):
            x, y = next(it)
            assert x.shape == (len(paths), int(SR * CD), 1)
            np.testing.assert_array_equal(y.numpy().sum(axis=0), np.ones(len(classes)))
        del it


_HANGUP_CHILD = """
import os, sys, time
import numpy as np, soundfile as sf
from birdnet_stm32.data.feeder import BatchFeeder, chunk_layout
from birdnet_stm32.data.generator import _worker_config

if __name__ == "__main__":
    root = sys.argv[1] + "/audio/a"
    os.makedirs(root)
    paths = []
    for i in range(4):
        path = f"{root}/c{i}.wav"
        sf.write(path, np.random.default_rng(i).standard_normal(8000).astype(np.float32) * 0.3, 8000)
        paths.append(path)
    classes = ["a"]
    cfg, shape = _worker_config(classes, "raw", 256, 64, 1, sample_rate=8000, chunk_duration=1.0)
    kw = dict(file_paths=paths, worker_cfg=cfg, num_workers=2, batch_size=2, reservoir_high=8, reservoir_low=4,
              max_inflight_files=32, file_task_timeout_s=0.0)
    feeder = BatchFeeder(kw, chunk_layout(shape, 1), seed=0)
    feeder.next()
    print(feeder.shm.name, flush=True)
    time.sleep(60)
"""


def test_a_hang_up_does_not_leak_the_arena(tmp_path):
    """A closing screen hangs up trainer, feeder and resource tracker at once; the arena must still go."""
    import select
    import signal
    import subprocess
    import sys
    import time

    script = tmp_path / "child.py"
    script.write_text(_HANGUP_CHILD)
    child = subprocess.Popen(
        [sys.executable, str(script), str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    try:
        ready, _, _ = select.select([child.stdout], [], [], 60)
        assert ready, "the child never produced a batch"
        name = child.stdout.readline().strip()
        assert name and os.path.exists(f"/dev/shm/{name}")
        os.killpg(child.pid, signal.SIGHUP)
        deadline = time.monotonic() + 30
        while os.path.exists(f"/dev/shm/{name}") and time.monotonic() < deadline:
            time.sleep(0.2)
        assert not os.path.exists(f"/dev/shm/{name}")
    finally:
        child.kill()
        child.wait()
