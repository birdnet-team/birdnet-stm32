"""Tests for the feeder process that assembles training batches in shared memory."""

import os

import numpy as np
import pytest
import soundfile as sf

from birdnet_stm32.data.feeder import BatchFeeder, batch_layout
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
    return BatchFeeder(stream_kwargs, batch_layout(batch_size, sample_shape, len(classes)), seed=0, n_slots=3)


class TestBatchFeeder:
    def test_each_batch_is_one_pass_over_the_files(self, tmp_path):
        paths, classes = _one_file_per_class(tmp_path)
        feeder = _feeder(paths, classes, batch_size=len(paths))
        try:
            for _ in range(5):  # more batches than slots: slots are recycled
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
        feeder = _feeder(paths, classes, batch_size=4, reservoir_high=None)  # breaks the stream
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
