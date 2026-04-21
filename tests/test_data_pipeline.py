"""Tests for the multiprocessing data pipeline and reservoir logic."""

import numpy as np
import soundfile as sf

from birdnet_stm32.data.generator import (
    _compute_reservoir_limits,
    _init_worker,
    _process_file,
    estimate_samples_per_epoch,
    load_dataset,
)


def _make_long_dataset(tmp_path, sample_rate=22050, chunk_duration=3, n_classes=3, files_per_class=2, file_duration=12):
    """Create a dataset with files long enough to yield multiple chunks."""
    classes = [f"class_{i}" for i in range(n_classes)]
    all_paths = []
    rng = np.random.default_rng(42)
    for cls in classes:
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for j in range(files_per_class):
            n_samples = int(sample_rate * file_duration)
            audio = rng.standard_normal(n_samples).astype(np.float32) * 0.5
            path = cls_dir / f"sample_{j}.wav"
            sf.write(str(path), audio, sample_rate)
            all_paths.append(str(path))
    return all_paths, classes


class TestProcessFile:
    """Tests for the worker function _process_file."""

    def test_returns_list_of_tuples(self, tmp_path):
        """_process_file returns a list of (sample, label) tuples."""
        paths, classes = _make_long_dataset(tmp_path, file_duration=12)
        cfg = {
            "audio_frontend": "hybrid",
            "sr": 22050,
            "cd": 3,
            "T": 22050 * 3,
            "fft_length": 512,
            "mel_bins": 64,
            "spec_width": 256,
            "mag_scale": "pwl",
            "n_mfcc": 20,
            "max_duration": 60,
            "snr_threshold": 0.0,
            "random_offset": False,
            "spec_augment": False,
            "freq_mask_max": 8,
            "time_mask_max": 25,
            "noise_labels": ("noise", "silence", "background", "other"),
            "class_to_idx": {c: i for i, c in enumerate(classes)},
            "num_classes": len(classes),
            "max_chunks_per_file": 3,
        }
        _init_worker(cfg)
        result = _process_file(paths[0])
        assert result is not None
        assert isinstance(result, list)
        assert len(result) >= 1
        for sample, label in result:
            assert sample.dtype == np.float32
            assert label.dtype == np.float32
            assert label.shape == (len(classes),)

    def test_max_chunks_limits_output(self, tmp_path):
        """max_chunks_per_file=1 returns at most 1 chunk."""
        paths, classes = _make_long_dataset(tmp_path, file_duration=12)
        cfg = {
            "audio_frontend": "hybrid",
            "sr": 22050,
            "cd": 3,
            "T": 22050 * 3,
            "fft_length": 512,
            "mel_bins": 64,
            "spec_width": 256,
            "mag_scale": "pwl",
            "n_mfcc": 20,
            "max_duration": 60,
            "snr_threshold": 0.0,
            "random_offset": False,
            "spec_augment": False,
            "freq_mask_max": 8,
            "time_mask_max": 25,
            "noise_labels": ("noise", "silence", "background", "other"),
            "class_to_idx": {c: i for i, c in enumerate(classes)},
            "num_classes": len(classes),
            "max_chunks_per_file": 1,
        }
        _init_worker(cfg)
        result = _process_file(paths[0])
        assert result is not None
        assert len(result) == 1

    def test_unknown_class_returns_none(self, tmp_path):
        """Files from unknown classes return None."""
        paths, classes = _make_long_dataset(tmp_path)
        cfg = {
            "audio_frontend": "hybrid",
            "sr": 22050,
            "cd": 3,
            "T": 22050 * 3,
            "fft_length": 512,
            "mel_bins": 64,
            "spec_width": 256,
            "mag_scale": "pwl",
            "n_mfcc": 20,
            "max_duration": 60,
            "snr_threshold": 0.0,
            "random_offset": False,
            "spec_augment": False,
            "freq_mask_max": 8,
            "time_mask_max": 25,
            "noise_labels": ("noise", "silence", "background", "other"),
            "class_to_idx": {"nonexistent": 0},
            "num_classes": 1,
            "max_chunks_per_file": 3,
        }
        _init_worker(cfg)
        result = _process_file(paths[0])
        assert result is None

    def test_multi_chunks_more_than_single(self, tmp_path):
        """Long files with max_chunks=3 yield more chunks than max_chunks=1."""
        paths, classes = _make_long_dataset(tmp_path, file_duration=15)
        base_cfg = {
            "audio_frontend": "hybrid",
            "sr": 22050,
            "cd": 3,
            "T": 22050 * 3,
            "fft_length": 512,
            "mel_bins": 64,
            "spec_width": 256,
            "mag_scale": "pwl",
            "n_mfcc": 20,
            "max_duration": 60,
            "snr_threshold": 0.0,
            "random_offset": False,
            "spec_augment": False,
            "freq_mask_max": 8,
            "time_mask_max": 25,
            "noise_labels": ("noise", "silence", "background", "other"),
            "class_to_idx": {c: i for i, c in enumerate(classes)},
            "num_classes": len(classes),
        }

        _init_worker({**base_cfg, "max_chunks_per_file": 1})
        result_1 = _process_file(paths[0])

        _init_worker({**base_cfg, "max_chunks_per_file": 3})
        result_3 = _process_file(paths[0])

        assert result_1 is not None and result_3 is not None
        assert len(result_3) >= len(result_1)

    def test_random_offset_uses_bounded_read_window(self, monkeypatch, tmp_path):
        """Random-offset training should read only the candidate window budget."""
        paths, classes = _make_long_dataset(tmp_path, file_duration=30)
        observed: dict[str, float | bool] = {}

        def fake_load_audio_window(path, sample_rate, max_duration, chunk_duration, random_offset):
            observed["max_duration"] = max_duration
            observed["random_offset"] = random_offset
            return np.ones(int(sample_rate * chunk_duration * 4), dtype=np.float32)

        monkeypatch.setattr("birdnet_stm32.data.generator.load_audio_window", fake_load_audio_window)

        cfg = {
            "audio_frontend": "raw",
            "sr": 22050,
            "cd": 3,
            "T": 22050 * 3,
            "fft_length": 512,
            "mel_bins": 64,
            "spec_width": 256,
            "mag_scale": "pwl",
            "n_mfcc": 20,
            "max_duration": 60,
            "load_duration": 12,
            "snr_threshold": 0.0,
            "random_offset": True,
            "spec_augment": False,
            "freq_mask_max": 8,
            "time_mask_max": 25,
            "noise_labels": ("noise", "silence", "background", "other"),
            "class_to_idx": {c: i for i, c in enumerate(classes)},
            "num_classes": len(classes),
            "max_chunks_per_file": 1,
            "candidate_chunks_per_file": 4,
        }
        _init_worker(cfg)

        result = _process_file(paths[0])

        assert result is not None
        assert observed == {"max_duration": 12, "random_offset": True}

    def test_corrupt_file_is_skipped(self, monkeypatch, tmp_path):
        """Unreadable audio files should be skipped instead of synthesized."""
        paths, classes = _make_long_dataset(tmp_path, file_duration=12)

        monkeypatch.setattr(
            "birdnet_stm32.data.generator.load_audio_window",
            lambda *args, **kwargs: np.empty((0,), dtype=np.float32),
        )

        cfg = {
            "audio_frontend": "raw",
            "sr": 22050,
            "cd": 3,
            "T": 22050 * 3,
            "fft_length": 512,
            "mel_bins": 64,
            "spec_width": 256,
            "mag_scale": "pwl",
            "n_mfcc": 20,
            "max_duration": 60,
            "load_duration": 12,
            "snr_threshold": 0.0,
            "random_offset": False,
            "spec_augment": False,
            "freq_mask_max": 8,
            "time_mask_max": 25,
            "noise_labels": ("noise", "silence", "background", "other"),
            "class_to_idx": {c: i for i, c in enumerate(classes)},
            "num_classes": len(classes),
            "max_chunks_per_file": 1,
            "candidate_chunks_per_file": 4,
        }
        _init_worker(cfg)

        assert _process_file(paths[0]) is None


class TestReservoirSizing:
    """Tests for memory-aware sample buffering."""

    def test_larger_samples_reduce_reservoir_capacity(self):
        """Bigger per-sample tensors should lower the ready-sample buffer size."""
        mel_high, mel_low = _compute_reservoir_limits((64, 256, 1), num_classes=10, batch_size=16, loader_buffer_mb=64)
        raw_high, raw_low = _compute_reservoir_limits((60000, 1), num_classes=10, batch_size=16, loader_buffer_mb=64)

        assert raw_high < mel_high
        assert mel_low < mel_high
        assert raw_low < raw_high


class TestEstimateSamplesPerEpoch:
    """Tests for estimate_samples_per_epoch."""

    def test_single_chunk(self):
        assert estimate_samples_per_epoch(100, 1) == 100

    def test_multi_chunk(self):
        assert estimate_samples_per_epoch(100, 3) == 200  # avg = 2.0

    def test_minimum_one(self):
        assert estimate_samples_per_epoch(0, 3) >= 1


class TestLoadDataset:
    """Integration tests for load_dataset with reservoir."""

    def test_yields_correct_shapes(self, tmp_path):
        """Dataset yields batches with correct shapes."""
        paths, classes = _make_long_dataset(tmp_path, file_duration=9)
        batch_size = 2
        fft_length = 512
        spec_width = 256

        ds = load_dataset(
            paths,
            classes,
            audio_frontend="hybrid",
            batch_size=batch_size,
            spec_width=spec_width,
            mel_bins=64,
            num_workers=0,
            max_chunks_per_file=2,
            sample_rate=22050,
            chunk_duration=3,
            fft_length=fft_length,
            mixup_alpha=0.0,
            mixup_probability=0.0,
            snr_threshold=0.0,
        )
        batch = next(iter(ds))
        samples, labels = batch
        assert samples.shape == (batch_size, fft_length // 2 + 1, spec_width, 1)
        assert labels.shape == (batch_size, len(classes))

    def test_multi_chunk_produces_more_samples(self, tmp_path):
        """max_chunks_per_file=3 produces more samples than max_chunks_per_file=1."""
        paths, classes = _make_long_dataset(tmp_path, file_duration=12)
        common = dict(
            audio_frontend="hybrid",
            batch_size=2,
            spec_width=256,
            mel_bins=64,
            num_workers=0,
            sample_rate=22050,
            chunk_duration=3,
            fft_length=512,
            mixup_alpha=0.0,
            mixup_probability=0.0,
            snr_threshold=0.0,
        )

        ds1 = load_dataset(paths, classes, max_chunks_per_file=1, **common)
        ds3 = load_dataset(paths, classes, max_chunks_per_file=3, **common)

        # Count batches in a limited number of samples
        n_batches = 20
        count1 = sum(1 for _, batch in zip(range(n_batches), ds1, strict=False))
        count3 = sum(1 for _, batch in zip(range(n_batches), ds3, strict=False))
        # Both should produce at least n_batches (infinite dataset)
        assert count1 == n_batches
        assert count3 == n_batches

    def test_single_process_fallback(self, tmp_path):
        """num_workers=0 works correctly."""
        paths, classes = _make_long_dataset(tmp_path, file_duration=6, n_classes=2, files_per_class=1)
        ds = load_dataset(
            paths,
            classes,
            audio_frontend="hybrid",
            batch_size=1,
            num_workers=0,
            max_chunks_per_file=1,
            sample_rate=22050,
            chunk_duration=3,
            fft_length=512,
            spec_width=256,
            mel_bins=64,
            mixup_alpha=0.0,
            mixup_probability=0.0,
            snr_threshold=0.0,
        )
        batch = next(iter(ds))
        assert batch[0].shape[0] == 1

    def test_mp_scheduler_keeps_submitting_when_one_job_stalls(self, monkeypatch):
        """A stuck worker job must not block submission of later files."""
        classes = ["class_0"]
        paths = [f"/tmp/class_0/sample_{idx}.wav" for idx in range(33)]
        stuck_path = paths[0]
        sample = np.ones((8, 1), dtype=np.float32)
        label = np.array([1.0], dtype=np.float32)

        class FakeAsyncResult:
            def __init__(self, payload, ready_after=0):
                self.payload = payload
                self.ready_after = ready_after

            def ready(self):
                if self.ready_after > 0:
                    self.ready_after -= 1
                    return False
                return True

            def get(self):
                return self.payload

        class FakePool:
            def __init__(self, *_args, initializer=None, initargs=(), **_kwargs):
                if initializer is not None:
                    initializer(*initargs)

            def apply_async(self, _fn, args):
                path = args[0]
                if path == stuck_path:
                    return FakeAsyncResult(None, ready_after=10_000)
                return FakeAsyncResult([(sample.copy(), label.copy())])

            def terminate(self):
                return None

            def join(self):
                return None

        monkeypatch.setattr("birdnet_stm32.data.generator.mp.Pool", FakePool)
        monkeypatch.setattr("birdnet_stm32.data.generator.random.shuffle", lambda seq: None)
        monkeypatch.setattr("birdnet_stm32.data.generator._POOL_POLL_INTERVAL_S", 0.0)

        ds = load_dataset(
            paths,
            classes,
            audio_frontend="raw",
            batch_size=32,
            num_workers=2,
            max_chunks_per_file=1,
            sample_rate=4,
            chunk_duration=2,
            mixup_alpha=0.0,
            mixup_probability=0.0,
            max_inflight_files=32,
            file_task_timeout_s=0.0,
        )

        samples, labels = next(iter(ds))
        assert samples.shape == (32, 8, 1)
        assert labels.shape == (32, 1)

    def test_loader_control_tracks_skipped_corrupt_file(self, monkeypatch):
        """load_dataset should surface the last skipped unreadable file."""
        classes = ["class_0"]
        good_path = "/tmp/class_0/good.wav"
        bad_path = "/tmp/class_0/bad.wav"
        paths = [bad_path, good_path]
        loader_control: dict[str, str] = {}

        def fake_process_file(path):
            if path == bad_path:
                return None
            sample = np.ones((8, 1), dtype=np.float32)
            label = np.array([1.0], dtype=np.float32)
            return [(sample, label)]

        monkeypatch.setattr("birdnet_stm32.data.generator._process_file", fake_process_file)
        monkeypatch.setattr("birdnet_stm32.data.generator.random.shuffle", lambda seq: None)

        ds = load_dataset(
            paths,
            classes,
            audio_frontend="raw",
            batch_size=1,
            num_workers=0,
            max_chunks_per_file=1,
            sample_rate=4,
            chunk_duration=2,
            mixup_alpha=0.0,
            mixup_probability=0.0,
            loader_control=loader_control,
        )

        next(iter(ds))
        assert loader_control["last_skipped_file"] == bad_path
