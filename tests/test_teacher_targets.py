"""Tests for cached teacher targets and their use in the training loader."""

import json

import numpy as np
import pytest
import soundfile as sf

from birdnet_stm32.audio.activity import smart_crop, uniform_crop
from birdnet_stm32.audio.io import chunk_start_samples, load_audio_window, split_audio_into_chunks
from birdnet_stm32.data.teacher import TeacherTargets, blend_targets

CLASSES = ["a", "b", "c"]
SR = 24000
CD = 2.5


def write_cache(root, classes=CLASSES, mask=(True, True, False), recordings=None, window_s=3.0):
    """Write a tiny cache: recordings maps sample_id -> (starts, scores[n, C])."""
    recordings = recordings or {
        "rec1": ([0.0, 1.25, 2.5], [[0.9, 0.0, 0.0], [0.1, 0.8, 0.0], [0.0, 0.0, 0.7]]),
    }
    root.mkdir(parents=True, exist_ok=True)
    ids, offsets, counts, starts, mapped = [], [], [], [], []
    row = 0
    for sid, (s, m) in recordings.items():
        ids.append(sid)
        offsets.append(row)
        counts.append(len(s))
        starts.extend(s)
        mapped.extend(m)
        row += len(s)
    np.save(root / "mapped.npy", np.asarray(mapped, dtype=np.float16))
    np.save(root / "starts.npy", np.asarray(starts, dtype=np.float32))
    np.savez(
        root / "index.npz",
        sample_id=np.array(ids),
        row_offset=np.array(offsets, dtype=np.int64),
        n_windows=np.array(counts, dtype=np.int32),
    )
    (root / "meta.json").write_text(
        json.dumps({"classes": list(classes), "teacher_mask": list(mask), "teacher_window_s": window_s})
    )
    return root


class TestTeacherTargets:
    def test_picks_the_window_whose_centre_is_nearest_the_chunk(self, tmp_path):
        t = TeacherTargets(write_cache(tmp_path / "c"), CLASSES)
        # Teacher windows are 3.0 s, centred at 1.5, 2.75, 4.0.
        # A 2.5 s chunk at 0.0 is centred at 1.25 -> window 0.
        np.testing.assert_allclose(t.lookup("rec1", 0.0, CD), [0.9, 0.0, 0.0], atol=1e-3)
        # A chunk at 1.5 is centred at 2.75 -> window 1.
        np.testing.assert_allclose(t.lookup("rec1", 1.5, CD), [0.1, 0.8, 0.0], atol=1e-3)
        # A chunk at 2.8 is centred at 4.05 -> window 2.
        np.testing.assert_allclose(t.lookup("rec1", 2.8, CD), [0.0, 0.0, 0.7], atol=1e-3)

    def test_returns_none_for_an_unknown_recording(self, tmp_path):
        t = TeacherTargets(write_cache(tmp_path / "c"), CLASSES)
        assert t.lookup("missing", 0.0, CD) is None
        assert "rec1" in t and "missing" not in t

    def test_returns_none_when_no_window_is_near_the_chunk(self, tmp_path):
        t = TeacherTargets(write_cache(tmp_path / "c"), CLASSES)
        assert t.lookup("rec1", 60.0, CD) is None

    def test_refuses_a_cache_built_for_another_class_list(self, tmp_path):
        root = write_cache(tmp_path / "c")
        with pytest.raises(ValueError, match="different class list"):
            TeacherTargets(root, ["a", "b", "x"])

    def test_refuses_a_mask_of_the_wrong_length(self, tmp_path):
        root = write_cache(tmp_path / "c", mask=(True, True))
        with pytest.raises(ValueError, match="teacher_mask"):
            TeacherTargets(root, CLASSES)


class TestBlendTargets:
    def test_zero_weight_is_the_hard_label(self):
        hard = np.array([1.0, 0.0, 0.0], np.float32)
        out = blend_targets(hard, np.array([0.1, 0.9, 0.9], np.float32), np.array([True, True, True]), 0.0)
        np.testing.assert_array_equal(out, hard)

    def test_blends_only_where_the_teacher_has_an_output(self):
        hard = np.array([1.0, 0.0, 1.0], np.float32)
        teacher = np.array([0.0, 0.8, 0.0], np.float32)
        out = blend_targets(hard, teacher, np.array([True, True, False]), 0.5)
        # Labelled class the teacher does not hear: softened to 0.5, not dropped.
        assert out[0] == pytest.approx(0.5)
        # Unlabelled species the teacher hears: raised toward it.
        assert out[1] == pytest.approx(0.4)
        # A class the teacher has no output for keeps its hard label.
        assert out[2] == pytest.approx(1.0)

    def test_stays_a_valid_bce_target(self):
        rng = np.random.default_rng(0)
        hard = (rng.random(50) > 0.5).astype(np.float32)
        teacher = rng.random(50).astype(np.float32) * 1.2 - 0.1  # out of range on purpose
        out = blend_targets(hard, teacher, np.ones(50, bool), 0.7)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_does_not_modify_the_hard_label(self):
        hard = np.array([1.0, 0.0], np.float32)
        blend_targets(hard, np.array([0.0, 1.0], np.float32), np.array([True, True]), 0.5)
        np.testing.assert_array_equal(hard, [1.0, 0.0])

    def test_rejects_a_weight_outside_zero_one(self):
        with pytest.raises(ValueError):
            blend_targets(np.zeros(2), np.zeros(2), np.ones(2, bool), 1.5)


class TestChunkStarts:
    """Chunk positions must be exact, or every teacher lookup is misaligned."""

    def test_match_what_split_audio_into_chunks_emits(self):
        rng = np.random.default_rng(1)
        for n in (100, int(SR * CD), int(SR * CD) + 1, int(SR * 7.3), int(SR * 10)):
            audio = rng.standard_normal(n).astype(np.float32)
            chunks = split_audio_into_chunks(audio, SR, CD)
            starts = chunk_start_samples(n, SR, CD)
            assert len(starts) == len(chunks)
            for s, c in zip(starts, chunks, strict=True):
                if n > int(SR * CD):
                    np.testing.assert_array_equal(audio[s : s + int(SR * CD)], c)

    def test_smart_crop_starts_locate_its_chunks(self):
        rng = np.random.default_rng(2)
        audio = (rng.standard_normal(SR * 12) * 0.01).astype(np.float32)
        audio[SR * 5 : SR * 6] += 1.0
        chunks, starts = smart_crop(audio, SR, CD, max_chunks=3, return_starts=True)
        for s, c in zip(starts, chunks, strict=True):
            np.testing.assert_array_equal(audio[s : s + int(SR * CD)], c)

    def test_uniform_crop_starts_locate_its_chunks(self):
        rng = np.random.default_rng(3)
        audio = rng.standard_normal(SR * 12).astype(np.float32)
        chunks, starts = uniform_crop(audio, SR, CD, max_chunks=3, rng=rng, return_starts=True)
        for s, c in zip(starts, chunks, strict=True):
            np.testing.assert_array_equal(audio[s : s + int(SR * CD)], c)

    def test_without_return_starts_the_signatures_are_unchanged(self):
        audio = np.ones(SR * 8, dtype=np.float32)
        assert isinstance(smart_crop(audio, SR, CD, max_chunks=2), list)
        assert isinstance(uniform_crop(audio, SR, CD, max_chunks=2), list)

    def test_load_audio_window_reports_where_it_read(self, tmp_path):
        """The offset returned must be where the samples really came from."""
        n = SR * 20
        ramp = (np.arange(n, dtype=np.float64) / n).astype(np.float32)  # value encodes position
        path = tmp_path / "ramp.wav"
        sf.write(path, ramp, SR, subtype="FLOAT")
        np.random.seed(5)
        audio, offset = load_audio_window(
            str(path), SR, max_duration=4.0, chunk_duration=CD, random_offset=True, return_offset=True
        )
        assert 0.0 < offset < 20.0 - 4.0
        # Peak normalization divides by the window's last (largest) value.
        expected_first = (offset * SR) / n
        np.testing.assert_allclose(audio[0] * ramp[int(round(offset * SR)) + len(audio) - 1], expected_first, atol=1e-4)

    def test_load_audio_window_keeps_its_old_return_type(self, tmp_path):
        path = tmp_path / "x.wav"
        sf.write(path, np.ones(SR * 3, dtype=np.float32) * 0.5, SR)
        assert isinstance(load_audio_window(str(path), SR, max_duration=2.0), np.ndarray)


class TestWorkerIntegration:
    """The loader worker must attach the teacher target for the chunk it emits."""

    def _cfg(self, cache, weight):
        return {
            "audio_frontend": "raw",
            "sr": SR,
            "cd": CD,
            "T": int(SR * CD),
            "fft_length": 512,
            "mel_bins": 64,
            "spec_width": 256,
            "mag_scale": "pwl",
            "input_compression": "none",
            "max_duration": 60,
            "load_duration": 60,
            "snr_threshold": 0.1,
            "random_offset": False,
            "spec_augment": False,
            "freq_mask_max": 8,
            "time_mask_max": 25,
            "noise_labels": ("noise",),
            "class_to_idx": {c: i for i, c in enumerate(CLASSES)},
            "num_classes": len(CLASSES),
            "max_chunks_per_file": 1,
            "candidate_chunks_per_file": 4,
            "classes": CLASSES,
            "teacher_cache": str(cache) if cache else None,
            "teacher_weight": weight,
        }

    def _write_audio(self, tmp_path):
        d = tmp_path / "audio" / "a"
        d.mkdir(parents=True)
        path = d / "rec1.wav"
        rng = np.random.default_rng(0)
        # A single 2.5 s file: exactly one chunk, starting at 0.
        sf.write(path, (rng.standard_normal(int(SR * CD)) * 0.1).astype(np.float32), SR)
        return path

    def test_blends_the_teacher_into_the_emitted_target(self, tmp_path):
        from birdnet_stm32.data import worker

        cache = write_cache(tmp_path / "cache")
        path = self._write_audio(tmp_path)
        worker._init_worker(self._cfg(cache, 0.5))
        try:
            ((sample, target),) = worker._process_file(str(path))
        finally:
            worker._init_worker(self._cfg(None, 0.0))
        # Chunk at 0 -> teacher window 0 = [0.9, 0.0, 0.0]; hard label = class "a".
        np.testing.assert_allclose(target, [0.5 * 1.0 + 0.5 * 0.9, 0.0, 0.0], atol=1e-3)
        assert sample.shape == (int(SR * CD), 1)

    def test_without_a_teacher_the_target_is_the_hard_label(self, tmp_path):
        from birdnet_stm32.data import worker

        path = self._write_audio(tmp_path)
        worker._init_worker(self._cfg(None, 0.0))
        ((_, target),) = worker._process_file(str(path))
        np.testing.assert_array_equal(target, [1.0, 0.0, 0.0])

    def test_a_recording_missing_from_the_cache_keeps_its_hard_label(self, tmp_path):
        from birdnet_stm32.data import worker

        cache = write_cache(tmp_path / "cache", recordings={"other": ([0.0], [[0.0, 1.0, 0.0]])})
        path = self._write_audio(tmp_path)
        worker._init_worker(self._cfg(cache, 0.5))
        try:
            ((_, target),) = worker._process_file(str(path))
        finally:
            worker._init_worker(self._cfg(None, 0.0))
        np.testing.assert_array_equal(target, [1.0, 0.0, 0.0])
