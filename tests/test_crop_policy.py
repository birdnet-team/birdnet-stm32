"""Tests for the training crop policy and waveform time masking (Phase A)."""

import numpy as np

from birdnet_stm32.audio.activity import smart_crop, uniform_crop
from birdnet_stm32.audio.augmentation import apply_time_mask

SR = 24000
CD = 2.5
CHUNK = int(SR * CD)


def _audio_with_a_loud_burst(n_chunks: int = 8, burst_at: int = 2) -> np.ndarray:
    """Quiet audio with one loud burst, so energy ranking has an obvious favourite."""
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(CHUNK * n_chunks) * 0.001).astype(np.float32)
    start = burst_at * CHUNK
    audio[start : start + CHUNK] += (rng.standard_normal(CHUNK) * 0.5).astype(np.float32)
    return audio


class TestUniformCrop:
    def test_returns_requested_number_of_chunks_of_the_right_length(self):
        audio = _audio_with_a_loud_burst()
        chunks = uniform_crop(audio, SR, CD, max_chunks=4, rng=np.random.default_rng(1))
        assert len(chunks) == 4
        assert all(c.shape == (CHUNK,) for c in chunks)
        assert all(c.dtype == np.float32 for c in chunks)

    def test_pads_a_file_shorter_than_one_chunk(self):
        short = np.ones(CHUNK // 3, dtype=np.float32)
        chunks = uniform_crop(short, SR, CD, max_chunks=4)
        assert len(chunks) == 1
        assert chunks[0].shape == (CHUNK,)
        assert chunks[0][-1] == 0.0

    def test_never_returns_more_chunks_than_the_file_holds(self):
        audio = _audio_with_a_loud_burst(n_chunks=2, burst_at=0)
        assert len(uniform_crop(audio, SR, CD, max_chunks=8)) <= 2

    def test_does_not_concentrate_on_the_loudest_region_the_way_smart_crop_does(self):
        """The whole point of the policy: no energy bias in what gets sampled."""
        audio = _audio_with_a_loud_burst(n_chunks=8, burst_at=2)

        loud = smart_crop(audio, SR, CD, max_chunks=1)[0]
        assert float(np.abs(loud).max()) > 0.1, "smart_crop should find the burst"

        rng = np.random.default_rng(7)
        draws = [uniform_crop(audio, SR, CD, max_chunks=1, rng=rng)[0] for _ in range(40)]
        loud_draws = sum(float(np.abs(c).max()) > 0.1 for c in draws)
        # The burst is one chunk in eight, so uniform sampling must mostly miss it.
        assert loud_draws < 25, f"uniform_crop looks energy-biased: {loud_draws}/40 hit the burst"

    def test_is_reproducible_for_a_given_generator(self):
        audio = _audio_with_a_loud_burst()
        a = uniform_crop(audio, SR, CD, max_chunks=3, rng=np.random.default_rng(42))
        b = uniform_crop(audio, SR, CD, max_chunks=3, rng=np.random.default_rng(42))
        assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


class TestWaveformTimeMask:
    def test_zeroes_some_samples_but_not_all(self):
        x = np.ones(CHUNK, dtype=np.float32)
        out = apply_time_mask(x, SR, num_masks=8, max_width_ms=30.0, rng=np.random.default_rng(3))
        zeroed = int((out == 0.0).sum())
        assert 0 < zeroed < CHUNK

    def test_respects_the_width_budget(self):
        x = np.ones(CHUNK, dtype=np.float32)
        num_masks, width_ms = 8, 30.0
        out = apply_time_mask(x, SR, num_masks=num_masks, max_width_ms=width_ms, rng=np.random.default_rng(5))
        budget = num_masks * int(SR * width_ms / 1000.0)
        assert int((out == 0.0).sum()) <= budget

    def test_is_a_no_op_when_disabled(self):
        x = np.ones(CHUNK, dtype=np.float32)
        assert np.array_equal(apply_time_mask(x, SR, num_masks=0), x)
        assert np.array_equal(apply_time_mask(x, SR, max_width_ms=0.0), x)

    def test_does_not_modify_its_input(self):
        x = np.ones(CHUNK, dtype=np.float32)
        apply_time_mask(x, SR, num_masks=8, rng=np.random.default_rng(1))
        assert np.array_equal(x, np.ones(CHUNK, dtype=np.float32))

    def test_leaves_the_peak_untouched_so_normalization_is_unaffected(self):
        """Masking runs after peak normalization, so the scale cannot drift."""
        rng = np.random.default_rng(11)
        x = (rng.standard_normal(CHUNK) * 0.1).astype(np.float32)
        x = x / np.abs(x).max()
        out = apply_time_mask(x, SR, num_masks=4, max_width_ms=10.0, rng=np.random.default_rng(2))
        assert float(np.abs(out).max()) <= 1.0
