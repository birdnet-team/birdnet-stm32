"""Unit tests for activity detection and signal-to-noise sorting."""

import numpy as np

from birdnet_stm32.audio.activity import (
    get_activity_ratio,
    get_s2n_from_audio,
    get_s2n_from_spectrogram,
    pick_random_samples,
    sort_by_activity,
    sort_by_s2n,
)


class TestS2NFromSpectrogram:
    """Tests for get_s2n_from_spectrogram."""

    def test_constant_input(self):
        """Constant spectrogram has near-infinite SNR (mean/std -> high)."""
        spec = np.ones((64, 128), dtype=np.float32)
        snr = get_s2n_from_spectrogram(spec)
        assert snr > 1e6  # std ≈ 0 -> large ratio

    def test_noisy_input(self):
        """Random spectrogram should return finite positive SNR."""
        rng = np.random.default_rng(0)
        spec = rng.standard_normal((64, 128)).astype(np.float32)
        snr = get_s2n_from_spectrogram(spec)
        assert np.isfinite(snr)


class TestS2NFromAudio:
    """Tests for get_s2n_from_audio."""

    def test_silence(self):
        """All-zero audio should return near-zero SNR."""
        audio = np.zeros(1000, dtype=np.float32)
        snr = get_s2n_from_audio(audio)
        assert abs(snr) < 1e-3

    def test_sine(self, sine_wave):
        """Sine wave should return a finite SNR."""
        snr = get_s2n_from_audio(sine_wave)
        assert np.isfinite(snr)


class TestSortByS2N:
    """Tests for sort_by_s2n."""

    def test_ordering(self):
        """Higher-SNR samples should come first."""
        quiet = np.random.default_rng(0).standard_normal((64, 128)).astype(np.float32) * 0.01
        loud = np.ones((64, 128), dtype=np.float32) + 0.01 * np.random.default_rng(1).standard_normal((64, 128)).astype(
            np.float32
        )
        result = sort_by_s2n([quiet, loud], threshold=0.0)
        # Loud should sort before quiet
        assert np.mean(result[0]) > np.mean(result[1])

    def test_keeps_at_least_one(self):
        """Even with high threshold, at least one sample is kept."""
        samples = [np.zeros((64, 128), dtype=np.float32)]
        result = sort_by_s2n(samples, threshold=0.99)
        assert len(result) >= 1

    def test_1d_audio(self):
        """Should work with 1D audio arrays and return sorted results."""
        samples = [np.random.default_rng(i).standard_normal(1000).astype(np.float32) for i in range(5)]
        result = sort_by_s2n(samples, threshold=0.0)
        assert len(result) >= 1  # at least one survives


class TestGetActivityRatio:
    """Tests for get_activity_ratio."""

    def test_silence_low_activity(self):
        """All-zero input should have zero activity."""
        assert get_activity_ratio(np.zeros(1000, dtype=np.float32)) == 0.0

    def test_spike_gives_activity(self):
        """Input with a clear spike should have nonzero activity."""
        x = np.zeros(1000, dtype=np.float32)
        x[500] = 100.0  # big spike
        ratio = get_activity_ratio(x, k=2.0)
        assert ratio > 0.0

    def test_broadband_returns_zero(self):
        """Uniformly high input exceeds max_active and returns 0.0."""
        x = np.ones(1000, dtype=np.float32) * 10.0
        ratio = get_activity_ratio(x, max_active=0.8)
        # Constant -> all above threshold -> ratio > max_active -> returns 0.0
        assert ratio == 0.0

    def test_range(self):
        """Activity ratio should be in [0, 1]."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(2000).astype(np.float32)
        ratio = get_activity_ratio(x)
        assert 0.0 <= ratio <= 1.0


class TestSortByActivity:
    """Tests for sort_by_activity."""

    def test_keeps_at_least_one(self):
        """Even with high threshold, at least one sample is kept."""
        samples = [np.zeros(1000, dtype=np.float32)]
        result = sort_by_activity(samples, threshold=0.99)
        assert len(result) >= 1

    def test_active_first(self):
        """Active samples should come before quiet ones."""
        quiet = np.zeros(1000, dtype=np.float32)
        active = np.zeros(1000, dtype=np.float32)
        active[::10] = 50.0  # periodic spikes
        result = sort_by_activity([quiet, active], threshold=0.0)
        # Active should sort first (higher activity ratio)
        assert np.max(np.abs(result[0])) > np.max(np.abs(result[1]))


class TestPickRandomSamples:
    """Tests for pick_random_samples."""

    def test_pick_first(self):
        """pick_first=True should return the first sample."""
        samples = [np.array([1.0]), np.array([2.0]), np.array([3.0])]
        result = pick_random_samples(samples, pick_first=True)
        np.testing.assert_array_equal(result, samples[0])

    def test_single_sample(self):
        """num_samples=1 returns an ndarray, not a list."""
        samples = [np.array([1.0]), np.array([2.0])]
        result = pick_random_samples(samples, num_samples=1)
        assert isinstance(result, np.ndarray)

    def test_multiple_samples(self):
        """num_samples > 1 returns a list of that length."""
        samples = [np.array([i]) for i in range(10)]
        result = pick_random_samples(samples, num_samples=3)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_empty_input(self):
        """Empty input returns empty list."""
        result = pick_random_samples([], num_samples=1)
        assert result == []

    def test_num_samples_exceeds_available(self):
        """Requesting more samples than available should clamp."""
        samples = [np.array([1.0]), np.array([2.0])]
        result = pick_random_samples(samples, num_samples=10)
        assert isinstance(result, list)
        assert len(result) == 2
