"""Unit tests for audio I/O utilities."""

import numpy as np

from birdnet_stm32.audio.io import fast_resample


class TestFastResample:
    """Tests for the fast_resample function."""

    def test_upsample(self):
        """Upsampling should increase the number of samples."""
        audio = np.sin(np.linspace(0, 2 * np.pi, 1000, dtype=np.float32))
        resampled = fast_resample(audio, 16000, 22050)
        expected_len = int(len(audio) * 22050 / 16000)
        assert abs(len(resampled) - expected_len) <= 1

    def test_downsample(self):
        """Downsampling should decrease the number of samples."""
        audio = np.sin(np.linspace(0, 2 * np.pi, 2000, dtype=np.float32))
        resampled = fast_resample(audio, 44100, 22050)
        expected_len = int(len(audio) * 22050 / 44100)
        assert abs(len(resampled) - expected_len) <= 1

    def test_same_rate(self):
        """Same rate should return unchanged audio."""
        audio = np.ones(100, dtype=np.float32)
        resampled = fast_resample(audio, 22050, 22050)
        np.testing.assert_array_equal(audio, resampled)
