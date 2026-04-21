"""Unit tests for audio I/O utilities."""

import numpy as np
import soundfile as sf

from birdnet_stm32.audio.io import fast_resample, load_audio_file


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


class TestLoadAudioFile:
    """Tests for chunked file loading."""

    def test_short_file_preserves_leading_audio(self, tmp_path):
        """Short files should be padded, not truncated from the front."""
        sample_rate = 16000
        audio = np.linspace(-1.0, 1.0, sample_rate, dtype=np.float32)
        path = tmp_path / "short.wav"
        sf.write(str(path), audio, sample_rate, subtype="FLOAT")

        chunks = load_audio_file(str(path), sample_rate=sample_rate, chunk_duration=3.0)

        assert chunks.shape == (1, sample_rate * 3)
        expected = audio / (np.max(np.abs(audio)) + 1e-10)
        np.testing.assert_allclose(chunks[0, : audio.shape[0]], expected, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(chunks[0, audio.shape[0] :], 0.0, atol=1e-7)
