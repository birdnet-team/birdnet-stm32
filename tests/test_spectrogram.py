"""Unit tests for spectrogram computation."""

import numpy as np
import pytest

from birdnet_stm32.audio.spectrogram import get_spectrogram_from_audio, normalize
from birdnet_stm32.models.frontend import hybrid_fft_bins


class TestGetSpectrogram:
    """Tests for get_spectrogram_from_audio."""

    def test_output_shape_mel(self, sine_wave, sample_rate, mel_bins, spec_width, fft_length):
        """Mel spectrogram should have shape (mel_bins, spec_width)."""
        spec = get_spectrogram_from_audio(
            sine_wave, sample_rate, n_fft=fft_length, mel_bins=mel_bins, spec_width=spec_width
        )
        assert spec.shape == (mel_bins, spec_width)

    def test_output_shape_linear(self, sine_wave, sample_rate, spec_width, fft_length):
        """Linear spectrogram (mel_bins=-1) drops Nyquist: fft_bins = n_fft//2."""
        spec = get_spectrogram_from_audio(sine_wave, sample_rate, n_fft=fft_length, mel_bins=-1, spec_width=spec_width)
        expected_bins = hybrid_fft_bins(fft_length)
        assert spec.shape[0] == expected_bins
        assert spec.shape[1] == spec_width

    def test_silence_low_energy(self, silence, sample_rate, mel_bins, spec_width, fft_length):
        """Silence should produce near-zero spectrogram values."""
        spec = get_spectrogram_from_audio(
            silence, sample_rate, n_fft=fft_length, mel_bins=mel_bins, spec_width=spec_width
        )
        assert np.max(np.abs(spec)) < 1e-3

    def test_dtype_float32(self, sine_wave, sample_rate, fft_length):
        """Output should be float32."""
        spec = get_spectrogram_from_audio(sine_wave, sample_rate, n_fft=fft_length, mel_bins=32, spec_width=64)
        assert spec.dtype == np.float32

    @pytest.mark.parametrize("mode", ["mfcc", "log_mel", "garbage"])
    def test_removed_or_unknown_modes_fail(self, sine_wave, mode):
        with pytest.raises(ValueError, match="spectrogram mode"):
            get_spectrogram_from_audio(sine_wave, mode=mode)

    @pytest.mark.parametrize("mag_scale", ["pcen", "db", "garbage"])
    def test_removed_or_unknown_magnitude_scales_fail(self, sine_wave, mag_scale):
        with pytest.raises(ValueError, match="magnitude scale"):
            get_spectrogram_from_audio(sine_wave, mag_scale=mag_scale)

    @pytest.mark.parametrize("compression", ["sqrt", "log"])
    def test_compression_lifts_quiet_values(self, sine_wave, sample_rate, fft_length, compression):
        """Compression spends more of the [0, 1] range on low magnitudes."""
        audio = sine_wave + 0.01 * np.random.default_rng(0).standard_normal(sine_wave.shape).astype(np.float32)
        plain = get_spectrogram_from_audio(audio, sample_rate, n_fft=fft_length, mel_bins=-1, spec_width=64)
        squeezed = get_spectrogram_from_audio(
            audio, sample_rate, n_fft=fft_length, mel_bins=-1, spec_width=64, compression=compression
        )
        assert squeezed.shape == plain.shape
        assert squeezed.min() >= 0.0 and squeezed.max() <= 1.0 + 1e-6
        assert np.median(squeezed) > np.median(plain)

    def test_log_floor_bounds_dynamic_range(self, sine_wave, sample_rate, fft_length):
        """Everything more than LOG_FLOOR_DB below the peak maps to exactly 0."""
        spec = get_spectrogram_from_audio(
            sine_wave, sample_rate, n_fft=fft_length, mel_bins=-1, spec_width=64, compression="log"
        )
        assert np.isclose(spec.min(), 0.0) and np.isclose(spec.max(), 1.0)

    def test_log_compression_of_silence_is_finite(self, silence, sample_rate, fft_length):
        spec = get_spectrogram_from_audio(
            silence, sample_rate, n_fft=fft_length, mel_bins=-1, spec_width=64, compression="log"
        )
        assert np.all(np.isfinite(spec))

    def test_unknown_compression_fails(self, sine_wave):
        with pytest.raises(ValueError, match="input compression"):
            get_spectrogram_from_audio(sine_wave, compression="db")


class TestNormalize:
    """Tests for the normalize function."""

    def test_unit_range(self):
        """After normalization, values should be in [0, 1] (approx)."""
        data = np.random.rand(64, 128).astype(np.float32)
        normed = normalize(data)
        assert normed.max() <= 1.0 + 1e-6
        assert normed.min() >= -1e-6

    def test_constant_input(self):
        """Constant input should normalize to zeros."""
        data = np.ones((64, 128), dtype=np.float32) * 5.0
        normed = normalize(data)
        assert np.allclose(normed, 0.0, atol=1e-6)


def test_with_uncompressed_pairs_match_single_calls(sine_wave, sample_rate, fft_length):
    compressed, plain = get_spectrogram_from_audio(
        sine_wave, sample_rate, n_fft=fft_length, mel_bins=32, spec_width=64, compression="log", with_uncompressed=True
    )
    assert np.array_equal(
        compressed,
        get_spectrogram_from_audio(
            sine_wave, sample_rate, n_fft=fft_length, mel_bins=32, spec_width=64, compression="log"
        ),
    )
    assert np.array_equal(
        plain, get_spectrogram_from_audio(sine_wave, sample_rate, n_fft=fft_length, mel_bins=32, spec_width=64)
    )
