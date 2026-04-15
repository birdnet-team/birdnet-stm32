"""Unit tests for spectrogram computation."""

import numpy as np
import pytest

from birdnet_stm32.audio.spectrogram import get_spectrogram_from_audio, normalize


class TestGetSpectrogram:
    """Tests for get_spectrogram_from_audio."""

    def test_output_shape_mel(self, sine_wave, sample_rate, mel_bins, spec_width, fft_length):
        """Mel spectrogram should have shape (mel_bins, spec_width)."""
        spec = get_spectrogram_from_audio(
            sine_wave, sample_rate, n_fft=fft_length, mel_bins=mel_bins, spec_width=spec_width
        )
        assert spec.shape == (mel_bins, spec_width)

    def test_output_shape_linear(self, sine_wave, sample_rate, spec_width, fft_length):
        """Linear spectrogram (mel_bins=-1) should have fft_bins = n_fft//2+1."""
        spec = get_spectrogram_from_audio(
            sine_wave, sample_rate, n_fft=fft_length, mel_bins=-1, spec_width=spec_width
        )
        expected_bins = fft_length // 2 + 1
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
