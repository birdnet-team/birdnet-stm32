"""Integration test: librosa spectrogram ≈ hybrid frontend output (within tolerance)."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow required for frontend parity tests")

from birdnet_stm32.audio.spectrogram import get_spectrogram_from_audio
from birdnet_stm32.conversion.validate import cosine_similarity
from birdnet_stm32.models.frontend import AudioFrontendLayer


@pytest.mark.integration
class TestFrontendParity:
    """Verify that the hybrid Conv2D mel mixer approximates the librosa mel basis."""

    def test_hybrid_vs_librosa_mel(self, sine_wave, sample_rate):
        """Hybrid frontend mel output should be correlated with librosa mel."""
        n_fft = 512
        mel_bins = 64
        spec_width = 256

        # ── Librosa reference ──
        mel_spec = get_spectrogram_from_audio(
            sine_wave,
            sample_rate=sample_rate,
            n_fft=n_fft,
            mel_bins=mel_bins,
            spec_width=spec_width,
            mag_scale="none",
        )

        # ── Hybrid frontend with initialized mel basis ──
        linear_spec = get_spectrogram_from_audio(
            sine_wave,
            sample_rate=sample_rate,
            n_fft=n_fft,
            mel_bins=-1,
            spec_width=spec_width,
            mag_scale="none",
        )

        layer = AudioFrontendLayer(
            mode="hybrid",
            mel_bins=mel_bins,
            spec_width=spec_width,
            sample_rate=sample_rate,
            chunk_duration=3,
            fft_length=n_fft,
            init_mel=True,
            mag_scale="none",
        )
        # Build by running a forward pass
        x_in = tf.constant(linear_spec[np.newaxis, :, :, np.newaxis], dtype=tf.float32)
        hybrid_out = layer(x_in).numpy().squeeze()

        # Shapes should match
        assert mel_spec.shape[0] == mel_bins
        assert hybrid_out.shape[0] == mel_bins

        # The outputs should be positively correlated (cosine > 0.5)
        # since the mel mixer is initialized from the same librosa basis.
        cos = cosine_similarity(mel_spec.ravel(), hybrid_out.ravel())
        assert cos > 0.5, f"Hybrid vs librosa cosine similarity too low: {cos:.4f}"
