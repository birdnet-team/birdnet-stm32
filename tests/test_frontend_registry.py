"""Unit tests for the frontend registry and new frontend modes (mfcc, log_mel)."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow required for frontend tests")

from birdnet_stm32.models.frontend import VALID_FRONTENDS, normalize_frontend_name
from birdnet_stm32.models.registry import (
    FrontendInfo,
    get_frontend_info,
    is_n6_compatible,
    is_precomputed,
    list_frontends,
    register_frontend,
)


class TestFrontendRegistry:
    """Tests for frontend registry module."""

    def test_builtin_frontends_registered(self):
        """All built-in frontends should be registered."""
        names = list_frontends()
        assert "librosa" in names
        assert "hybrid" in names
        assert "raw" in names
        assert "mfcc" in names
        assert "log_mel" in names

    def test_get_frontend_info(self):
        """Should return correct info for known frontends."""
        info = get_frontend_info("librosa")
        assert info.name == "librosa"
        assert info.mode == "precomputed"
        assert info.precomputed is True

    def test_unknown_frontend_raises(self):
        """Should raise KeyError for unknown frontend."""
        with pytest.raises(KeyError, match="not registered"):
            get_frontend_info("nonexistent_frontend")

    def test_is_precomputed(self):
        """Precomputed frontends should be identified correctly."""
        assert is_precomputed("librosa") is True
        assert is_precomputed("mfcc") is True
        assert is_precomputed("log_mel") is True
        assert is_precomputed("hybrid") is False
        assert is_precomputed("raw") is False

    def test_is_n6_compatible(self):
        """All built-in frontends should be N6 compatible."""
        for name in list_frontends():
            assert is_n6_compatible(name) is True

    def test_duplicate_registration_raises(self):
        """Registering a frontend twice should raise ValueError."""
        with pytest.raises(ValueError, match="already registered"):
            register_frontend(
                FrontendInfo(
                    name="librosa",
                    mode="precomputed",
                    precomputed=True,
                    n6_compatible=True,
                )
            )


class TestNormalizeFrontendName:
    """Tests for frontend name normalization with new names."""

    def test_valid_frontends(self):
        """All valid frontend names should return unchanged."""
        for name in VALID_FRONTENDS:
            assert normalize_frontend_name(name) == name

    def test_mfcc_valid(self):
        """mfcc should be a valid frontend."""
        assert normalize_frontend_name("mfcc") == "mfcc"

    def test_log_mel_valid(self):
        """log_mel should be a valid frontend."""
        assert normalize_frontend_name("log_mel") == "log_mel"

    def test_deprecated_aliases(self):
        """Deprecated aliases should still work with warnings."""
        with pytest.warns(DeprecationWarning):
            assert normalize_frontend_name("precomputed") == "librosa"
        with pytest.warns(DeprecationWarning):
            assert normalize_frontend_name("tf") == "raw"


class TestMfccSpectrogram:
    """Tests for MFCC spectrogram computation."""

    def test_mfcc_output_shape(self):
        """MFCC should produce (n_mfcc, spec_width) output."""
        from birdnet_stm32.audio.spectrogram import get_spectrogram_from_audio

        audio = np.random.randn(22050 * 3).astype(np.float32)
        S = get_spectrogram_from_audio(audio, sample_rate=22050, mel_bins=64, spec_width=256, mode="mfcc", n_mfcc=20)
        assert S.shape[0] == 20
        assert S.shape[1] == 256

    def test_mfcc_normalized(self):
        """MFCC output should be in [0, 1]."""
        from birdnet_stm32.audio.spectrogram import get_spectrogram_from_audio

        audio = np.random.randn(22050 * 3).astype(np.float32)
        S = get_spectrogram_from_audio(audio, sample_rate=22050, mel_bins=64, spec_width=256, mode="mfcc", n_mfcc=20)
        assert S.min() >= -1e-6
        assert S.max() <= 1.0 + 1e-6


class TestLogMelSpectrogram:
    """Tests for log-mel spectrogram computation."""

    def test_log_mel_output_shape(self):
        """Log-mel should produce (mel_bins, spec_width) output."""
        from birdnet_stm32.audio.spectrogram import get_spectrogram_from_audio

        audio = np.random.randn(22050 * 3).astype(np.float32)
        S = get_spectrogram_from_audio(audio, sample_rate=22050, mel_bins=64, spec_width=256, mode="log_mel")
        assert S.shape == (64, 256)

    def test_log_mel_normalized(self):
        """Log-mel output should be in [0, 1]."""
        from birdnet_stm32.audio.spectrogram import get_spectrogram_from_audio

        audio = np.random.randn(22050 * 3).astype(np.float32)
        S = get_spectrogram_from_audio(audio, sample_rate=22050, mel_bins=64, spec_width=256, mode="log_mel")
        assert S.min() >= -1e-6
        assert S.max() <= 1.0 + 1e-6


class TestMfccFrontendLayer:
    """Tests for MFCC frontend through the DS-CNN model builder."""

    def test_mfcc_model_builds(self):
        """Model with mfcc frontend should build without errors."""
        from birdnet_stm32.models.dscnn import build_dscnn_model

        model = build_dscnn_model(
            num_mels=64,
            spec_width=64,
            sample_rate=22050,
            chunk_duration=3,
            embeddings_size=32,
            num_classes=5,
            audio_frontend="mfcc",
            alpha=0.25,
            depth_multiplier=1,
            n_mfcc=20,
        )
        assert model.input_shape == (None, 20, 64, 1)
        assert model.output_shape == (None, 5)

    def test_log_mel_model_builds(self):
        """Model with log_mel frontend should build without errors."""
        from birdnet_stm32.models.dscnn import build_dscnn_model

        model = build_dscnn_model(
            num_mels=64,
            spec_width=64,
            sample_rate=22050,
            chunk_duration=3,
            embeddings_size=32,
            num_classes=5,
            audio_frontend="log_mel",
            alpha=0.25,
            depth_multiplier=1,
        )
        assert model.input_shape == (None, 64, 64, 1)
        assert model.output_shape == (None, 5)
