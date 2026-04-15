"""Unit tests for AudioFrontendLayer."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow required for frontend tests")

from birdnet_stm32.models.frontend import AudioFrontendLayer


@pytest.fixture
def frontend_params():
    """Common frontend parameters."""
    return dict(
        mel_bins=64,
        spec_width=256,
        sample_rate=22050,
        chunk_duration=3,
        fft_length=512,
        mag_scale="none",
    )


class TestPrecomputedMode:
    """Tests for precomputed (pass-through) frontend."""

    def test_output_shape(self, frontend_params):
        """Output shape should match (B, mel_bins, spec_width, 1)."""
        layer = AudioFrontendLayer(mode="precomputed", **frontend_params)
        x = tf.random.uniform((2, 64, 300, 1))
        y = layer(x)
        assert y.shape == (2, 64, 256, 1)

    def test_passthrough_values(self, frontend_params):
        """Without mag_scale, values should pass through with only slicing."""
        layer = AudioFrontendLayer(mode="precomputed", **frontend_params)
        x = tf.ones((1, 64, 256, 1))
        y = layer(x)
        np.testing.assert_allclose(y.numpy(), 1.0, atol=1e-5)


class TestHybridMode:
    """Tests for hybrid (STFT + mel mixer) frontend."""

    def test_output_shape(self, frontend_params):
        """Output should be (B, mel_bins, spec_width, 1)."""
        fft_bins = frontend_params["fft_length"] // 2 + 1
        layer = AudioFrontendLayer(mode="hybrid", **frontend_params)
        x = tf.random.uniform((2, fft_bins, 256, 1))
        y = layer(x)
        assert y.shape[0] == 2
        assert y.shape[1] == frontend_params["mel_bins"]
        assert y.shape[2] == frontend_params["spec_width"]
        assert y.shape[3] == 1

    def test_mel_mixer_initialized(self, frontend_params):
        """Mel mixer kernel should be initialized from librosa basis."""
        layer = AudioFrontendLayer(mode="hybrid", init_mel=True, **frontend_params)
        fft_bins = frontend_params["fft_length"] // 2 + 1
        x = tf.random.uniform((1, fft_bins, 256, 1))
        _ = layer(x)  # build
        weights = layer.get_weights()
        # Should have non-zero weights (mel basis)
        assert any(np.any(w != 0) for w in weights)


class TestRawMode:
    """Tests for raw (waveform) frontend."""

    def test_output_shape(self, frontend_params):
        """Raw frontend should produce (B, mel_bins, spec_width, 1)."""
        params = {**frontend_params, "sample_rate": 16000, "chunk_duration": 2}
        T = params["sample_rate"] * params["chunk_duration"]
        layer = AudioFrontendLayer(mode="raw", **params)
        x = tf.random.uniform((2, T, 1))
        y = layer(x)
        assert y.shape[0] == 2
        assert y.shape[-1] == 1


class TestMagScaling:
    """Tests for magnitude scaling modes."""

    def test_pwl_output_shape(self, frontend_params):
        """PWL scaling should preserve shape."""
        params = {**frontend_params, "mag_scale": "pwl"}
        layer = AudioFrontendLayer(mode="precomputed", **params)
        x = tf.random.uniform((1, 64, 256, 1))
        y = layer(x)
        assert y.shape == (1, 64, 256, 1)

    def test_pcen_output_shape(self, frontend_params):
        """PCEN scaling should preserve shape."""
        params = {**frontend_params, "mag_scale": "pcen"}
        layer = AudioFrontendLayer(mode="precomputed", **params)
        x = tf.random.uniform((1, 64, 256, 1))
        y = layer(x)
        assert y.shape == (1, 64, 256, 1)

    def test_none_output_shape(self, frontend_params):
        """No scaling should preserve shape."""
        layer = AudioFrontendLayer(mode="precomputed", **frontend_params)
        x = tf.random.uniform((1, 64, 256, 1))
        y = layer(x)
        assert y.shape == (1, 64, 256, 1)


class TestSerializationRoundtrip:
    """Test that the layer config can be roundtripped."""

    def test_get_config(self, frontend_params):
        """get_config should return a valid config dict."""
        layer = AudioFrontendLayer(mode="precomputed", **frontend_params)
        config = layer.get_config()
        assert config["mode"] == "precomputed"
        assert config["mel_bins"] == 64
        assert config["spec_width"] == 256
