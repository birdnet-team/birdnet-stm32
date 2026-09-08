"""Unit tests for MagnitudeScalingLayer."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow required for magnitude tests")

from birdnet_stm32.models.magnitude import VALID_MAG_SCALES, MagnitudeScalingLayer


class TestMagnitudeScalingLayer:
    """Tests for MagnitudeScalingLayer."""

    @pytest.mark.parametrize("method", ["none", "pwl"])
    def test_output_shape(self, method):
        """All methods should preserve input shape."""
        layer = MagnitudeScalingLayer(method=method, channels=8)
        x = tf.random.uniform((2, 8, 16, 1))
        y = layer(x)
        assert y.shape == (2, 8, 16, 1)

    def test_none_passthrough(self):
        """'none' method should pass input through unchanged."""
        layer = MagnitudeScalingLayer(method="none", channels=8)
        x = tf.constant(np.ones((1, 8, 16, 1), dtype=np.float32))
        y = layer(x)
        np.testing.assert_allclose(y.numpy(), x.numpy())

    def test_invalid_method_raises(self):
        """Invalid method should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid mag_scale"):
            MagnitudeScalingLayer(method="invalid", channels=8)

    def test_get_config(self):
        """get_config should return a valid config dict."""
        layer = MagnitudeScalingLayer(method="pwl", channels=32)
        cfg = layer.get_config()
        assert cfg["method"] == "pwl"
        assert cfg["channels"] == 32

    def test_valid_mag_scales_constant(self):
        """Only the two scales any release uses remain.

        PCEN was never used by a release and dB's log op produces a dynamic
        range INT8 cannot hold, which is the failure this frontend exists to
        avoid. Both were removed rather than kept as untested options.
        """
        assert set(VALID_MAG_SCALES) == {"none", "pwl"}

    def test_pwl_finite_output(self):
        """PWL output should be finite for arbitrary input."""
        layer = MagnitudeScalingLayer(method="pwl", channels=8)
        x = tf.random.uniform((2, 8, 16, 1), minval=-1.0, maxval=1.0)
        y = layer(x)
        assert np.all(np.isfinite(y.numpy()))
