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
        assert set(VALID_MAG_SCALES) == {"none", "pwl", "cpwl"}

    def test_pwl_finite_output(self):
        """PWL output should be finite for arbitrary input."""
        layer = MagnitudeScalingLayer(method="pwl", channels=8)
        x = tf.random.uniform((2, 8, 16, 1), minval=-1.0, maxval=1.0)
        y = layer(x)
        assert np.all(np.isfinite(y.numpy()))


class TestCompressivePwl:
    """cpwl: the same hinge sum as pwl, held concave and monotone."""

    @staticmethod
    def _curve(method, xs):
        import numpy as np
        import tensorflow as tf

        layer = MagnitudeScalingLayer(method=method, channels=1, is_trainable=True)
        x = tf.constant(np.asarray(xs, np.float32).reshape(1, 1, -1, 1))
        layer.build(x.shape)
        return np.asarray(layer(x)).ravel()

    def test_initial_curve_is_monotone_and_concave(self):
        import numpy as np

        # Coarse grid: float32 rounding over a fine one swamps the slope diffs.
        xs = np.linspace(0.0, 10.0, 201)
        slope = np.diff(self._curve("cpwl", xs)) / np.diff(xs)
        assert (slope >= -1e-3).all()
        assert (np.diff(slope) <= 1e-3).all()
        assert slope[0] > 5 * slope[-1]

    def test_squeezes_the_tail_that_pwl_stretches(self):
        cp, p = self._curve("cpwl", [10.0])[0], self._curve("pwl", [10.0])[0]
        assert cp < p / 5

    def test_constraints_restore_the_compressive_signs(self):
        import numpy as np
        import tensorflow as tf

        layer = MagnitudeScalingLayer(method="cpwl", channels=4, is_trainable=True)
        layer.build(tf.TensorShape([None, 1, 8, 4]))
        for sub_layer, bad in [(layer._pwl_k0_dw, -1.0), (layer._pwl_shift_dws[0], -1.0), (layer._pwl_k_dws[0], 1.0)]:
            kernel = sub_layer.weights[0]
            kernel.assign(tf.fill(kernel.shape, bad))
            kernel.assign(sub_layer.depthwise_constraint(kernel))
            assert np.all(np.asarray(kernel) == 0.0)

    def test_pwl_stays_unconstrained(self):
        layer = MagnitudeScalingLayer(method="pwl", channels=4, is_trainable=True)
        assert layer._pwl_k0_dw.depthwise_constraint is None
        assert all(k.depthwise_constraint is None for k in layer._pwl_k_dws)
