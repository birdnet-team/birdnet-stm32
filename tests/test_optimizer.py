"""Unit tests for training utilities."""

import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow required for trainer tests")

from birdnet_stm32.training.trainer import VALID_OPTIMIZERS, _build_optimizer


class TestBuildOptimizer:
    """Tests for _build_optimizer."""

    def _make_schedule(self):
        return tf.keras.optimizers.schedules.CosineDecay(1e-3, 100)

    def test_adam(self):
        """'adam' should return an Adam optimizer."""
        opt = _build_optimizer("adam", self._make_schedule())
        assert isinstance(opt, tf.keras.optimizers.Adam)

    def test_sgd(self):
        """'sgd' should return an SGD optimizer."""
        opt = _build_optimizer("sgd", self._make_schedule())
        assert isinstance(opt, tf.keras.optimizers.SGD)

    def test_adamw(self):
        """'adamw' should return an AdamW optimizer."""
        opt = _build_optimizer("adamw", self._make_schedule(), weight_decay=1e-4)
        assert isinstance(opt, tf.keras.optimizers.AdamW)

    def test_invalid_raises(self):
        """Invalid optimizer name should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid optimizer"):
            _build_optimizer("rmsprop", self._make_schedule())

    def test_valid_optimizers_constant(self):
        """VALID_OPTIMIZERS should contain expected names."""
        assert set(VALID_OPTIMIZERS) == {"adam", "sgd", "adamw"}
