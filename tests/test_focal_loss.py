"""Unit tests for BinaryFocalLoss."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow required for loss tests")

from birdnet_stm32.training.losses import BinaryFocalLoss


class TestBinaryFocalLoss:
    """Tests for BinaryFocalLoss."""

    def test_gamma_zero_equals_bce(self):
        """gamma=0 should match standard binary crossentropy."""
        y_true = tf.constant([[1.0, 0.0, 1.0]])
        y_pred = tf.constant([[0.9, 0.1, 0.8]])
        focal = BinaryFocalLoss(gamma=0.0)(y_true, y_pred)
        bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
        np.testing.assert_allclose(focal.numpy(), bce.numpy(), atol=1e-5)

    def test_output_is_scalar(self):
        """Loss should return a scalar."""
        y_true = tf.constant([[1.0, 0.0], [0.0, 1.0]])
        y_pred = tf.constant([[0.8, 0.2], [0.3, 0.7]])
        loss = BinaryFocalLoss(gamma=2.0)(y_true, y_pred)
        assert loss.shape == ()

    def test_perfect_prediction_low_loss(self):
        """Perfect predictions should have near-zero loss."""
        y_true = tf.constant([[1.0, 0.0]])
        y_pred = tf.constant([[0.999, 0.001]])
        loss = BinaryFocalLoss(gamma=2.0)(y_true, y_pred)
        assert loss.numpy() < 0.01

    def test_higher_gamma_lower_easy_loss(self):
        """Higher gamma should down-weight easy examples more."""
        y_true = tf.constant([[1.0, 0.0]])
        y_pred = tf.constant([[0.9, 0.1]])
        loss_g0 = BinaryFocalLoss(gamma=0.0)(y_true, y_pred).numpy()
        loss_g2 = BinaryFocalLoss(gamma=2.0)(y_true, y_pred).numpy()
        assert loss_g2 < loss_g0

    def test_get_config_roundtrip(self):
        """get_config should return serializable config."""
        loss = BinaryFocalLoss(gamma=3.0, from_logits=True)
        cfg = loss.get_config()
        assert cfg["gamma"] == 3.0
        assert cfg["from_logits"] is True

    def test_from_logits(self):
        """from_logits mode should produce valid loss."""
        y_true = tf.constant([[1.0, 0.0]])
        y_pred = tf.constant([[2.0, -2.0]])  # logits
        loss = BinaryFocalLoss(gamma=2.0, from_logits=True)(y_true, y_pred)
        assert np.isfinite(loss.numpy())
        assert loss.numpy() > 0
