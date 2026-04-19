"""Tests for quantization-aware training (QAT) module."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow required for QAT tests")

from birdnet_stm32.training.qat import (
    QATCallback,
    _channel_axis,
    _is_quantizable,
    fake_quantize_weights,
    freeze_batch_norm,
)

# ---------------------------------------------------------------------------
# fake_quantize_weights
# ---------------------------------------------------------------------------


class TestFakeQuantizeWeights:
    """Verify fake-quantize simulates INT8 rounding correctly."""

    def test_per_tensor_roundtrip(self):
        """Fake-quantized values should be close to originals for 8-bit."""
        w = np.linspace(-1.0, 1.0, 256).astype(np.float32)
        fq = fake_quantize_weights(w, num_bits=8, per_channel=False)
        assert fq.shape == w.shape
        assert fq.dtype == np.float32
        np.testing.assert_allclose(fq, w, atol=0.01)

    def test_per_channel_shape_preserved(self):
        """Per-channel fake-quant should preserve tensor shape."""
        w = np.random.default_rng(42).standard_normal((3, 3, 8, 16)).astype(np.float32)
        fq = fake_quantize_weights(w, num_bits=8, per_channel=True, channel_axis=-1)
        assert fq.shape == w.shape
        assert fq.dtype == np.float32
        # Should differ from original (quantization noise)
        assert not np.array_equal(fq, w)
        # MSE should be small for 8-bit
        assert np.mean((fq - w) ** 2) < 0.01

    def test_per_channel_depthwise(self):
        """Depthwise kernel [H, W, C_in, 1] quantized along axis -2."""
        w = np.random.default_rng(7).standard_normal((3, 3, 16, 1)).astype(np.float32)
        fq = fake_quantize_weights(w, num_bits=8, per_channel=True, channel_axis=-2)
        assert fq.shape == w.shape
        assert np.mean((fq - w) ** 2) < 0.01

    def test_constant_weight_unchanged(self):
        """Constant weight array should survive quantization unchanged."""
        w = np.full((4, 4), 0.5, dtype=np.float32)
        fq = fake_quantize_weights(w, num_bits=8, per_channel=False)
        np.testing.assert_allclose(fq, w, atol=1e-6)

    def test_zero_weight_unchanged(self):
        """All-zero weights should remain zero."""
        w = np.zeros((3, 3, 1, 8), dtype=np.float32)
        fq = fake_quantize_weights(w, num_bits=8, per_channel=True, channel_axis=-1)
        np.testing.assert_allclose(fq, 0.0, atol=1e-6)

    def test_1d_per_tensor_only(self):
        """1-D weight (bias-like) falls back to per-tensor."""
        w = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        fq = fake_quantize_weights(w, num_bits=8, per_channel=True, channel_axis=-1)
        assert fq.shape == w.shape
        np.testing.assert_allclose(fq, w, atol=0.01)

    def test_low_bit_quantization(self):
        """4-bit quantization should introduce more noise than 8-bit."""
        rng = np.random.default_rng(99)
        w = rng.standard_normal((16, 16)).astype(np.float32)
        fq4 = fake_quantize_weights(w, num_bits=4, per_channel=False)
        fq8 = fake_quantize_weights(w, num_bits=8, per_channel=False)
        mse4 = np.mean((fq4 - w) ** 2)
        mse8 = np.mean((fq8 - w) ** 2)
        assert mse4 > mse8


# ---------------------------------------------------------------------------
# _channel_axis / _is_quantizable
# ---------------------------------------------------------------------------


class TestLayerFiltering:
    """Verify layer type detection and filtering."""

    def test_channel_axis_conv2d(self):
        layer = tf.keras.layers.Conv2D(8, 3, name="stem_conv")
        layer.build((None, 16, 16, 1))
        assert _channel_axis(layer) == -1

    def test_channel_axis_depthwise(self):
        layer = tf.keras.layers.DepthwiseConv2D(3, name="dw")
        layer.build((None, 16, 16, 8))
        assert _channel_axis(layer) == -2

    def test_channel_axis_dense(self):
        layer = tf.keras.layers.Dense(10, name="pred")
        layer.build((None, 64))
        assert _channel_axis(layer) == -1

    def test_is_quantizable_conv(self):
        layer = tf.keras.layers.Conv2D(8, 3, name="stem_conv")
        layer.build((None, 16, 16, 1))
        assert _is_quantizable(layer) is True

    def test_is_quantizable_bn_excluded(self):
        layer = tf.keras.layers.BatchNormalization(name="bn")
        layer.build((None, 16, 16, 8))
        assert _is_quantizable(layer) is False

    def test_is_quantizable_relu_excluded(self):
        layer = tf.keras.layers.ReLU(name="relu")
        assert _is_quantizable(layer) is False

    def test_is_quantizable_audio_frontend_skipped(self):
        """Layers named 'audio_frontend*' should be skipped."""
        layer = tf.keras.layers.Conv2D(8, 3, name="audio_frontend_conv")
        layer.build((None, 16, 16, 1))
        assert _is_quantizable(layer) is False


# ---------------------------------------------------------------------------
# freeze_batch_norm
# ---------------------------------------------------------------------------


class TestFreezeBatchNorm:
    """Verify BatchNorm freezing."""

    def test_freeze_bn_layers(self):
        inp = tf.keras.Input(shape=(8, 8, 1))
        x = tf.keras.layers.Conv2D(8, 3, padding="same", name="conv")(inp)
        x = tf.keras.layers.BatchNormalization(name="bn1")(x)
        x = tf.keras.layers.BatchNormalization(name="bn2")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(2)(x)
        model = tf.keras.Model(inp, x)

        n_frozen = freeze_batch_norm(model)
        assert n_frozen == 2

        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                assert layer.trainable is False

    def test_freeze_bn_no_bn_layers(self):
        inp = tf.keras.Input(shape=(4,))
        x = tf.keras.layers.Dense(2)(inp)
        model = tf.keras.Model(inp, x)
        assert freeze_batch_norm(model) == 0


# ---------------------------------------------------------------------------
# QATCallback
# ---------------------------------------------------------------------------


class TestQATCallback:
    """Test shadow-weight QAT callback mechanics."""

    @pytest.fixture
    def tiny_model(self):
        """Build a minimal model with Conv2D, BN, Dense."""
        inp = tf.keras.Input(shape=(4, 4, 1))
        x = tf.keras.layers.Conv2D(8, 3, padding="same", use_bias=False, name="conv")(inp)
        x = tf.keras.layers.BatchNormalization(name="bn")(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(3, activation="softmax", name="pred")(x)
        return tf.keras.Model(inp, x)

    @pytest.fixture
    def tiny_data(self):
        """Generate random training data for the tiny model."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((8, 4, 4, 1)).astype(np.float32)
        y = np.eye(3, dtype=np.float32)[rng.integers(0, 3, size=8)]
        return X, y

    def test_callback_identifies_layers(self, tiny_model, tiny_data):
        """QATCallback should identify Conv2D and Dense as quantizable."""
        X, y = tiny_data
        tiny_model.compile(optimizer="adam", loss="categorical_crossentropy")
        cb = QATCallback(num_bits=8, per_channel=True)
        tiny_model.fit(X, y, epochs=1, batch_size=8, callbacks=[cb], verbose=0)
        # Should have found conv and pred layers
        assert len(cb._qat_layers) == 2
        names = {lyr.name for lyr in cb._qat_layers}
        assert "conv" in names
        assert "pred" in names

    def test_callback_weights_change(self, tiny_model, tiny_data):
        """Weights should be updated after QAT training."""
        X, y = tiny_data
        tiny_model.compile(optimizer="adam", loss="categorical_crossentropy")
        cb = QATCallback(num_bits=8, per_channel=True)

        w_before = [w.numpy().copy() for w in tiny_model.trainable_weights]
        tiny_model.fit(X, y, epochs=1, batch_size=8, callbacks=[cb], verbose=0)
        w_after = [w.numpy() for w in tiny_model.trainable_weights]

        # At least some weights should have changed
        changed = any(not np.array_equal(a, b) for a, b in zip(w_before, w_after, strict=False))
        assert changed

    def test_callback_preserves_fp_precision(self, tiny_model, tiny_data):
        """After training, weights should be full-precision (not quantized grid)."""
        X, y = tiny_data
        tiny_model.compile(optimizer="adam", loss="categorical_crossentropy")
        cb = QATCallback(num_bits=8, per_channel=True)
        tiny_model.fit(X, y, epochs=2, batch_size=4, callbacks=[cb], verbose=0)

        # Get the conv kernel
        conv = tiny_model.get_layer("conv")
        kernel = conv.kernel.numpy()
        # Fake-quantize it
        fq = fake_quantize_weights(kernel, num_bits=8, per_channel=True, channel_axis=-1)
        # The stored kernel should NOT exactly equal its fake-quantized version
        # (it should retain FP precision from shadow weights)
        assert not np.array_equal(kernel, fq)

    def test_callback_no_bn_quantized(self, tiny_model, tiny_data):
        """BatchNorm layers should not appear in QAT layers."""
        X, y = tiny_data
        tiny_model.compile(optimizer="adam", loss="categorical_crossentropy")
        cb = QATCallback()
        tiny_model.fit(X, y, epochs=1, batch_size=8, callbacks=[cb], verbose=0)
        for layer in cb._qat_layers:
            assert not isinstance(layer, tf.keras.layers.BatchNormalization)

    def test_callback_skips_bias_quantization(self, tiny_model, tiny_data):
        """Biases should not be tracked for fake-quantization."""
        X, y = tiny_data
        tiny_model.compile(optimizer="adam", loss="categorical_crossentropy")
        cb = QATCallback()

        # Manually trigger on_train_begin to populate _qat_layers
        cb.set_model(tiny_model)
        cb.on_train_begin()
        # Simulate one batch
        cb.on_train_batch_begin(0)
        # Check tracked variables — none should be biases
        for var, _, _ in cb._tracked:
            assert "bias" not in var.name
