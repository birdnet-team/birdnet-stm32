"""Tests for quantization-aware training (QAT) module."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow required for QAT tests")

from birdnet_stm32.training.qat import (
    FakeQuantActivation,
    _channel_axis,
    _DistilledQATModel,
    _is_activation_boundary,
    _is_quantizable,
    build_qat_model,
    calibrate_activation_ranges,
    fake_quantize_weights,
    freeze_batch_norm,
    sync_frontend_weights,
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

    def test_audio_frontend_kernel_is_quantized(self):
        """Frontend kernels must see the same INT8 noise as deployment."""
        layer = tf.keras.layers.Conv2D(8, 3, name="audio_frontend_conv")
        layer.build((None, 16, 16, 1))
        assert _is_quantizable(layer) is True

    def test_bn_followed_by_inference_dropout_and_relu_is_fused(self):
        inputs = tf.keras.Input(shape=(4, 4, 2))
        x = tf.keras.layers.BatchNormalization(name="bn")(inputs)
        x = tf.keras.layers.SpatialDropout2D(0.1, name="drop")(x)
        outputs = tf.keras.layers.ReLU(max_value=6, name="relu")(x)
        model = tf.keras.Model(inputs, outputs)

        assert _is_activation_boundary(model.get_layer("bn")) is False

    def test_bn_followed_by_inference_dropout_and_add_stays_boundary(self):
        inputs = tf.keras.Input(shape=(4, 4, 2))
        x = tf.keras.layers.BatchNormalization(name="bn")(inputs)
        x = tf.keras.layers.SpatialDropout2D(0.1, name="drop")(x)
        outputs = tf.keras.layers.Add(name="add")([inputs, x])
        model = tf.keras.Model(inputs, outputs)

        assert _is_activation_boundary(model.get_layer("bn")) is True


class TestActivationQAT:
    """Exercise the activation side of the QAT graph."""

    def test_fake_quant_activation_uses_int8_grid(self):
        layer = FakeQuantActivation(-1.0, 2.0)
        values = tf.linspace(-1.0, 2.0, 1000)
        quantized = layer(values).numpy()
        assert np.unique(quantized).size <= 256
        assert quantized.min() >= -1.01
        assert quantized.max() <= 2.01

    def test_calibration_and_shared_weight_graph(self):
        inputs = tf.keras.Input(shape=(4, 4, 1), name="input")
        x = tf.keras.layers.Conv2D(4, 3, padding="same", use_bias=False, name="conv")(inputs)
        x = tf.keras.layers.BatchNormalization(name="bn")(x)
        x = tf.keras.layers.ReLU(max_value=6, name="relu")(x)
        x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
        outputs = tf.keras.layers.Dense(2, activation="sigmoid", name="pred")(x)
        deployment = tf.keras.Model(inputs, outputs)

        rng = np.random.default_rng(3)
        samples = rng.normal(size=(4, 4, 4, 1)).astype(np.float32)
        labels = np.zeros((4, 2), dtype=np.float32)
        dataset = tf.data.Dataset.from_tensor_slices((samples, labels)).batch(2)
        ranges = calibrate_activation_ranges(deployment, dataset, max_samples=4)
        qat_model = build_qat_model(deployment, ranges)

        assert {"__input__", "relu", "gap", "pred"} <= ranges.keys()
        assert "bn" not in ranges  # folded into Conv + ReLU by TFLite
        inner_model = next(layer for layer in qat_model.layers if isinstance(layer, tf.keras.Model))
        assert inner_model.get_layer("conv_quantized_kernel").target is deployment.get_layer("conv")
        assert qat_model.get_layer("input_fake_quant") is not None
        assert any(layer.name.endswith("_fake_quant") for layer in inner_model.layers)

    def test_calibration_accepts_converter_style_single_input_lists(self):
        """QAT must consume the exact tensor iterator used by conversion."""
        inputs = tf.keras.Input(shape=(4,), name="input")
        outputs = tf.keras.layers.Dense(2, activation="sigmoid", name="pred")(inputs)
        deployment = tf.keras.Model(inputs, outputs)
        samples = np.random.default_rng(17).normal(size=(4, 4)).astype(np.float32)

        ranges = calibrate_activation_ranges(
            deployment,
            ([sample[None]] for sample in samples),
            max_samples=4,
        )

        assert ranges["__input__"][0] <= float(samples.min())
        assert ranges["__input__"][1] >= float(samples.max())

    def test_raw_frontend_is_quantized_and_synced_to_clean_model(self):
        """QAT must cover opaque frontend kernels without polluting deployment."""
        from birdnet_stm32.models.frontend import AudioFrontendLayer

        inputs = tf.keras.Input(shape=(2000, 1), name="raw_audio_input")
        x = AudioFrontendLayer(
            mode="raw",
            mel_bins=8,
            spec_width=8,
            sample_rate=8000,
            chunk_duration=0.25,
            mag_scale="pwl",
            name="audio_frontend",
        )(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
        outputs = tf.keras.layers.Dense(2, activation="sigmoid", name="pred")(x)
        deployment = tf.keras.Model(inputs, outputs)

        rng = np.random.default_rng(11)
        samples = rng.uniform(-1.0, 1.0, size=(4, 2000, 1)).astype(np.float32)
        dataset = tf.data.Dataset.from_tensor_slices((samples, np.zeros((4, 2), np.float32))).batch(2)
        ranges = calibrate_activation_ranges(deployment, dataset, max_samples=4)
        qat_model = build_qat_model(deployment, ranges)

        assert "audio_frontend_fb_re" in ranges
        assert "audio_frontend_mag_pwl_add_3" in ranges
        clean_frontend = deployment.get_layer("audio_frontend")
        qat_frontend = next(
            layer
            for layer in qat_model._flatten_layers(include_self=False, recursive=True)  # noqa: SLF001
            if layer.__class__.__name__ == "AudioFrontendLayer"
        )
        assert qat_frontend is not clean_frontend
        assert qat_frontend._quantization_hook is not None  # noqa: SLF001
        assert clean_frontend._quantization_hook is None  # noqa: SLF001

        updated = qat_frontend.get_weights()
        updated[0] = updated[0] + 0.01
        qat_frontend.set_weights(updated)
        sync_frontend_weights(qat_model, deployment)
        for qat_weight, clean_weight in zip(
            qat_frontend.get_weights(),
            clean_frontend.get_weights(),
            strict=True,
        ):
            np.testing.assert_array_equal(qat_weight, clean_weight)

    def test_distillation_loss_is_finite_and_reported(self):
        """The QAT objective must constrain probabilities beyond hard labels."""
        inputs = tf.keras.Input(shape=(4,), name="input")
        outputs = tf.keras.layers.Dense(2, activation="sigmoid", name="pred")(inputs)
        deployment = tf.keras.Model(inputs, outputs)
        teacher = tf.keras.models.clone_model(deployment)
        teacher.set_weights(deployment.get_weights())
        samples = np.random.default_rng(5).normal(size=(8, 4)).astype(np.float32)
        targets = np.zeros((8, 2), np.float32)
        dataset = tf.data.Dataset.from_tensor_slices((samples, targets)).batch(4)
        ranges = calibrate_activation_ranges(deployment, dataset, max_samples=8)
        student = build_qat_model(deployment, ranges)
        distilled = _DistilledQATModel(student, teacher)
        distilled.compile(optimizer="adam", loss="binary_crossentropy")

        metrics = distilled.train_on_batch(samples, targets, return_dict=True)
        assert np.isfinite(metrics["loss"])
        assert np.isfinite(metrics["distillation_kl"])
        assert np.isfinite(metrics["distillation_cosine_loss"])
        assert np.isfinite(metrics["distillation_cosine_tail_loss"])
        assert metrics["distillation_kl"] >= -1e-6
        assert 0 <= metrics["distillation_cosine_loss"] <= 2
        assert metrics["distillation_cosine_loss"] <= metrics["distillation_cosine_tail_loss"] <= 2


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


class TestQuantizationGridMatchesTFLite:
    """The simulated grid must be the one TFLite uses for kernels."""

    def test_grid_is_symmetric_around_zero(self):
        """Quantized values must be integer multiples of a scale, with no offset.

        TFLite quantizes kernels symmetrically (zero_point=0). An asymmetric
        min/max grid would train against a quantizer conversion never applies.
        """
        w = np.linspace(-0.2, 1.0, 512).astype(np.float32)  # deliberately off-centre
        fq = fake_quantize_weights(w, num_bits=8, per_channel=False)

        scale = np.max(np.abs(w)) / 127.0
        steps = fq / scale
        np.testing.assert_allclose(steps, np.round(steps), atol=1e-3)
        # Exact zero must survive: a nonzero zero-point would shift it.
        assert fake_quantize_weights(np.zeros((4,), dtype=np.float32), per_channel=False).tolist() == [0.0] * 4

    def test_per_channel_uses_per_channel_scales(self):
        """Channels with very different magnitudes must not share one scale."""
        w = np.zeros((1, 1, 1, 2), dtype=np.float32)
        w[..., 0] = 1.0  # large channel
        w[..., 1] = 0.001  # small channel

        per_ch = fake_quantize_weights(w, per_channel=True, channel_axis=-1)
        per_tensor = fake_quantize_weights(w, per_channel=False)

        # Under one shared scale the small channel collapses toward zero;
        # with its own scale it is represented exactly.
        assert per_ch[..., 1] == pytest.approx(0.001, rel=1e-3)
        assert per_tensor[..., 1] != pytest.approx(0.001, rel=1e-3)

    def test_depthwise_axis_is_respected(self):
        """Depthwise kernels quantize along axis -2, not the trailing axis."""
        w = np.zeros((1, 1, 2, 1), dtype=np.float32)
        w[:, :, 0, :] = 1.0
        w[:, :, 1, :] = 0.001
        fq = fake_quantize_weights(w, per_channel=True, channel_axis=-2)
        assert fq[:, :, 1, :].item() == pytest.approx(0.001, rel=1e-3)
