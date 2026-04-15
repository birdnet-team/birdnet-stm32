"""Unit tests for model inference runners."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow required for runner tests")

from birdnet_stm32.models.runners import KerasRunner, TFLiteRunner, load_model_runner


@pytest.fixture
def tiny_keras_model():
    """Build a minimal Keras model for testing."""
    inp = tf.keras.Input(shape=(8, 8, 1))
    x = tf.keras.layers.Flatten()(inp)
    x = tf.keras.layers.Dense(4, activation="softmax")(x)
    return tf.keras.Model(inp, x)


@pytest.fixture
def tiny_tflite_path(tiny_keras_model, tmp_path):
    """Convert the tiny model to TFLite and return its path."""
    converter = tf.lite.TFLiteConverter.from_keras_model(tiny_keras_model)
    tflite_model = converter.convert()
    path = str(tmp_path / "tiny.tflite")
    with open(path, "wb") as f:
        f.write(tflite_model)
    return path


class TestKerasRunner:
    """Tests for KerasRunner."""

    def test_predict_shape(self, tiny_keras_model):
        """Predict should return (B, num_classes)."""
        runner = KerasRunner(tiny_keras_model)
        x = np.random.rand(3, 8, 8, 1).astype(np.float32)
        y = runner.predict(x)
        assert y.shape == (3, 4)

    def test_predict_dtype(self, tiny_keras_model):
        """Output should be float32."""
        runner = KerasRunner(tiny_keras_model)
        x = np.random.rand(1, 8, 8, 1).astype(np.float32)
        y = runner.predict(x)
        assert y.dtype == np.float32

    def test_softmax_sums_to_one(self, tiny_keras_model):
        """Softmax outputs should sum to ~1 per sample."""
        runner = KerasRunner(tiny_keras_model)
        x = np.random.rand(2, 8, 8, 1).astype(np.float32)
        y = runner.predict(x)
        np.testing.assert_allclose(y.sum(axis=1), 1.0, atol=1e-5)


class TestTFLiteRunner:
    """Tests for TFLiteRunner."""

    def test_predict_shape(self, tiny_tflite_path):
        """Predict should return (B, num_classes)."""
        runner = TFLiteRunner(tiny_tflite_path)
        x = np.random.rand(2, 8, 8, 1).astype(np.float32)
        y = runner.predict(x)
        assert y.shape == (2, 4)

    def test_predict_single_sample(self, tiny_tflite_path):
        """Single-sample batch should work."""
        runner = TFLiteRunner(tiny_tflite_path)
        x = np.random.rand(1, 8, 8, 1).astype(np.float32)
        y = runner.predict(x)
        assert y.shape == (1, 4)

    def test_dynamic_batch_resize(self, tiny_tflite_path):
        """Runner should handle different batch sizes across calls."""
        runner = TFLiteRunner(tiny_tflite_path)
        x1 = np.random.rand(1, 8, 8, 1).astype(np.float32)
        x2 = np.random.rand(4, 8, 8, 1).astype(np.float32)
        y1 = runner.predict(x1)
        y2 = runner.predict(x2)
        assert y1.shape == (1, 4)
        assert y2.shape == (4, 4)


class TestLoadModelRunner:
    """Tests for load_model_runner dispatcher."""

    def test_loads_tflite(self, tiny_tflite_path):
        """Should return a TFLiteRunner for .tflite files."""
        runner = load_model_runner(tiny_tflite_path)
        assert isinstance(runner, TFLiteRunner)

    def test_loads_keras(self, tiny_keras_model, tmp_path):
        """Should return a KerasRunner for .keras files."""
        path = str(tmp_path / "model.keras")
        tiny_keras_model.save(path)
        runner = load_model_runner(path)
        assert isinstance(runner, KerasRunner)
