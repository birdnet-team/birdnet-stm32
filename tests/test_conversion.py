"""Unit tests for TFLite quantization conversion."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow required for conversion tests")

from birdnet_stm32.cli.convert import _export_and_validate_onnx, _manifest_record
from birdnet_stm32.conversion.quantize import stratified_sample_paths
from birdnet_stm32.conversion.validate import cosine_similarity, pearson_correlation


class TestCosineSimilarity:
    """Tests for cosine_similarity."""

    def test_identical(self):
        """Identical vectors should have cosine similarity 1.0."""
        a = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(a, a) == pytest.approx(1.0, abs=1e-6)

    def test_opposite(self):
        """Negated vectors should have cosine similarity -1.0."""
        a = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(a, -a) == pytest.approx(-1.0, abs=1e-6)

    def test_orthogonal(self):
        """Orthogonal vectors should have cosine similarity ~0."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_both_zero(self):
        """Two zero vectors should return 1.0 (degenerate case)."""
        a = np.zeros(5)
        assert cosine_similarity(a, a) == 1.0

    def test_one_zero(self):
        """One zero vector should return 0.0."""
        a = np.zeros(5)
        b = np.ones(5)
        assert cosine_similarity(a, b) == 0.0


class TestPearsonCorrelation:
    """Tests for pearson_correlation."""

    def test_perfect_positive(self):
        """Perfectly correlated should return ~1.0."""
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([2.0, 4.0, 6.0, 8.0])
        assert pearson_correlation(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_perfect_negative(self):
        """Perfectly anti-correlated should return ~-1.0."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([3.0, 2.0, 1.0])
        assert pearson_correlation(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_constant(self):
        """Constant arrays should return 1.0 (both zero variance)."""
        a = np.ones(5) * 3.0
        b = np.ones(5) * 7.0
        assert pearson_correlation(a, b) == 1.0


class TestCalibrationSampling:
    """Calibration selection is exact, balanced, deterministic, and disjoint."""

    def test_fills_requested_count_and_excludes_paths(self):
        paths = [
            f"/data/class_{class_idx}/sample_{sample_idx}.wav" for class_idx in range(3) for sample_idx in range(7)
        ]
        calibration = stratified_sample_paths(paths, 10, seed=42)
        validation = stratified_sample_paths(paths, 8, seed=43, exclude=set(calibration))

        assert len(calibration) == 10
        assert len(validation) == 8
        assert set(calibration).isdisjoint(validation)
        assert calibration == stratified_sample_paths(paths, 10, seed=42)

    def test_manifest_record_is_location_independent(self):
        """Manifest identity uses stable relative paths and class counts."""
        first = _manifest_record(["/one/a/x.wav", "/one/b/y.wav"], "/one")
        second = _manifest_record(["/two/a/x.wav", "/two/b/y.wav"], "/two")
        assert first == second
        assert first["count"] == 2
        assert first["class_counts"] == {"a": 1, "b": 1}
        assert len(first["sha256"]) == 64


class TestQuantizationSmoke:
    """Smoke test: build a tiny model, convert to TFLite, verify it runs."""

    @pytest.mark.slow
    def test_tiny_model_quantizes(self, tmp_path):
        """A minimal Keras model should convert to TFLite without error."""
        # Build tiny model
        inp = tf.keras.Input(shape=(8, 8, 1))
        x = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu")(inp)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(3, activation="softmax")(x)
        model = tf.keras.Model(inp, x)

        # Convert with PTQ
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        def rep_gen():
            for _ in range(10):
                yield [np.random.rand(1, 8, 8, 1).astype(np.float32)]

        converter.representative_dataset = rep_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        tflite_model = converter.convert()

        # Save and load
        path = str(tmp_path / "tiny_q.tflite")
        with open(path, "wb") as f:
            f.write(tflite_model)

        # Run inference
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        in_det = interp.get_input_details()[0]
        out_det = interp.get_output_details()[0]

        x_test = np.random.rand(1, 8, 8, 1).astype(np.float32)
        interp.set_tensor(in_det["index"], x_test)
        interp.invoke()
        y = interp.get_tensor(out_det["index"])

        assert y.shape == (1, 3)
        np.testing.assert_allclose(y.sum(), 1.0, atol=0.1)

        # Compare Keras vs TFLite
        y_keras = model(x_test, training=False).numpy()
        cos = cosine_similarity(y_keras.ravel(), y.ravel())
        assert cos > 0.8, f"Cosine similarity too low: {cos}"


class TestOnnxExport:
    """ONNX is only advertised after checker and runtime parity pass."""

    def test_export_checker_and_runtime(self, tmp_path):
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
        pytest.importorskip("tf2onnx")
        inputs = tf.keras.Input(shape=(4,), name="features")
        outputs = tf.keras.layers.Dense(2, activation="sigmoid")(inputs)
        model = tf.keras.Model(inputs, outputs)
        sample = np.ones((1, 4), dtype=np.float32)
        model(sample)

        def validation_gen():
            yield [sample]

        output_path = str(tmp_path / "tiny.onnx")
        report = _export_and_validate_onnx(model, output_path, validation_gen)
        assert report["checker_passed"] is True
        assert report["cosine_min"] >= 0.9999
        assert report["max_abs_error"] <= 1e-4
