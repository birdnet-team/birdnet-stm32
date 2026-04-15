"""Integration test: quantized vs float model cosine similarity."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow required for quantization sim tests")

from birdnet_stm32.conversion.validate import cosine_similarity


@pytest.mark.integration
@pytest.mark.slow
class TestQuantizationSimilarity:
    """Verify that PTQ preserves model behavior above a minimum threshold."""

    def test_cosine_sim_above_threshold(self, tmp_path):
        """Float vs INT8 TFLite cosine similarity should exceed 0.90 on tiny model."""
        # Build a small model
        inp = tf.keras.Input(shape=(16, 16, 1))
        x = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu")(inp)
        x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(4, activation="softmax")(x)
        model = tf.keras.Model(inp, x)

        # Train briefly on random data so weights are non-trivial
        rng = np.random.default_rng(42)
        X = rng.standard_normal((32, 16, 16, 1)).astype(np.float32)
        y = np.eye(4, dtype=np.float32)[rng.integers(0, 4, size=32)]
        model.compile(optimizer="adam", loss="categorical_crossentropy")
        model.fit(X, y, epochs=3, batch_size=8, verbose=0)

        # Convert with PTQ
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        def rep_gen():
            for i in range(20):
                yield [X[i : i + 1]]

        converter.representative_dataset = rep_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        tflite_model = converter.convert()

        tflite_path = str(tmp_path / "model_q.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        # Run both models on test data
        interp = tf.lite.Interpreter(model_path=tflite_path)
        interp.allocate_tensors()
        in_det = interp.get_input_details()[0]
        out_det = interp.get_output_details()[0]

        cos_values = []
        X_test = rng.standard_normal((10, 16, 16, 1)).astype(np.float32)
        for i in range(10):
            x_i = X_test[i : i + 1]
            y_keras = model(x_i, training=False).numpy().ravel()

            interp.set_tensor(in_det["index"], x_i)
            interp.invoke()
            y_tflite = interp.get_tensor(out_det["index"]).ravel()

            cos_values.append(cosine_similarity(y_keras, y_tflite))

        mean_cos = np.mean(cos_values)
        assert mean_cos > 0.90, f"Mean cosine similarity {mean_cos:.4f} below 0.90 threshold"
