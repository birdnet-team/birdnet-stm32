"""Validation utilities for comparing Keras vs. TFLite model outputs."""

import numpy as np
import tensorflow as tf


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Cosine similarity between two flattened arrays.

    Args:
        a: Flattened predictions from Keras.
        b: Flattened predictions from TFLite.
        eps: Small constant to avoid division-by-zero.

    Returns:
        Cosine similarity in [-1, 1].
    """
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an < eps or bn < eps:
        return 1.0 if an < eps and bn < eps else 0.0
    return float(np.dot(a, b) / (an * bn))


def pearson_correlation(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Pearson correlation coefficient between two flattened arrays.

    Args:
        a: Flattened predictions from Keras.
        b: Flattened predictions from TFLite.
        eps: Small constant to guard against zero variance.

    Returns:
        Pearson r in [-1, 1].
    """
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < eps:
        return 1.0
    return float(np.dot(a, b) / denom)


def validate_models(keras_model: tf.keras.Model, tflite_model_path: str, rep_data_gen) -> dict[str, float]:
    """Compare Keras vs. TFLite predictions and print summary statistics.

    Runs the TFLite interpreter without delegates to minimize numeric differences.

    Args:
        keras_model: Loaded Keras model.
        tflite_model_path: Path to the converted .tflite model.
        rep_data_gen: Callable returning an iterable of [input_tensor].

    Returns:
        Dict with keys 'cosine_mean', 'mse_mean', 'mae_mean', 'pearson_mean'.
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path, experimental_delegates=None, num_threads=1)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    print(f"TFLite input shape: {in_det['shape']}, output shape: {out_det['shape']}")

    cos_list, mse_list, mae_list, pcc_list = [], [], [], []

    for sample in rep_data_gen():
        yk = keras_model(sample[0], training=False).numpy()
        interpreter.set_tensor(in_det["index"], sample[0].astype(np.float32))
        interpreter.invoke()
        yt = interpreter.get_tensor(out_det["index"])

        a = yk.reshape(-1).astype(np.float64)
        b = yt.reshape(-1).astype(np.float64)

        cos_list.append(cosine_similarity(a, b))
        mse_list.append(float(np.mean((a - b) ** 2)))
        mae_list.append(float(np.mean(np.abs(a - b))))
        pcc_list.append(pearson_correlation(a, b))

    def _summ(name: str, vals: list[float]):
        if vals:
            print(
                f"{name}: mean={np.mean(vals):.6f}  std={np.std(vals):.6f}  min={np.min(vals):.6f}  max={np.max(vals):.6f}"
            )

    _summ("cosine", cos_list)
    _summ("mse", mse_list)
    _summ("mae", mae_list)
    _summ("pearson_r", pcc_list)

    return {
        "cosine_mean": float(np.mean(cos_list)) if cos_list else 0.0,
        "mse_mean": float(np.mean(mse_list)) if mse_list else float("inf"),
        "mae_mean": float(np.mean(mae_list)) if mae_list else float("inf"),
        "pearson_mean": float(np.mean(pcc_list)) if pcc_list else 0.0,
    }
