"""Validation utilities for comparing Keras vs. TFLite model outputs."""

from collections.abc import Callable, Iterable

import numpy as np
import tensorflow as tf


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """Cosine similarity between two flattened arrays.

    When both vectors have negligible magnitude (e.g., background/noise class
    predictions near zero), returns 1.0 since both models agree on "no detection".

    Args:
        a: Flattened predictions from Keras.
        b: Flattened predictions from TFLite.
        eps: Small constant to avoid division-by-zero.

    Returns:
        Cosine similarity in [-1, 1].
    """
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    # Both near-zero: models agree on no-detection — treat as perfect match
    if an < eps and bn < eps:
        return 1.0
    # One near-zero but not both: genuine disagreement
    if an < eps or bn < eps:
        return 0.0
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


def validate_models(
    keras_model: tf.keras.Model,
    tflite_model_path: str,
    rep_data_gen: Callable[[], Iterable[list[np.ndarray]]],
) -> dict[str, float]:
    """Compare Keras vs. TFLite predictions and print summary statistics.

    Runs the TFLite interpreter without delegates to minimize numeric differences.

    Args:
        keras_model: Loaded Keras model.
        tflite_model_path: Path to the converted .tflite model.
        rep_data_gen: Callable returning an iterable of [input_tensor].

    Returns:
        Dict with keys 'cosine_mean', 'mse_mean', 'mae_mean', 'pearson_mean'.
    """
    # Go through the shared runner so INT8-I/O models are quantized/dequantized
    # exactly the way evaluation and deployment will do it.
    from birdnet_stm32.models.runners import TFLiteRunner

    runner = TFLiteRunner(tflite_model_path)
    in_det = runner.interpreter.get_input_details()[0]
    out_det = runner.interpreter.get_output_details()[0]

    print(
        f"TFLite input shape: {in_det['shape']} ({np.dtype(in_det['dtype']).name}), "
        f"output shape: {out_det['shape']} ({np.dtype(out_det['dtype']).name})"
    )

    cos_list, mse_list, mae_list, pcc_list = [], [], [], []

    for sample in rep_data_gen():
        yk = keras_model(sample[0], training=False).numpy()
        yt = runner.predict(np.asarray(sample[0], dtype=np.float32))

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
