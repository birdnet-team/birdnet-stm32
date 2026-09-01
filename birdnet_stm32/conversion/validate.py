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


def parity_metrics(
    reference: Callable[[np.ndarray], np.ndarray],
    candidate: Callable[[np.ndarray], np.ndarray],
    rep_data_gen: Callable[[], Iterable[list[np.ndarray]]],
    label: str = "",
) -> dict[str, float]:
    """Compare two prediction functions sample by sample and summarize the gap.

    Tail statistics are included alongside the means so a high average cannot
    hide a handful of badly quantized inputs.

    Args:
        reference: Callable mapping one input batch to the reference output.
        candidate: Callable mapping the same batch to the candidate output.
        rep_data_gen: Callable returning an iterable of ``[input_tensor]``.
        label: Optional prefix for the printed summary lines.

    Returns:
        Distribution statistics for cosine similarity, MSE, MAE, and Pearson
        correlation.
    """
    cos_list, mse_list, mae_list, pcc_list = [], [], [], []

    for sample in rep_data_gen():
        inputs = np.asarray(sample[0], dtype=np.float32)
        a = np.asarray(reference(inputs)).reshape(-1).astype(np.float64)
        b = np.asarray(candidate(inputs)).reshape(-1).astype(np.float64)

        cos_list.append(cosine_similarity(a, b))
        mse_list.append(float(np.mean((a - b) ** 2)))
        mae_list.append(float(np.mean(np.abs(a - b))))
        pcc_list.append(pearson_correlation(a, b))

    prefix = f"{label} " if label else ""

    def _summ(name: str, vals: list[float]):
        if vals:
            print(
                f"{prefix}{name}: mean={np.mean(vals):.6f}  std={np.std(vals):.6f}  "
                f"min={np.min(vals):.6f}  max={np.max(vals):.6f}"
            )

    _summ("cosine", cos_list)
    _summ("mse", mse_list)
    _summ("mae", mae_list)
    _summ("pearson_r", pcc_list)

    def _stats(name: str, values: list[float], empty: float) -> dict[str, float]:
        if not values:
            return {
                f"{name}_mean": empty,
                f"{name}_std": empty,
                f"{name}_min": empty,
                f"{name}_p05": empty,
                f"{name}_median": empty,
                f"{name}_p95": empty,
                f"{name}_max": empty,
            }
        array = np.asarray(values, dtype=np.float64)
        return {
            f"{name}_mean": float(np.mean(array)),
            f"{name}_std": float(np.std(array)),
            f"{name}_min": float(np.min(array)),
            f"{name}_p05": float(np.percentile(array, 5)),
            f"{name}_median": float(np.median(array)),
            f"{name}_p95": float(np.percentile(array, 95)),
            f"{name}_max": float(np.max(array)),
        }

    metrics: dict[str, float] = {"num_samples": float(len(cos_list))}
    metrics.update(_stats("cosine", cos_list, 0.0))
    metrics.update(_stats("mse", mse_list, float("inf")))
    metrics.update(_stats("mae", mae_list, float("inf")))
    metrics.update(_stats("pearson", pcc_list, 0.0))
    return metrics


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
        Distribution statistics for cosine similarity, MSE, MAE, and Pearson
        correlation.  Tail statistics are included so a high mean cannot hide
        badly quantized inputs.
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

    return parity_metrics(
        lambda inputs: keras_model(inputs, training=False).numpy(),
        runner.predict,
        rep_data_gen,
    )
