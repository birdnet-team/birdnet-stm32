"""Evaluation metrics and per-file inference pipeline."""

import math
import os
import resource
import time

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from tqdm import tqdm

from birdnet_stm32.audio.io import load_audio_file
from birdnet_stm32.audio.spectrogram import get_spectrogram_from_audio
from birdnet_stm32.evaluation.pooling import pool_scores
from birdnet_stm32.models.frontend import normalize_frontend_name


def make_chunks_for_file(
    path: str,
    cfg: dict,
    frontend: str,
    mag_scale: str,
    n_fft: int,
    chunk_overlap: float,
) -> list[np.ndarray]:
    """Build a list of model-ready inputs from one audio file by chunking.

    Args:
        path: Audio file path.
        cfg: Training config dict (sample_rate, chunk_duration, num_mels, spec_width).
        frontend: 'librosa'|'hybrid'|'raw'.
        mag_scale: Magnitude scaling for precomputed paths.
        n_fft: FFT length.
        chunk_overlap: Overlap fraction (seconds) between consecutive chunks.

    Returns:
        List of per-chunk inputs ready for model.predict.
    """
    sr = int(cfg["sample_rate"])
    cd = float(cfg["chunk_duration"])
    num_mels = int(cfg["num_mels"])
    spec_width = int(cfg["spec_width"])

    chunks = load_audio_file(
        path, sample_rate=sr, max_duration=60, chunk_duration=cd, random_offset=False, chunk_overlap=chunk_overlap
    )

    out: list[np.ndarray] = []
    if frontend == "librosa":
        for ch in chunks:
            S = get_spectrogram_from_audio(
                ch, sample_rate=sr, n_fft=n_fft, mel_bins=num_mels, spec_width=spec_width, mag_scale=mag_scale
            )
            out.append(S[:, :, None].astype(np.float32))
    elif frontend == "hybrid":
        fft_bins = n_fft // 2 + 1
        for ch in chunks:
            S = get_spectrogram_from_audio(ch, sample_rate=sr, n_fft=n_fft, mel_bins=-1, spec_width=spec_width)
            if S.shape[0] != fft_bins:
                S = S[:fft_bins, :spec_width]
            out.append(S[:, :, None].astype(np.float32))
    elif frontend == "raw":
        chunk_len = int(cfg["chunk_duration"] * cfg["sample_rate"])
        for ch in chunks:
            x = ch[:chunk_len]
            if x.shape[0] < chunk_len:
                x = np.pad(x, (0, chunk_len - x.shape[0]))
            x = x / (np.max(np.abs(x)) + 1e-6)
            out.append(x[:, None].astype(np.float32))
    else:
        raise ValueError(f"Invalid audio_frontend: {frontend}")
    return out


def evaluate(
    model_runner,
    files: list[str],
    classes: list[str],
    cfg: dict,
    pooling: str = "average",
    batch_size: int = 64,
    overlap: float = 0.0,
    mep_beta: float = 10.0,
    measure_latency: bool = False,
    profile_memory: bool = False,
) -> tuple[dict, list[dict], np.ndarray, np.ndarray]:
    """Run inference per chunk, pool to file-level, and compute metrics.

    Args:
        model_runner: Runner exposing predict(x_batch).
        files: List of file paths to evaluate.
        classes: Ordered class names.
        cfg: Training config dict.
        pooling: 'avg'|'max'|'lme' pooling method.
        batch_size: Batch size for chunk inference.
        overlap: Overlap in seconds for chunking.
        mep_beta: Temperature for LME pooling.
        measure_latency: If True, measure per-chunk inference latency.
        profile_memory: If True, report peak RSS during inference.

    Returns:
        Tuple of (metrics dict, per_file list, y_true array, y_scores array).
    """
    frontend = normalize_frontend_name(cfg["audio_frontend"])
    mag_scale = cfg.get("mag_scale", "none")
    n_fft = int(cfg["fft_length"])
    num_classes = len(classes)

    y_true: list[np.ndarray] = []
    y_scores: list[np.ndarray] = []
    per_file: list[dict] = []
    chunk_latencies_ms: list[float] = []
    total_chunks = 0

    rss_before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if profile_memory else 0

    for path in tqdm(files, total=len(files), desc="Evaluating", unit="file"):
        label_name = os.path.basename(os.path.dirname(path))
        if label_name not in classes:
            continue
        target = np.zeros((num_classes,), dtype=np.float32)
        target[classes.index(label_name)] = 1.0

        chunks = make_chunks_for_file(path, cfg, frontend, mag_scale, n_fft, overlap)
        if len(chunks) == 0:
            continue

        preds: list[np.ndarray] = []
        for i in range(0, len(chunks), batch_size):
            batch = np.stack(chunks[i : i + batch_size], axis=0)
            if measure_latency:
                t0 = time.perf_counter()
                p = model_runner.predict(batch)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                per_sample_ms = elapsed_ms / batch.shape[0]
                chunk_latencies_ms.extend([per_sample_ms] * batch.shape[0])
            else:
                p = model_runner.predict(batch)
            preds.append(p)
            total_chunks += batch.shape[0]
        chunk_scores = np.concatenate(preds, axis=0)

        pooled = pool_scores(chunk_scores, method=pooling, beta=mep_beta)

        y_true.append(target)
        y_scores.append(pooled)
        per_file.append({"file": path, "label": label_name, "scores": pooled.tolist()})

    if len(y_true) == 0:
        raise RuntimeError("No valid test samples found for the provided class set.")

    y_true_arr = np.asarray(y_true, dtype=np.float32)
    y_scores_arr = np.asarray(y_scores, dtype=np.float32)

    metrics: dict = {}

    # ROC-AUC
    try:
        metrics["roc-auc"] = float(roc_auc_score(y_true_arr, y_scores_arr, average="micro"))
    except Exception:
        metrics["roc-auc"] = float("nan")

    # F1 at 0.5 threshold
    y_pred = (y_scores_arr >= 0.5).astype(np.float32)
    tp = np.sum(y_true_arr * y_pred)
    fp = np.sum((1 - y_true_arr) * y_pred)
    fn = np.sum(y_true_arr * (1 - y_pred))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    metrics["f1"] = float(2 * (precision * recall) / (precision + recall)) if precision + recall > 0 else 0.0
    metrics["precision"] = float(precision)
    metrics["recall"] = float(recall)

    # Per-class AP
    ap_per_class: list[float] = []
    for ci in range(y_true_arr.shape[1]):
        try:
            ap = average_precision_score(y_true_arr[:, ci], y_scores_arr[:, ci])
        except Exception:
            ap = np.nan
        ap_per_class.append(ap)
    ap_valid = [a for a in ap_per_class if not (a is None or (isinstance(a, float) and math.isnan(a)))]
    metrics["ap_per_class"] = ap_per_class
    metrics["cmAP"] = float(np.mean(ap_valid)) if ap_valid else float("nan")

    # Micro AP
    try:
        metrics["mAP"] = float(average_precision_score(y_true_arr, y_scores_arr, average="micro"))
    except Exception:
        metrics["mAP"] = float("nan")

    # Latency stats
    if measure_latency and chunk_latencies_ms:
        lat = np.array(chunk_latencies_ms)
        metrics["latency_mean_ms"] = float(np.mean(lat))
        metrics["latency_median_ms"] = float(np.median(lat))
        metrics["latency_p95_ms"] = float(np.percentile(lat, 95))
        metrics["latency_p99_ms"] = float(np.percentile(lat, 99))
        metrics["total_chunks"] = total_chunks

    # Memory stats
    if profile_memory:
        rss_after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metrics["peak_rss_mb"] = round(rss_after_kb / 1024, 1)
        metrics["rss_delta_mb"] = round((rss_after_kb - rss_before_kb) / 1024, 1)

    return metrics, per_file, y_true_arr, y_scores_arr


def optimize_thresholds(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    classes: list[str],
) -> dict[str, float]:
    """Find per-class thresholds that maximize F1 using the precision-recall curve.

    Args:
        y_true: Binary ground-truth array of shape ``(n_samples, n_classes)``.
        y_scores: Predicted scores of shape ``(n_samples, n_classes)``.
        classes: Class name list matching the column order.

    Returns:
        Dictionary mapping each class name to its optimal threshold.
    """
    optimal: dict[str, float] = {}
    for ci, cls_name in enumerate(classes):
        col_true = y_true[:, ci]
        col_scores = y_scores[:, ci]
        if col_true.sum() == 0:
            optimal[cls_name] = 0.5
            continue
        prec, rec, thresholds = precision_recall_curve(col_true, col_scores)
        # precision_recall_curve returns len(thresholds) == len(prec) - 1
        f1_scores = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
        best_idx = int(np.argmax(f1_scores))
        optimal[cls_name] = float(thresholds[best_idx])
    return optimal


def bootstrap_ap_ci(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    classes: list[str],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> list[dict]:
    """Compute per-class AP with bootstrap confidence intervals.

    Args:
        y_true: Binary ground-truth array ``(n_samples, n_classes)``.
        y_scores: Predicted scores ``(n_samples, n_classes)``.
        classes: Class name list matching column order.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (e.g. 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        List of dicts, one per class:
        ``{"class": str, "ap": float, "ci_lower": float, "ci_upper": float,
          "n_positive": int, "n_total": int}``.
    """
    rng = np.random.default_rng(seed)
    n_samples = y_true.shape[0]
    alpha = (1 - confidence) / 2

    results: list[dict] = []
    for ci, cls_name in enumerate(classes):
        col_true = y_true[:, ci]
        col_scores = y_scores[:, ci]
        n_pos = int(col_true.sum())

        # Point estimate
        try:
            ap = float(average_precision_score(col_true, col_scores))
        except Exception:
            ap = float("nan")

        if n_pos == 0 or n_pos == n_samples:
            results.append(
                {
                    "class": cls_name,
                    "ap": ap,
                    "ci_lower": ap,
                    "ci_upper": ap,
                    "n_positive": n_pos,
                    "n_total": n_samples,
                }
            )
            continue

        # Bootstrap
        boot_aps: list[float] = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n_samples, size=n_samples)
            bt = col_true[idx]
            bs = col_scores[idx]
            if bt.sum() == 0 or bt.sum() == len(bt):
                continue
            try:
                boot_aps.append(float(average_precision_score(bt, bs)))
            except Exception:
                continue

        if boot_aps:
            ci_lower = float(np.percentile(boot_aps, 100 * alpha))
            ci_upper = float(np.percentile(boot_aps, 100 * (1 - alpha)))
        else:
            ci_lower = ci_upper = ap

        results.append(
            {
                "class": cls_name,
                "ap": ap,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n_positive": n_pos,
                "n_total": n_samples,
            }
        )

    return results


def compute_det_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Detection Error Tradeoff (DET) curve points.

    The DET curve plots false rejection rate (FRR = 1 - recall) against
    false acceptance rate (FAR = FPR) across thresholds. Standard in
    bioacoustics evaluation.

    Args:
        y_true: Binary ground-truth (flattened or 2-D, treated as binary).
        y_scores: Predicted scores (same shape as y_true).

    Returns:
        Tuple (far, frr, thresholds) — arrays of equal length.
    """
    y_t = y_true.ravel()
    y_s = y_scores.ravel()

    # Sort by descending score
    desc = np.argsort(-y_s)
    y_t[desc]
    y_s_sorted = y_s[desc]

    total_pos = y_t.sum()
    total_neg = len(y_t) - total_pos

    if total_pos == 0 or total_neg == 0:
        return np.array([0.0]), np.array([0.0]), np.array([0.5])

    # Unique thresholds
    unique_scores = np.unique(y_s_sorted)[::-1]

    far_list: list[float] = []
    frr_list: list[float] = []
    thr_list: list[float] = []

    for thr in unique_scores:
        pred_pos = y_s >= thr
        tp = np.sum(y_t[pred_pos])
        fp = np.sum(1 - y_t[pred_pos])
        fn = total_pos - tp

        far = fp / total_neg  # false acceptance rate
        frr = fn / total_pos  # false rejection rate
        far_list.append(far)
        frr_list.append(frr)
        thr_list.append(float(thr))

    return np.array(far_list), np.array(frr_list), np.array(thr_list)
