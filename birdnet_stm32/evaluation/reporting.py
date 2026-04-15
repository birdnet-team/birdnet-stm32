"""ASCII visualization and CSV export for evaluation results."""

import os

import numpy as np
from sklearn.metrics import precision_recall_curve


def print_ascii_histogram(scores: np.ndarray, bins: int = 10, width: int = 40):
    """Print an ASCII histogram of scores in [0,1].

    Args:
        scores: Array of scores in [0, 1].
        bins: Number of histogram bins.
        width: Bar width in characters.
    """
    hist, bin_edges = np.histogram(scores, bins=bins, range=(0, 1))
    max_count = np.max(hist)
    for i in range(bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        bar = "#" * int(width * hist[i] / max_count) if max_count > 0 else ""
        print(f"{left:4.2f} - {right:4.2f} | {bar} ({hist[i]})")


def print_ascii_pr_curve(y_true: np.ndarray, y_scores: np.ndarray, bins: int = 10, width: int = 40):
    """Print an ASCII PR curve with fixed precision bins.

    Args:
        y_true: One-hot ground-truth labels [N_files, C].
        y_scores: Pooled scores [N_files, C].
        bins: Number of precision bins.
        width: Bar width in characters.
    """
    y_true_flat = y_true.ravel()
    y_scores_flat = y_scores.ravel()
    precisions, recalls, _thresholds = precision_recall_curve(y_true_flat, y_scores_flat)
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    bin_edges = np.linspace(1.0, 0.0, bins + 1)
    print("\nASCII Precision-Recall Curve (precision down, recall right):")
    for i in range(bins):
        p_lo = bin_edges[i + 1]
        p_hi = bin_edges[i]
        mask = (precisions >= p_lo) & (precisions <= p_hi)
        max_recall = float(np.max(recalls[mask])) if np.any(mask) else 0.0
        bar = "#" * int(width * max_recall)
        print(f"{p_hi:4.1f} | {bar} ({max_recall:4.2f})")


def save_predictions_csv(per_file: list[dict], classes: list[str], out_path: str):
    """Save per-file predictions to CSV.

    Columns: file, label, top1_label, top1_score, <class columns...>

    Args:
        per_file: One dict per file with keys: file, label, scores.
        classes: Ordered class names.
        out_path: Path to save the CSV.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        header = ["file", "label", "top1_label", "top1_score"] + classes
        f.write(",".join(header) + "\n")
        for row in per_file:
            scores = np.array(row["scores"])
            top1_idx = int(np.argmax(scores))
            top1_label = classes[top1_idx]
            top1_score = scores[top1_idx]
            vals = [
                row["file"],
                row["label"],
                top1_label,
                f"{top1_score:.3f}",
            ] + [f"{s:.3f}" for s in scores]
            f.write(",".join(vals) + "\n")
