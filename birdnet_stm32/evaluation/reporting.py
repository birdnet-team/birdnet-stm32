"""ASCII visualization and CSV export for evaluation results."""

import os

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve


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


def print_confusion_matrix(y_true: np.ndarray, y_scores: np.ndarray, classes: list[str], threshold: float = 0.5):
    """Print an ASCII confusion matrix based on top-1 predictions.

    Args:
        y_true: One-hot ground-truth labels [N, C].
        y_scores: Predicted scores [N, C].
        classes: Ordered class names.
        threshold: Minimum score to count as a prediction (else "none").
    """
    true_idx = np.argmax(y_true, axis=1)
    pred_idx = np.argmax(y_scores, axis=1)

    # Suppress predictions below threshold
    max_scores = np.max(y_scores, axis=1)
    pred_idx[max_scores < threshold] = -1

    cm = confusion_matrix(true_idx, pred_idx, labels=list(range(len(classes))))

    # Truncate long class names for display
    max_name_len = min(12, max(len(c) for c in classes)) if classes else 6
    short_names = [c[:max_name_len] for c in classes]

    # Header
    header = " " * (max_name_len + 1) + " ".join(f"{n:>{max_name_len}}" for n in short_names)
    print(f"\nConfusion Matrix (rows=true, cols=predicted):\n{header}")

    for i, row in enumerate(cm):
        row_str = " ".join(f"{v:>{max_name_len}}" for v in row)
        print(f"{short_names[i]:>{max_name_len}} {row_str}")

    # Summary stats
    correct = np.trace(cm)
    total = np.sum(cm)
    print(f"\nAccuracy: {correct}/{total} ({100 * correct / max(total, 1):.1f}%)")


def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    classes: list[str],
    out_path: str,
    threshold: float = 0.5,
):
    """Save a matplotlib confusion matrix heatmap to a file.

    Requires matplotlib. Silently skips if not available.

    Args:
        y_true: One-hot ground-truth labels [N, C].
        y_scores: Predicted scores [N, C].
        classes: Ordered class names.
        out_path: Path to save the image (e.g., .png).
        threshold: Minimum score to count as a prediction.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping confusion matrix plot.")
        return

    true_idx = np.argmax(y_true, axis=1)
    pred_idx = np.argmax(y_scores, axis=1)
    max_scores = np.max(y_scores, axis=1)
    pred_idx[max_scores < threshold] = -1

    cm = confusion_matrix(true_idx, pred_idx, labels=list(range(len(classes))))

    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.5), max(5, len(classes) * 0.4)))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix plot saved to {out_path}")
