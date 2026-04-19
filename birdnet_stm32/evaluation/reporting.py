"""ASCII visualization, CSV/JSON export, and optional HTML report for evaluation results."""

import json
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


def save_species_report_csv(species_data: list[dict], out_path: str) -> None:
    """Save per-species AP with confidence intervals to CSV.

    Args:
        species_data: Output of ``bootstrap_ap_ci()``.
        out_path: Output CSV path.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("class,ap,ci_lower,ci_upper,n_positive,n_total\n")
        for row in sorted(species_data, key=lambda r: r["ap"], reverse=True):
            f.write(
                f"{row['class']},{row['ap']:.6f},"
                f"{row['ci_lower']:.6f},{row['ci_upper']:.6f},"
                f"{row['n_positive']},{row['n_total']}\n"
            )
    print(f"Species AP report saved to {out_path}")


def save_benchmark_json(
    metrics: dict,
    classes: list[str],
    model_path: str,
    out_path: str,
    species_data: list[dict] | None = None,
    config: dict | None = None,
) -> None:
    """Save a structured JSON benchmark report.

    Args:
        metrics: Metrics dict from ``evaluate()``.
        classes: Ordered class names.
        model_path: Path to the evaluated model.
        out_path: Output JSON path.
        species_data: Optional per-species AP with CIs from ``bootstrap_ap_ci()``.
        config: Optional model config dict.
    """
    report: dict = {
        "model_path": model_path,
        "num_classes": len(classes),
        "num_files": metrics.get("total_chunks", 0),
    }

    # Core metrics (exclude internal arrays)
    core = {}
    for k, v in metrics.items():
        if k == "ap_per_class":
            continue
        if isinstance(v, float):
            core[k] = round(v, 6)
        else:
            core[k] = v
    report["metrics"] = core

    if species_data:
        report["species"] = species_data

    if config:
        report["config"] = config

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Benchmark report saved to {out_path}")


def print_ascii_det_curve(far: np.ndarray, frr: np.ndarray, bins: int = 10, width: int = 40) -> None:
    """Print an ASCII DET curve (FAR vs FRR).

    Args:
        far: False acceptance rate array.
        frr: False rejection rate array.
        bins: Number of FRR bins.
        width: Bar width in characters.
    """
    print("\nASCII DET Curve (FRR down, FAR right):")
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        mask = (frr >= lo) & (frr < hi)
        min_far = float(np.min(far[mask])) if np.any(mask) else 1.0
        bar = "#" * int(width * min_far)
        print(f"FRR {lo:4.2f}-{hi:4.2f} | {bar} (FAR={min_far:4.3f})")


def save_det_curve_plot(
    far: np.ndarray, frr: np.ndarray, out_path: str
) -> None:
    """Save a matplotlib DET curve plot.

    Requires matplotlib. Silently skips if not available.

    Args:
        far: False acceptance rate array.
        frr: False rejection rate array.
        out_path: Output image path.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping DET curve plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(far, frr, linewidth=1.5)
    ax.set_xlabel("False Acceptance Rate (FAR)")
    ax.set_ylabel("False Rejection Rate (FRR)")
    ax.set_title("Detection Error Tradeoff (DET) Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"DET curve saved to {out_path}")


def save_html_report(
    metrics: dict,
    classes: list[str],
    y_true: np.ndarray,
    y_scores: np.ndarray,
    model_path: str,
    out_path: str,
    species_data: list[dict] | None = None,
    per_file: list[dict] | None = None,
) -> None:
    """Generate a self-contained HTML evaluation report.

    Uses inline CSS and basic HTML tables — no external dependencies.
    Optionally embeds matplotlib charts as base64 PNG if matplotlib is available.

    Args:
        metrics: Metrics dict from ``evaluate()``.
        classes: Ordered class names.
        y_true: Ground-truth labels ``(N, C)``.
        y_scores: Predicted scores ``(N, C)``.
        model_path: Path to the evaluated model.
        out_path: Output HTML path.
        species_data: Optional per-species AP with CIs.
        per_file: Optional per-file predictions.
    """
    has_mpl = False
    try:
        import base64
        import io

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        has_mpl = True
    except ImportError:
        pass

    html_parts: list[str] = []
    html_parts.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    html_parts.append("<title>Evaluation Report</title>")
    html_parts.append("<style>")
    html_parts.append("""
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               max-width: 960px; margin: 2em auto; padding: 0 1em; color: #333; }
        h1, h2, h3 { color: #1a1a2e; }
        table { border-collapse: collapse; width: 100%; margin: 1em 0; }
        th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; }
        th { background: #f5f5f5; }
        tr:nth-child(even) { background: #fafafa; }
        .metric-val { font-weight: bold; font-size: 1.1em; }
        .chart { margin: 1em 0; text-align: center; }
        .chart img { max-width: 100%; }
    """)
    html_parts.append("</style></head><body>")

    # Header
    html_parts.append("<h1>Evaluation Report</h1>")
    html_parts.append(f"<p>Model: <code>{os.path.basename(model_path)}</code></p>")

    # Metrics table
    html_parts.append("<h2>Summary Metrics</h2><table><tr><th>Metric</th><th>Value</th></tr>")
    for k, v in metrics.items():
        if k == "ap_per_class":
            continue
        if isinstance(v, float):
            html_parts.append(f"<tr><td>{k}</td><td class='metric-val'>{v:.4f}</td></tr>")
        else:
            html_parts.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
    html_parts.append("</table>")

    # Species AP table
    if species_data:
        html_parts.append("<h2>Per-Species Average Precision</h2>")
        html_parts.append("<table><tr><th>Species</th><th>AP</th>"
                         "<th>95% CI</th><th>N pos</th><th>N total</th></tr>")
        for row in sorted(species_data, key=lambda r: r["ap"], reverse=True):
            html_parts.append(
                f"<tr><td>{row['class']}</td><td class='metric-val'>{row['ap']:.4f}</td>"
                f"<td>[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]</td>"
                f"<td>{row['n_positive']}</td><td>{row['n_total']}</td></tr>"
            )
        html_parts.append("</table>")

    # Confusion matrix
    if has_mpl:
        true_idx = np.argmax(y_true, axis=1)
        pred_idx = np.argmax(y_scores, axis=1)
        cm = confusion_matrix(true_idx, pred_idx, labels=list(range(len(classes))))

        fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.5), max(5, len(classes) * 0.4)))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
               xticklabels=classes, yticklabels=classes,
               ylabel="True", xlabel="Predicted", title="Confusion Matrix")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("ascii")
        html_parts.append(f"<h2>Confusion Matrix</h2><div class='chart'>"
                         f"<img src='data:image/png;base64,{img_b64}'></div>")

    html_parts.append("</body></html>")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(html_parts))
    print(f"HTML report saved to {out_path}")
