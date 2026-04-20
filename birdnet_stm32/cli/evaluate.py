"""CLI entry point for model evaluation."""

import argparse
import math
import os

from birdnet_stm32.data.dataset import SUPPORTED_AUDIO_EXTS, load_file_paths_from_directory
from birdnet_stm32.evaluation.metrics import (
    bootstrap_ap_ci,
    compute_det_curve,
    evaluate,
    optimize_thresholds,
)
from birdnet_stm32.evaluation.reporting import (
    print_ascii_det_curve,
    print_ascii_histogram,
    print_ascii_pr_curve,
    print_confusion_matrix,
    save_benchmark_json,
    save_confusion_matrix_plot,
    save_det_curve_plot,
    save_html_report,
    save_predictions_csv,
    save_species_report_csv,
)
from birdnet_stm32.models.runners import load_model_runner


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate model on test audio (file-level pooling).")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .keras or .tflite model")
    parser.add_argument("--model_config", type=str, default="", help="Path to model config JSON")
    parser.add_argument("--data_path_test", type=str, required=True, help="Path to test dataset root")
    parser.add_argument("--max_files", type=int, default=-1, help="Max test files per class")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for chunk inference")
    parser.add_argument("--overlap", type=float, default=0.0, help="Chunk overlap (seconds)")
    parser.add_argument("--pooling", type=str, default="avg", choices=["avg", "max", "lme"])
    parser.add_argument("--save_csv", type=str, default="", help="Optional path to save predictions CSV")
    parser.add_argument("--confusion_matrix", action="store_true", default=False, help="Print confusion matrix")
    parser.add_argument("--save_cm_plot", type=str, default="", help="Save confusion matrix plot to file")
    parser.add_argument(
        "--optimize_thresholds", action="store_true", default=False, help="Find per-class optimal F1 thresholds"
    )
    parser.add_argument("--benchmark", type=str, default="", help="Save structured JSON benchmark report to this path")
    parser.add_argument(
        "--benchmark_latency",
        action="store_true",
        default=False,
        help="Measure per-chunk inference latency (mean, median, p95, p99)",
    )
    parser.add_argument(
        "--species_report",
        type=str,
        default="",
        help="Save per-species AP report with 95%% bootstrap CI to this CSV path",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for species AP confidence intervals (default: 1000)",
    )
    parser.add_argument("--det_curve", action="store_true", default=False, help="Print ASCII DET curve")
    parser.add_argument("--save_det_plot", type=str, default="", help="Save DET curve plot to file")
    parser.add_argument(
        "--report_html",
        type=str,
        default="",
        help="Generate a self-contained HTML evaluation report",
    )
    parser.add_argument(
        "--profile_memory",
        action="store_true",
        default=False,
        help="Report peak memory (RSS) during inference",
    )
    return parser.parse_args()


def main():
    """Evaluate a trained model on a class-structured test dataset."""
    args = get_args()

    # Resolve config
    model_cfg_path = args.model_config
    if not model_cfg_path:
        root, _ = os.path.splitext(args.model_path)
        model_cfg_path = root + "_model_config.json"
    if not os.path.isfile(model_cfg_path):
        raise FileNotFoundError(f"Model config JSON not found: {model_cfg_path}")
    from birdnet_stm32.training.config import ModelConfig

    cfg = ModelConfig.load(model_cfg_path).to_dict()

    classes = cfg.get("class_names", [])
    if not classes:
        raise ValueError("class_names missing in model config.")

    # Collect test files
    files, _ = load_file_paths_from_directory(
        args.data_path_test, classes=classes, exts=SUPPORTED_AUDIO_EXTS, max_samples=args.max_files
    )
    if not files:
        raise RuntimeError(f"No test audio found in {args.data_path_test}")

    # Load model
    runner = load_model_runner(args.model_path)

    # Evaluate
    metrics, per_file, y_true, y_scores = evaluate(
        model_runner=runner,
        files=files,
        classes=classes,
        cfg=cfg,
        pooling=args.pooling,
        batch_size=args.batch_size,
        overlap=max(0.0, min(cfg["chunk_duration"] - 0.1, args.overlap)),
        measure_latency=args.benchmark_latency,
        profile_memory=args.profile_memory,
    )

    # Print summary
    print(f"\nEvaluated {len(per_file)} files across {len(classes)} classes.")
    for k, v in metrics.items():
        if k == "ap_per_class":
            continue
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    if metrics.get("ap_per_class"):
        ap_list = metrics["ap_per_class"]
        ap_with_names = [
            (classes[i], ap_list[i])
            for i in range(len(classes))
            if not (ap_list[i] is None or (isinstance(ap_list[i], float) and math.isnan(ap_list[i])))
        ]
        ap_with_names.sort(key=lambda x: x[1], reverse=True)
        print("\nTop 10 classes by AP:")
        for cls_name, ap in ap_with_names[:10]:
            print(f"  {cls_name}: {ap:.4f}")
        print("\nBottom 10 classes by AP:")
        for cls_name, ap in ap_with_names[-10:]:
            print(f"  {cls_name}: {ap:.4f}")

    # ASCII visualizations
    print_ascii_histogram(y_scores.ravel())
    print_ascii_pr_curve(y_true, y_scores)

    # DET curve
    if args.det_curve or args.save_det_plot:
        far, frr, _thr = compute_det_curve(y_true, y_scores)
        if args.det_curve:
            print_ascii_det_curve(far, frr)
        if args.save_det_plot:
            save_det_curve_plot(far, frr, args.save_det_plot)

    # Species AP report with bootstrap CI
    species_data = None
    if args.species_report or args.benchmark or args.report_html:
        species_data = bootstrap_ap_ci(y_true, y_scores, classes, n_bootstrap=args.n_bootstrap)
        if args.species_report:
            save_species_report_csv(species_data, args.species_report)

    # Save CSV
    if args.save_csv:
        save_predictions_csv(per_file, classes, args.save_csv)
        print(f"Predictions saved to {args.save_csv}")

    # Confusion matrix
    if args.confusion_matrix:
        print_confusion_matrix(y_true, y_scores, classes)
    if args.save_cm_plot:
        save_confusion_matrix_plot(y_true, y_scores, classes, args.save_cm_plot)

    # Threshold optimization
    if args.optimize_thresholds:
        thresholds = optimize_thresholds(y_true, y_scores, classes)
        print("\nOptimal per-class thresholds (max F1):")
        for cls_name, thr in sorted(thresholds.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cls_name}: {thr:.4f}")

    # Benchmark JSON
    if args.benchmark:
        save_benchmark_json(
            metrics,
            classes,
            args.model_path,
            args.benchmark,
            species_data=species_data,
            config=cfg,
        )

    # HTML report
    if args.report_html:
        save_html_report(
            metrics,
            classes,
            y_true,
            y_scores,
            args.model_path,
            args.report_html,
            species_data=species_data,
            per_file=per_file,
        )


if __name__ == "__main__":
    main()
