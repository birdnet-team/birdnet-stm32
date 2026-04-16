"""CLI entry point for on-board inference tests."""

import argparse
import os

from birdnet_stm32.deploy.board_test import BoardTestConfig, run_board_test
from birdnet_stm32.deploy.config import resolve_deploy_config


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for board test."""
    parser = argparse.ArgumentParser(
        description="Run inference on real audio data using the STM32N6570-DK NPU."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Path to quantized .tflite model (default: from config.json)",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="",
        help="Path to _model_config.json (required)",
    )
    parser.add_argument("--labels", type=str, default="", help="Path to _labels.txt")
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="data/test",
        help="Directory tree of .wav files to test",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Top-K predictions per file")
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.01,
        help="Minimum score to display",
    )
    parser.add_argument("--config", type=str, default="config.json", help="Deploy config JSON")
    parser.add_argument(
        "--save_results",
        type=str,
        default="",
        help="Save results summary to a CSV file",
    )
    return parser.parse_args()


def main():
    """Run on-board inference test."""
    args = get_args()

    # Resolve deploy config
    cli_overrides = {}
    if args.model_path:
        cli_overrides["model_path"] = args.model_path
    deploy_cfg = resolve_deploy_config(cli_args=cli_overrides, config_path=args.config)

    # Resolve model config path
    model_config = args.model_config
    if not model_config:
        root, _ = os.path.splitext(deploy_cfg.model_path)
        root = root.replace("_quantized", "")
        candidate = root + "_model_config.json"
        if os.path.isfile(candidate):
            model_config = candidate
    if not model_config or not os.path.isfile(model_config):
        print("[ERROR] Model config not found. Supply --model_config path.")
        raise SystemExit(1)

    # Resolve labels path
    labels_path = args.labels
    if not labels_path:
        root, _ = os.path.splitext(deploy_cfg.model_path)
        root = root.replace("_quantized", "")
        candidate = root + "_labels.txt"
        if os.path.isfile(candidate):
            labels_path = candidate

    board_cfg = BoardTestConfig(
        deploy_cfg=deploy_cfg,
        model_config_path=model_config,
        labels_path=labels_path,
        audio_dir=args.audio_dir,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
    )

    result = run_board_test(board_cfg)

    if args.save_results and result["results"]:
        import csv

        with open(args.save_results, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "top_label", "top_score"])
            for r in result["results"]:
                top = r["detections"][0] if r["detections"] else {"label": "", "score": 0.0}
                writer.writerow([r["file"], top["label"], f"{top['score']:.4f}"])
        print(f"\nResults saved to {args.save_results}")


if __name__ == "__main__":
    main()
