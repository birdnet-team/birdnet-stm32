"""CLI entry point for standalone on-board inference tests."""

import argparse
import os

from birdnet_stm32.deploy.board_test import BoardTestConfig, run_board_test
from birdnet_stm32.deploy.config import resolve_deploy_config


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for board test."""
    parser = argparse.ArgumentParser(
        description=(
            "Run standalone inference on the STM32N6570-DK: firmware reads WAV "
            "from SD card, applies frontend-specific preprocessing, runs NPU "
            "inference, and streams results over UART."
        ),
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
        "--serial_port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port for UART capture (default: /dev/ttyACM0)",
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
        "--timeout",
        type=int,
        default=300,
        help="Max seconds to wait for firmware (default: 300)",
    )
    parser.add_argument(
        "--host_audio_dir",
        type=str,
        default="",
        help=(
            "Local copy of the SD card's audio/ folder. The host scores the same files with the same "
            "model and every board result is checked against it; the command exits nonzero on a mismatch"
        ),
    )
    parser.add_argument(
        "--parity_tolerance",
        type=float,
        default=0.05,
        help="Largest allowed |board - host| score per printed label, and the top-1 tie margin",
    )
    parser.add_argument(
        "--save_results",
        type=str,
        default="",
        help="Save results summary to a CSV file",
    )
    return parser.parse_args()


def main():
    """Run standalone on-board inference test."""
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
        serial_port=args.serial_port,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        timeout=args.timeout,
        host_audio_dir=args.host_audio_dir,
        parity_tolerance=args.parity_tolerance,
    )

    result = run_board_test(board_cfg)

    parity = result.get("parity")
    if args.save_results and result["results"]:
        import csv

        by_file = {row["file"]: row for row in parity["rows"]} if parity else {}
        with open(args.save_results, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["file", "top_label", "top_score"]
            if parity:
                header += ["host_top_label", "host_top_score", "agreement", "max_score_diff", "true_label"]
            writer.writerow(header)
            for r in result["results"]:
                top = r["detections"][0] if r["detections"] else {"label": "", "score": 0.0}
                line = [r["file"], top["label"], f"{top['score']:.4f}"]
                if parity:
                    row = by_file[r["file"]]
                    line += [
                        row["host_top1"],
                        f"{row['host_score']:.4f}",
                        row["agreement"] if row["ok"] else f"FAIL:{row['agreement']}",
                        f"{row['max_score_diff']:.4f}",
                        row["true_label"],
                    ]
                writer.writerow(line)
        print(f"\nResults saved to {args.save_results}")

    if parity is not None and not parity["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
