"""CLI entry point for on-board SD card inference tests."""

import argparse
import os

from birdnet_stm32.deploy.board_test import BoardTestConfig, run_board_test
from birdnet_stm32.deploy.config import resolve_deploy_config


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for board test."""
    parser = argparse.ArgumentParser(
        description="Run batch inference on audio files stored on the STM32N6570-DK SD card."
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
        help="Path to model_config.json (for num_classes, labels)",
    )
    parser.add_argument("--labels", type=str, default="", help="Path to _labels.txt")
    parser.add_argument("--serial_port", type=str, default="/dev/ttyACM0", help="Serial port")
    parser.add_argument("--serial_baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--timeout", type=int, default=300, help="Max seconds to wait for completion")
    parser.add_argument("--config", type=str, default="config.json", help="Deploy config JSON")
    parser.add_argument(
        "--save_output",
        type=str,
        default="",
        help="Save raw serial output to file",
    )
    return parser.parse_args()


def main():
    """Run on-board SD card inference test."""
    args = get_args()

    # Resolve deploy config
    cli_overrides = {}
    if args.model_path:
        cli_overrides["model_path"] = args.model_path
    deploy_cfg = resolve_deploy_config(cli_args=cli_overrides, config_path=args.config)

    # Resolve labels path
    labels_path = args.labels
    if not labels_path and args.model_config:
        root, _ = os.path.splitext(args.model_config)
        candidate = root.replace("_model_config", "_labels") + ".txt"
        if os.path.isfile(candidate):
            labels_path = candidate
    if not labels_path:
        # Infer from model path
        root, _ = os.path.splitext(deploy_cfg.model_path)
        root = root.replace("_quantized", "")
        candidate = root + "_labels.txt"
        if os.path.isfile(candidate):
            labels_path = candidate

    board_cfg = BoardTestConfig(
        deploy_cfg=deploy_cfg,
        labels_path=labels_path,
        serial_port=args.serial_port,
        serial_baud=args.serial_baud,
        timeout=args.timeout,
    )

    output = run_board_test(board_cfg)

    if args.save_output:
        with open(args.save_output, "w") as f:
            f.write(output)
        print(f"\nSerial output saved to {args.save_output}")


if __name__ == "__main__":
    main()
