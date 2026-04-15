"""CLI entry point for STM32N6 deployment."""

import argparse

from birdnet_stm32.deploy.config import resolve_deploy_config
from birdnet_stm32.deploy.stedgeai import deploy_full


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for deployment."""
    parser = argparse.ArgumentParser(description="Deploy quantized TFLite model to STM32N6570-DK.")
    parser.add_argument("--x_cube_ai_path", type=str, default="", help="Path to X-CUBE-AI installation")
    parser.add_argument("--model_path", type=str, default="", help="Path to quantized .tflite model")
    parser.add_argument("--output_dir", type=str, default="", help="stedgeai output directory")
    parser.add_argument("--workspace_dir", type=str, default="", help="stedgeai workspace directory")
    parser.add_argument("--n6_loader_config", type=str, default="", help="Path to n6_loader JSON config")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config JSON file")
    return parser.parse_args()


def main():
    """Deploy a quantized model to the STM32N6570-DK board."""
    args = get_args()

    cli_args = {k: v for k, v in vars(args).items() if v and k != "config"}

    cfg = resolve_deploy_config(cli_args=cli_args, config_path=args.config)
    deploy_full(cfg)


if __name__ == "__main__":
    main()
