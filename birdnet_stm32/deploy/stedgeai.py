"""stedgeai deployment commands: generate, load, and validate on STM32N6."""

import os
import subprocess
import sys

from birdnet_stm32.deploy.config import DeployConfig


def _run(cmd: list[str], description: str):
    """Run a subprocess command and exit on failure.

    Args:
        cmd: Command and arguments.
        description: Human-readable description for error messages.
    """
    print(f"[deploy] {description}")
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[deploy] FAILED: {description} (exit code {result.returncode})")
        sys.exit(result.returncode)


def generate(cfg: DeployConfig):
    """Run stedgeai generate to produce the target project.

    Args:
        cfg: Deployment configuration.
    """
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.workspace_dir, exist_ok=True)
    _run(
        [
            cfg.stedgeai_path,
            "generate",
            "--model", cfg.model_path,
            "--target", "stm32n6",
            "--st-neural-art",
            "--output", cfg.output_dir,
            "--workspace", cfg.workspace_dir,
        ],
        "Generate target project",
    )


def load_to_target(cfg: DeployConfig):
    """Flash the model to the STM32N6 board via n6_loader.py.

    Args:
        cfg: Deployment configuration.
    """
    _run(
        [sys.executable, cfg.n6_loader_script, "--n6-loader-config", cfg.n6_loader_config],
        "Load model to target via n6_loader",
    )


def validate_on_target(cfg: DeployConfig):
    """Run stedgeai validate on the physical STM32N6 board.

    Args:
        cfg: Deployment configuration.
    """
    _run(
        [
            cfg.stedgeai_path,
            "validate",
            "--model", cfg.model_path,
            "--target", "stm32n6",
            "--mode", "target",
            "--desc", "serial:921600",
            "--output", cfg.output_dir,
            "--workspace", cfg.workspace_dir,
        ],
        "Validate on target board",
    )


def deploy_full(cfg: DeployConfig):
    """Run the full deployment pipeline: generate, load, validate.

    Args:
        cfg: Deployment configuration.
    """
    # Pre-flight checks
    if not os.path.isfile(cfg.stedgeai_path):
        print(f"[deploy] stedgeai not found: {cfg.stedgeai_path}")
        sys.exit(1)
    if not os.path.isfile(cfg.model_path):
        print(f"[deploy] Model not found: {cfg.model_path}")
        sys.exit(1)
    if not os.path.isfile(cfg.n6_loader_script):
        print(f"[deploy] n6_loader.py not found: {cfg.n6_loader_script}")
        sys.exit(1)
    if not os.path.isfile(cfg.n6_loader_config):
        print(f"[deploy] N6 loader config not found: {cfg.n6_loader_config}")
        sys.exit(1)

    generate(cfg)
    load_to_target(cfg)
    validate_on_target(cfg)
