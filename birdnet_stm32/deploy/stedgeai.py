"""stedgeai deployment commands: generate, load, and validate on STM32N6."""

import glob
import os
import subprocess
import sys

from birdnet_stm32.deploy.config import DeployConfig

# ANSI color helpers (disabled when stdout is not a terminal)
_USE_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    """Wrap *text* with ANSI color *code* if colors are enabled."""
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _green(text: str) -> str:
    return _c("32", text)


def _yellow(text: str) -> str:
    return _c("33", text)


def _red(text: str) -> str:
    return _c("1;31", text)


def _cyan(text: str) -> str:
    return _c("36", text)


def detect_board() -> str | None:
    """Auto-detect an STM32 board serial port.

    Scans ``/dev/ttyACM*`` and returns the first match, or ``None``.
    """
    ports = sorted(glob.glob("/dev/ttyACM*"))
    return ports[0] if ports else None


def _run(cmd: list[str], description: str, *, dry_run: bool = False):
    """Run a subprocess command and exit on failure.

    Args:
        cmd: Command and arguments.
        description: Human-readable description for error messages.
        dry_run: If True, print the command without executing it.
    """
    print(f"{_cyan('[deploy]')} {description}")
    print(f"  $ {' '.join(cmd)}")
    if dry_run:
        print(f"  {_yellow('(dry-run — skipped)')}")
        return
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"{_red('[deploy] FAILED:')} {description} (exit code {result.returncode})")
        sys.exit(result.returncode)


def generate(cfg: DeployConfig, *, dry_run: bool = False):
    """Run stedgeai generate to produce the target project.

    Args:
        cfg: Deployment configuration.
        dry_run: If True, print the command without executing it.
    """
    if not dry_run:
        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(cfg.workspace_dir, exist_ok=True)
    _run(
        [
            cfg.stedgeai_path,
            "generate",
            "--model",
            cfg.model_path,
            "--target",
            "stm32n6",
            "--st-neural-art",
            "--output",
            cfg.output_dir,
            "--workspace",
            cfg.workspace_dir,
        ],
        "Generate target project",
        dry_run=dry_run,
    )


def load_to_target(cfg: DeployConfig, *, dry_run: bool = False):
    """Flash the model to the STM32N6 board via n6_loader.py.

    Args:
        cfg: Deployment configuration.
        dry_run: If True, print the command without executing it.
    """
    _run(
        [sys.executable, cfg.n6_loader_script, "--n6-loader-config", cfg.n6_loader_config],
        "Load model to target via n6_loader",
        dry_run=dry_run,
    )


def validate_on_target(cfg: DeployConfig, *, dry_run: bool = False):
    """Run stedgeai validate on the physical STM32N6 board.

    Args:
        cfg: Deployment configuration.
        dry_run: If True, print the command without executing it.
    """
    _run(
        [
            cfg.stedgeai_path,
            "validate",
            "--model",
            cfg.model_path,
            "--target",
            "stm32n6",
            "--mode",
            "target",
            "--desc",
            "serial:921600",
            "--output",
            cfg.output_dir,
            "--workspace",
            cfg.workspace_dir,
        ],
        "Validate on target board",
        dry_run=dry_run,
    )


def deploy_full(
    cfg: DeployConfig,
    *,
    dry_run: bool = False,
    skip_validate: bool = False,
):
    """Run the full deployment pipeline: generate, load, validate.

    Args:
        cfg: Deployment configuration.
        dry_run: If True, print commands without executing them.
        skip_validate: If True, skip the on-target validation step.
    """
    if dry_run:
        print(f"{_yellow('[deploy] DRY RUN — no commands will be executed')}")

    # Auto-detect board
    port = detect_board()
    if port:
        print(f"{_green('[deploy]')} Board detected on {port}")
    else:
        print(f"{_yellow('[deploy]')} No STM32 board detected on /dev/ttyACM*")

    # Pre-flight checks
    if not dry_run:
        if not os.path.isfile(cfg.stedgeai_path):
            print(f"{_red('[deploy]')} stedgeai not found: {cfg.stedgeai_path}")
            sys.exit(1)
        if not os.path.isfile(cfg.model_path):
            print(f"{_red('[deploy]')} Model not found: {cfg.model_path}")
            sys.exit(1)
        if not os.path.isfile(cfg.n6_loader_script):
            print(f"{_red('[deploy]')} n6_loader.py not found: {cfg.n6_loader_script}")
            sys.exit(1)
        if not os.path.isfile(cfg.n6_loader_config):
            print(f"{_red('[deploy]')} N6 loader config not found: {cfg.n6_loader_config}")
            sys.exit(1)

    generate(cfg, dry_run=dry_run)
    load_to_target(cfg, dry_run=dry_run)

    if skip_validate:
        print(f"{_yellow('[deploy]')} Skipping on-target validation (--skip_validate)")
    else:
        validate_on_target(cfg, dry_run=dry_run)

    print(f"{_green('[deploy] Done.')}")
