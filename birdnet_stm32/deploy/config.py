"""Configuration resolution for deployment paths.

Resolves paths to X-CUBE-AI tools, models, and project directories from
(in priority order): CLI arguments > environment variables > config file.

Supports both JSON (``config.json``) and TOML (``config.toml``) config files.
When a ``.toml`` file is provided, deploy-related keys are read from the
``[deploy]`` table.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass
class DeployConfig:
    """Deployment configuration for STM32N6 target.

    Attributes:
        x_cube_ai_path: Root directory of X-CUBE-AI installation.
        model_path: Path to the quantized .tflite model.
        output_dir: Directory for stedgeai output artifacts.
        workspace_dir: Directory for stedgeai workspace.
        n6_loader_config: Path to n6_loader JSON config.
        cubeide_path: Path to STM32CubeIDE (optional).
        arm_toolchain_path: Path to arm-none-eabi toolchain (optional).
        stedgeai_path: Full path to the stedgeai binary (derived).
        n6_loader_script: Full path to n6_loader.py (derived).
    """

    x_cube_ai_path: str = ""
    model_path: str = ""
    output_dir: str = "validation/st_ai_output"
    workspace_dir: str = "validation/st_ai_ws"
    n6_loader_config: str = "config_n6l.json"
    cubeide_path: str = ""
    arm_toolchain_path: str = ""
    stedgeai_path: str = field(init=False, default="")
    n6_loader_script: str = field(init=False, default="")

    def __post_init__(self):
        """Derive tool paths from x_cube_ai_path."""
        if self.x_cube_ai_path:
            self.stedgeai_path = os.path.join(self.x_cube_ai_path, "Utilities", "linux", "stedgeai")
            self.n6_loader_script = os.path.join(self.x_cube_ai_path, "scripts", "N6_scripts", "n6_loader.py")


def _load_config_file(config_path: str) -> dict:
    """Load a JSON or TOML config file and return a flat deploy dict.

    Args:
        config_path: Path to config.json or config.toml.

    Returns:
        Dictionary of deploy configuration values.
    """
    if not os.path.isfile(config_path):
        return {}

    if config_path.endswith(".toml"):
        import tomllib  # type: ignore[no-redef]

        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        # Flatten: merge [deploy] and [build] sections
        flat: dict = {}
        flat.update(data.get("deploy", {}))
        flat.update(data.get("build", {}))
        return flat

    with open(config_path) as f:
        return json.load(f)


def resolve_deploy_config(
    cli_args: dict | None = None,
    config_path: str = "config.json",
) -> DeployConfig:
    """Resolve deployment configuration from CLI args, environment, and config file.

    Priority: CLI arguments > environment variables > config file defaults.

    If the specified *config_path* does not exist, the resolver also tries
    ``config.toml`` (and vice-versa) as a fallback.

    Environment variables:
        X_CUBE_AI_PATH: Root directory of X-CUBE-AI installation.
        STEDGEAI_PATH: Direct path to the stedgeai binary.
        CUBEIDE_PATH: Path to STM32CubeIDE.
        ARM_TOOLCHAIN_PATH: Path to arm-none-eabi toolchain.
        BIRDNET_MODEL_PATH: Path to the quantized .tflite model.

    Args:
        cli_args: Optional dict of CLI overrides.
        config_path: Path to JSON or TOML config file.

    Returns:
        Populated DeployConfig.
    """
    # Try the given path; fall back to the other format if not found
    file_cfg = _load_config_file(config_path)
    if not file_cfg:
        alt = (
            config_path.replace(".json", ".toml")
            if config_path.endswith(".json")
            else config_path.replace(".toml", ".json")
        )
        file_cfg = _load_config_file(alt)

    cli_args = cli_args or {}

    def _resolve(cli_key: str, env_key: str, file_key: str, default: str = "") -> str:
        return cli_args.get(cli_key) or os.environ.get(env_key, "") or file_cfg.get(file_key, "") or default

    cfg = DeployConfig(
        x_cube_ai_path=_resolve("x_cube_ai_path", "X_CUBE_AI_PATH", "x_cube_ai_path"),
        model_path=_resolve(
            "model_path", "BIRDNET_MODEL_PATH", "model_path", "checkpoints/best_model_quantized.tflite"
        ),
        output_dir=_resolve("output_dir", "", "output_dir", "validation/st_ai_output"),
        workspace_dir=_resolve("workspace_dir", "", "workspace_dir", "validation/st_ai_ws"),
        n6_loader_config=_resolve("n6_loader_config", "", "n6_loader_config", "config_n6l.json"),
        cubeide_path=_resolve("cubeide_path", "CUBEIDE_PATH", "cubeide_path"),
        arm_toolchain_path=_resolve("arm_toolchain_path", "ARM_TOOLCHAIN_PATH", "arm_toolchain_path"),
    )

    # Allow STEDGEAI_PATH to override derived path
    stedgeai_override = cli_args.get("stedgeai_path") or os.environ.get("STEDGEAI_PATH", "")
    if stedgeai_override:
        cfg.stedgeai_path = stedgeai_override

    return cfg
