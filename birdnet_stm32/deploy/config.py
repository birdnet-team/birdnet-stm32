"""Configuration resolution for deployment paths.

Resolves paths to X-CUBE-AI tools, models, and project directories from
(in priority order): CLI arguments > environment variables > config file.
"""

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
        stedgeai_path: Full path to the stedgeai binary (derived).
        n6_loader_script: Full path to n6_loader.py (derived).
    """

    x_cube_ai_path: str = ""
    model_path: str = ""
    output_dir: str = "validation/st_ai_output"
    workspace_dir: str = "validation/st_ai_ws"
    n6_loader_config: str = "config_n6l.json"
    stedgeai_path: str = field(init=False, default="")
    n6_loader_script: str = field(init=False, default="")

    def __post_init__(self):
        """Derive tool paths from x_cube_ai_path."""
        if self.x_cube_ai_path:
            self.stedgeai_path = os.path.join(self.x_cube_ai_path, "Utilities", "linux", "stedgeai")
            self.n6_loader_script = os.path.join(self.x_cube_ai_path, "scripts", "N6_scripts", "n6_loader.py")


def resolve_deploy_config(
    cli_args: dict | None = None,
    config_path: str = "config.json",
) -> DeployConfig:
    """Resolve deployment configuration from CLI args, environment, and config file.

    Priority: CLI arguments > environment variables > config file defaults.

    Environment variables:
        X_CUBE_AI_PATH: Root directory of X-CUBE-AI installation.
        BIRDNET_MODEL_PATH: Path to the quantized .tflite model.

    Args:
        cli_args: Optional dict of CLI overrides.
        config_path: Path to JSON config file.

    Returns:
        Populated DeployConfig.
    """
    file_cfg: dict = {}
    if os.path.isfile(config_path):
        with open(config_path) as f:
            file_cfg = json.load(f)

    cli_args = cli_args or {}

    def _resolve(cli_key: str, env_key: str, file_key: str, default: str = "") -> str:
        return cli_args.get(cli_key) or os.environ.get(env_key) or file_cfg.get(file_key) or default

    cfg = DeployConfig(
        x_cube_ai_path=_resolve("x_cube_ai_path", "X_CUBE_AI_PATH", "x_cube_ai_path"),
        model_path=_resolve(
            "model_path", "BIRDNET_MODEL_PATH", "model_path", "checkpoints/best_model_quantized.tflite"
        ),
        output_dir=_resolve("output_dir", "", "output_dir", "validation/st_ai_output"),
        workspace_dir=_resolve("workspace_dir", "", "workspace_dir", "validation/st_ai_ws"),
        n6_loader_config=_resolve("n6_loader_config", "", "n6_loader_config", "config_n6l.json"),
    )
    return cfg
