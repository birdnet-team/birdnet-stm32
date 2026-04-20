"""Tests for deploy configuration and stedgeai module."""

import os

from birdnet_stm32.deploy.config import DeployConfig, resolve_deploy_config
from birdnet_stm32.deploy.stedgeai import detect_board


class TestDeployConfig:
    """Tests for DeployConfig and resolve_deploy_config."""

    def test_default_values(self):
        cfg = DeployConfig()
        assert cfg.output_dir == "validation/st_ai_output"
        assert cfg.workspace_dir == "validation/st_ai_ws"
        assert cfg.n6_loader_config == "config_n6l.json"
        assert cfg.stedgeai_path == ""
        assert cfg.n6_loader_script == ""

    def test_derives_tool_paths(self):
        cfg = DeployConfig(x_cube_ai_path="/opt/XCUBEAI")
        assert cfg.stedgeai_path == "/opt/XCUBEAI/Utilities/linux/stedgeai"
        assert "n6_loader.py" in cfg.n6_loader_script

    def test_resolve_cli_overrides(self, tmp_path):
        cli = {"model_path": str(tmp_path / "model.tflite")}
        cfg = resolve_deploy_config(cli_args=cli, config_path=str(tmp_path / "nonexistent.json"))
        assert cfg.model_path == str(tmp_path / "model.tflite")

    def test_resolve_from_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(
            '[deploy]\nx_cube_ai_path = "/opt/AI"\nmodel_path = "m.tflite"\n'
            '[n6_loader]\n"network.c" = "net.c"\nproject_path = "/proj"\n'
        )
        cfg = resolve_deploy_config(cli_args={}, config_path=str(toml_path))
        assert cfg.x_cube_ai_path == "/opt/AI"
        assert cfg.model_path == "m.tflite"
        # n6_loader section should generate a temp JSON
        assert cfg.n6_loader_config.endswith("_n6l.json")
        assert os.path.isfile(cfg.n6_loader_config)
        # Clean up temp file
        os.unlink(cfg.n6_loader_config)

    def test_resolve_fallback_json_to_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text('[deploy]\nmodel_path = "fallback.tflite"\n')
        cfg = resolve_deploy_config(cli_args={}, config_path=str(tmp_path / "config.json"))
        assert cfg.model_path == "fallback.tflite"


class TestDetectBoard:
    """Tests for detect_board auto-detection."""

    def test_returns_none_or_string(self):
        result = detect_board()
        assert result is None or result.startswith("/dev/ttyACM")
