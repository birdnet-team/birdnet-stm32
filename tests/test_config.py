"""Unit tests for deployment configuration resolution."""

from birdnet_stm32.deploy.config import DeployConfig, resolve_deploy_config


class TestDeployConfig:
    """Tests for DeployConfig dataclass."""

    def test_derived_paths(self):
        """stedgeai and n6_loader paths should be derived from x_cube_ai_path."""
        cfg = DeployConfig(x_cube_ai_path="/opt/X-CUBE-AI.10.2.0")
        assert "stedgeai" in cfg.stedgeai_path
        assert "n6_loader" in cfg.n6_loader_script

    def test_empty_x_cube(self):
        """Empty x_cube_ai_path should leave derived paths empty."""
        cfg = DeployConfig()
        assert cfg.stedgeai_path == ""
        assert cfg.n6_loader_script == ""


class TestResolveDeployConfig:
    """Tests for resolve_deploy_config."""

    def test_cli_overrides_env(self, monkeypatch):
        """CLI arguments should override environment variables."""
        monkeypatch.setenv("X_CUBE_AI_PATH", "/env/path")
        cfg = resolve_deploy_config(cli_args={"x_cube_ai_path": "/cli/path"}, config_path="/nonexistent")
        assert cfg.x_cube_ai_path == "/cli/path"

    def test_env_overrides_file(self, monkeypatch, tmp_path):
        """Environment variables should override config file values."""
        import json

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"x_cube_ai_path": "/file/path"}))
        monkeypatch.setenv("X_CUBE_AI_PATH", "/env/path")
        cfg = resolve_deploy_config(config_path=str(config_file))
        assert cfg.x_cube_ai_path == "/env/path"

    def test_defaults(self):
        """Without any source, defaults should be used."""
        cfg = resolve_deploy_config(config_path="/nonexistent")
        assert cfg.model_path == "checkpoints/best_model_quantized.tflite"
