"""Unit tests for deployment configuration resolution."""

import json

import pytest

from birdnet_stm32.deploy.config import DeployConfig, resolve_deploy_config
from birdnet_stm32.training.config import ModelConfig


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


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_defaults(self):
        """Default ModelConfig should have sensible values."""
        cfg = ModelConfig()
        assert cfg.sample_rate == 24000
        assert cfg.audio_frontend == "hybrid"
        assert cfg.mag_scale == "pwl"
        assert cfg.num_classes == 0
        assert cfg.class_names == []

    def test_round_trip_json(self, tmp_path):
        """Save and load should produce identical config."""
        cfg = ModelConfig(
            sample_rate=16000,
            num_classes=3,
            class_names=["a", "b", "c"],
            alpha=0.5,
            use_se=True,
        )
        path = tmp_path / "cfg.json"
        cfg.save(path)
        loaded = ModelConfig.load(path)
        assert loaded == cfg

    def test_to_dict(self):
        """to_dict should return a plain dict."""
        cfg = ModelConfig(num_classes=2, class_names=["x", "y"])
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["num_classes"] == 2
        assert d["class_names"] == ["x", "y"]

    def test_from_dict_ignores_unknown_keys(self):
        """Legacy configs with extra keys should load without error."""
        data = {"sample_rate": 16000, "unknown_key": True, "num_classes": 0}
        cfg = ModelConfig.from_dict(data)
        assert cfg.sample_rate == 16000

    def test_from_dict_fills_defaults(self):
        """Missing keys should get default values."""
        cfg = ModelConfig.from_dict({"sample_rate": 48000})
        assert cfg.audio_frontend == "hybrid"
        assert cfg.num_classes == 0

    def test_validation_bad_sample_rate(self):
        """Negative sample_rate should raise ValueError."""
        with pytest.raises(ValueError, match="sample_rate"):
            ModelConfig(sample_rate=-1)

    def test_validation_bad_frontend(self):
        """Invalid audio_frontend should raise ValueError."""
        with pytest.raises(ValueError, match="audio_frontend"):
            ModelConfig(audio_frontend="invalid")

    def test_validation_class_names_mismatch(self):
        """class_names length != num_classes should raise ValueError."""
        with pytest.raises(ValueError, match="class_names"):
            ModelConfig(num_classes=2, class_names=["a"])

    def test_load_legacy_json(self, tmp_path):
        """Loading a legacy JSON (no new fields) should work."""
        legacy = {
            "sample_rate": 22050,
            "num_mels": 64,
            "spec_width": 256,
            "fft_length": 512,
            "chunk_duration": 3,
            "hop_length": 281,
            "audio_frontend": "hybrid",
            "mag_scale": "pwl",
            "embeddings_size": 256,
            "alpha": 1.0,
            "depth_multiplier": 1,
            "num_classes": 10,
            "class_names": [f"cls_{i}" for i in range(10)],
            "frontend_trainable": False,
        }
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps(legacy))
        cfg = ModelConfig.load(path)
        assert cfg.sample_rate == 22050
        assert cfg.num_classes == 10
        assert cfg.use_se is False  # default
