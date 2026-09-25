"""Tests for deploy configuration and stedgeai module."""

import os
from pathlib import Path

import pytest

from birdnet_stm32.deploy.board_test import _load_gen_app_config
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


def test_generated_firmware_drops_nyquist_bin():
    """Generated firmware geometry must match the hybrid model input."""
    generator = _load_gen_app_config()
    config = generator.generate_app_config_h(
        {
            "sample_rate": 24000,
            "chunk_duration": 2.5,
            "fft_length": 512,
            "hop_length": 234,
            "spec_width": 256,
            "num_mels": 64,
            "audio_frontend": "hybrid",
        },
        num_classes=25,
    )
    assert "#define APP_CHUNK_SAMPLES     60000" in config
    assert "#define APP_FFT_BINS          (APP_FFT_LENGTH / 2)" in config


def test_firmware_makefile_links_external_memory_drivers():
    """Every BSP function called during startup must have its implementation linked."""
    makefile = (Path(__file__).parents[1] / "firmware" / "Makefile").read_text()
    for source in (
        "stm32n6570_discovery_xspi.c",
        "Components/aps256xx/aps256xx.c",
        "Components/mx66uw1g45g/mx66uw1g45g.c",
    ):
        assert source in makefile


def _hybrid_cfg(**overrides):
    cfg = {
        "sample_rate": 24000,
        "chunk_duration": 2.5,
        "fft_length": 512,
        "hop_length": 234,
        "spec_width": 256,
        "num_mels": 64,
        "audio_frontend": "hybrid",
    }
    cfg.update(overrides)
    return cfg


def test_firmware_applies_the_sigmoid_for_a_logit_model():
    """The released models emit logits; the firmware owns the sigmoid.

    Without this the firmware would compare logits against APP_SCORE_THRESHOLD
    and print them as percentages, so a 0.5 threshold would silently become
    0.62 and every reported score would change meaning between releases.
    """
    generator = _load_gen_app_config()
    config = generator.generate_app_config_h(_hybrid_cfg(output_activation="logit"), num_classes=25)
    assert "#define APP_OUTPUT_LOGIT      1" in config


def test_a_probability_model_leaves_the_scores_alone():
    generator = _load_gen_app_config()
    for cfg in (_hybrid_cfg(output_activation="sigmoid"), _hybrid_cfg()):
        assert "#define APP_OUTPUT_LOGIT      0" in generator.generate_app_config_h(cfg, num_classes=25)


def test_an_unknown_output_activation_is_refused():
    """Better a build error than firmware that misreads its own model."""
    generator = _load_gen_app_config()
    with pytest.raises(ValueError, match="output_activation"):
        generator.generate_app_config_h(_hybrid_cfg(output_activation="softmax"), num_classes=25)
