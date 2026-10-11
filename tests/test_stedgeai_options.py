"""Neural-ART compiler options follow the installed ST Edge AI Core version."""

import subprocess
from types import SimpleNamespace

import pytest

from birdnet_stm32.deploy import stedgeai


@pytest.fixture(autouse=True)
def _fresh_cache():
    stedgeai.core_version.cache_clear()
    yield
    stedgeai.core_version.cache_clear()


def _fake_version(monkeypatch, text):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=text))


@pytest.mark.parametrize(
    "text, version, epoch_controller",
    [
        ("ST Edge AI Core v2.2.0-20266 2adc00962\n   STM32CubeAI 10.2.0-RC1", (2, 2, 0), False),
        ("ST Edge AI Core v4.0.1-20581 7ed50de05\n   STM32CubeAI 12.0.1-RC2", (4, 0, 1), True),
        ("something else", (), False),
    ],
)
def test_epoch_controller_from_core_4(monkeypatch, text, version, epoch_controller):
    _fake_version(monkeypatch, text)
    assert stedgeai.core_version("/x/stedgeai") == version
    options = stedgeai.neural_art_options("/x/stedgeai")
    assert options[0] == "--st-neural-art"
    assert ("--enable-epoch-controller" in options) is epoch_controller


def test_unreadable_binary_gives_no_version(monkeypatch):
    def boom(*a, **k):
        raise OSError("not found")

    monkeypatch.setattr(subprocess, "run", boom)
    assert stedgeai.core_version("/missing/stedgeai") == ()


@pytest.mark.parametrize("layout", [("scripts",), ("Utilities", "scripts")])
def test_n6_loader_found_in_either_layout(tmp_path, layout):
    from birdnet_stm32.deploy.config import DeployConfig

    loader = tmp_path.joinpath(*layout, "N6_scripts", "n6_loader.py")
    loader.parent.mkdir(parents=True)
    loader.write_text("")
    cfg = DeployConfig(x_cube_ai_path=str(tmp_path))
    assert cfg.n6_loader_script == str(loader)
