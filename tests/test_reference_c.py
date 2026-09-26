"""reference/c computes what the Python reference and the recorded vectors say.

Builds the C frontend natively against the vendored CMSIS-DSP (its portable
path) and checks every stage it prints for both frontends against the
committed vectors, as a porter would with --check-vectors --stages.
"""

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
C_DIR = ROOT / "reference" / "c"
VECTORS = ROOT / "reference" / "vectors"
SPEC = importlib.util.spec_from_file_location("reference", ROOT / "reference" / "birdnet_tiny_reference.py")
ref = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ref)


@pytest.fixture(scope="module")
def cli(tmp_path_factory):
    if shutil.which("make") is None or (shutil.which("gcc") or shutil.which("cc")) is None:
        pytest.skip("no native C toolchain")
    out = tmp_path_factory.mktemp("refc")
    subprocess.run(["make", "-s", "-C", str(C_DIR), "frontend_cli", f"CC={shutil.which('gcc') or 'cc'}"], check=True)
    exe = out / "frontend_cli"
    shutil.move(str(C_DIR / "frontend_cli"), exe)
    return exe


@pytest.mark.parametrize(
    "bundle, args",
    [
        ("BirdNET_Tiny_N6_USNE_90_V1.5_Raw", ["raw", "2.5"]),
        ("BirdNET_Tiny_N6_USNE_90_V1.5_Hybrid", ["hybrid", "2.5", "512", "384", "sqrt"]),
    ],
)
def test_c_frontend_matches_the_vectors(cli, tmp_path, bundle, args):
    stages = tmp_path / "stages.jsonl"
    with stages.open("w") as f:
        subprocess.run([str(cli), str(VECTORS / "test_signal.wav"), *args], check=True, stdout=f)
    assert ref.check_stages(VECTORS / f"{bundle}.json", stages)
