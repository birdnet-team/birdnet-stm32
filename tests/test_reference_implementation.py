"""reference/ is the specification; the package must compute what it says.

The reference implementation is written without the package so that it can be
read and ported on its own. These tests keep the two in step: on the committed
test signal, every model input the reference builds must match the one the
package's evaluation path builds, for both frontends.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from birdnet_stm32.evaluation.metrics import make_chunks_for_file

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("reference", ROOT / "reference" / "birdnet_tiny_reference.py")
ref = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ref)

SIGNAL = ROOT / "reference" / "vectors" / "test_signal.wav"
BASE = {"sample_rate": 24000, "chunk_duration": 2.5, "num_mels": 64, "fft_length": 512}


def test_test_signal_is_the_documented_one():
    audio, rate = sf.read(str(SIGNAL), dtype="int16")
    expected = np.clip(np.round(ref.make_test_signal() * 32767), -32768, 32767).astype(np.int16)
    assert rate == 24000
    assert np.abs(audio.astype(np.int32) - expected).max() <= 1


@pytest.mark.parametrize(
    "config",
    [
        {**BASE, "audio_frontend": "raw", "spec_width": 256},
        {**BASE, "audio_frontend": "hybrid", "spec_width": 384, "input_compression": "sqrt"},
        {**BASE, "audio_frontend": "hybrid", "spec_width": 256, "input_compression": "none"},
    ],
    ids=["raw", "hybrid-sqrt", "hybrid"],
)
def test_reference_inputs_match_the_package(config):
    audio = ref.read_audio(SIGNAL, config["sample_rate"])
    windows = ref.windows(audio, config["sample_rate"], config["chunk_duration"], config["chunk_duration"] / 2)
    mine = np.stack([ref.model_input(w, config) for w in windows])
    package = np.stack(
        make_chunks_for_file(
            str(SIGNAL), config, config["audio_frontend"], "pwl", config["fft_length"], config["chunk_duration"] / 2
        )
    )
    assert package.shape == mine.shape
    # Float32 rounding only: the package also normalizes the file as a whole first.
    assert np.abs(package - mine).max() < 1e-4


def test_window_starts_right_align_the_last_window():
    assert ref.window_starts(144000, 60000, 30000) == [0, 30000, 60000, 84000]
    assert ref.window_starts(90000, 60000, 30000) == [0, 30000]
    assert ref.window_starts(40000, 60000, 30000) == [0]
