"""The firmware's hybrid STFT must compute the host's hybrid input.

A hybrid model is trained and evaluated on get_spectrogram_from_audio(); on the
board its input comes from firmware/Src/audio_stft.c. If the two differ, the
device runs the right model on the wrong input, and neither `stedgeai
validate` (which is fed host inputs) nor timings would show it. Before this
test existed they differed at cos 0.32: the firmware framed without centering
and skipped the host's min-max normalization.

The C sources are compiled natively, so this runs without a board.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from birdnet_stm32.audio.spectrogram import get_spectrogram_from_audio

FIRMWARE = Path(__file__).resolve().parent.parent / "firmware"

HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>
#include "audio_stft.h"
int main(int argc, char **argv) {
    uint32_t n = atoi(argv[3]), fft = atoi(argv[4]), hop = atoi(argv[5]), w = atoi(argv[6]);
    float *a = malloc(n * sizeof(float)), *o = calloc((fft / 2) * w, sizeof(float));
    FILE *f = fopen(argv[1], "rb");
    if (fread(a, sizeof(float), n, f) != n) return 2;
    fclose(f);
    stft_magnitude(a, n, fft, hop, w, o);
    spec_minmax_normalize(o, (fft / 2) * w);
    f = fopen(argv[2], "wb");
    fwrite(o, sizeof(float), (fft / 2) * w, f);
    fclose(f);
    return 0;
}
"""


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    compiler = shutil.which("gcc") or shutil.which("cc")
    if compiler is None:
        pytest.skip("no native C compiler")
    work = tmp_path_factory.mktemp("fwstft")
    (work / "harness.c").write_text(HARNESS)
    exe = work / "harness"
    subprocess.run(
        [
            compiler,
            "-O2",
            "-o",
            str(exe),
            str(work / "harness.c"),
            str(FIRMWARE / "Src" / "audio_stft.c"),
            str(FIRMWARE / "Src" / "fft.c"),
            f"-I{FIRMWARE / 'Inc'}",
            "-lm",
        ],
        check=True,
    )
    return exe, work


def _firmware_spectrogram(harness, audio, fft_length, hop, spec_width):
    exe, work = harness
    audio.astype(np.float32).tofile(work / "in.bin")
    subprocess.run(
        [
            str(exe),
            str(work / "in.bin"),
            str(work / "out.bin"),
            str(len(audio)),
            str(fft_length),
            str(hop),
            str(spec_width),
        ],
        check=True,
    )
    return np.fromfile(work / "out.bin", np.float32).reshape(fft_length // 2, spec_width)


def _bird_like(seed, samples=60000, sample_rate=24000, gain=0.3):
    """A few frequency-modulated whistles in noise, deliberately not peak-normalized."""
    rng = np.random.default_rng(seed)
    t = np.arange(samples) / sample_rate
    audio = 0.02 * rng.standard_normal(samples)
    for _ in range(4):
        onset, length = rng.uniform(0, 2.0), rng.uniform(0.1, 0.4)
        f0 = rng.uniform(2000, 7000)
        envelope = np.exp(-0.5 * ((t - onset - length / 2) / (length / 4)) ** 2)
        audio += envelope * np.sin(2 * np.pi * (f0 * t + 800 * np.sin(2 * np.pi * 5 * t) / (2 * np.pi * 5)))
    return (gain * audio / np.abs(audio).max()).astype(np.float32)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_firmware_stft_matches_the_host_hybrid_input(harness, seed):
    sample_rate, fft_length, spec_width = 24000, 512, 256
    audio = _bird_like(seed)
    hop = len(audio) // spec_width  # what the host and gen_app_config use
    host = get_spectrogram_from_audio(
        audio, sample_rate=sample_rate, n_fft=fft_length, mel_bins=-1, spec_width=spec_width
    )
    device = _firmware_spectrogram(harness, audio, fft_length, hop, spec_width)

    assert device.shape == host.shape
    cosine = float((host * device).sum() / (np.linalg.norm(host) * np.linalg.norm(device)))
    assert cosine > 0.9999
    # Well under one INT8 input step (1/255) anywhere in the spectrogram.
    assert np.abs(host - device).max() < 1e-3


def test_normalization_makes_the_input_independent_of_level(harness):
    """The host peak-normalizes audio; the firmware's min-max makes that moot."""
    audio = _bird_like(3)
    loud = _firmware_spectrogram(harness, audio, 512, 234, 256)
    quiet = _firmware_spectrogram(harness, audio * 0.05, 512, 234, 256)
    assert np.abs(loud - quiet).max() < 1e-4
