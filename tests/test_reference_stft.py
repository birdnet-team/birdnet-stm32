"""The reference spectrogram input against librosa, which defined it until 1.2.

librosa is kept only as a test oracle: the host computes birdnet_stm32.audio.stft,
the firmware reproduces that (tests/test_firmware_stft.py), and
docs/dev/spectrogram-input.md specifies it for re-implementation.
"""

import numpy as np
import pytest

from birdnet_stm32.audio import stft as ref
from birdnet_stm32.audio.activity import _short_time_energy

librosa = pytest.importorskip("librosa")


def _signal(seed: int, samples: int = 60000, sample_rate: int = 24000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(samples) / sample_rate
    tone = np.sin(2 * np.pi * rng.uniform(500, 9000) * t) * (t > rng.uniform(0, 1.5))
    return (0.05 * rng.standard_normal(samples) + 0.5 * tone).astype(np.float32)


def test_hann_window_is_periodic():
    np.testing.assert_allclose(ref.hann_window(512), librosa.filters.get_window("hann", 512, fftbins=True), atol=1e-7)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_stft_magnitude_matches_librosa(seed):
    x = _signal(seed)
    expected = np.abs(librosa.stft(x, n_fft=512, hop_length=len(x) // 256, win_length=512, window="hann"))[:256, :256]
    got = ref.stft_magnitude(x, 512, 256)
    assert got.shape == (256, 256) and got.dtype == np.float32
    assert np.abs(got - expected).max() <= 1e-5 * expected.max()


@pytest.mark.parametrize(("n_mels", "fmin"), [(64, 150.0), (96, 150.0), (40, 0.0)])
def test_mel_filterbank_matches_librosa_slaney(n_mels, fmin):
    full = librosa.filters.mel(sr=24000, n_fft=512, n_mels=n_mels, fmin=fmin, fmax=12000.0, htk=False, norm="slaney")
    np.testing.assert_allclose(ref.mel_filterbank(24000, 512, n_mels, fmin, 12000.0), full[:, :256], atol=1e-7)
    # Dropping Nyquist is exact when the top edge is Nyquist.
    assert np.abs(full[:, 256]).max() == 0.0


def test_mel_scale_round_trip_and_breakpoint():
    hz = np.array([0.0, 150.0, 999.0, 1000.0, 1001.0, 6400.0, 12000.0])
    np.testing.assert_allclose(ref.mel_to_hz(ref.hz_to_mel(hz)), hz, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(ref.hz_to_mel(hz), librosa.hz_to_mel(hz, htk=False), rtol=1e-12)
    assert ref.hz_to_mel(1000.0) == pytest.approx(15.0)


@pytest.mark.parametrize(("n_mels", "compression"), [(0, "none"), (0, "sqrt"), (64, "none"), (64, "log")])
def test_spectrogram_input_matches_librosa_pipeline(n_mels, compression):
    x = _signal(3)
    if n_mels:
        s = librosa.feature.melspectrogram(
            y=x,
            sr=24000,
            n_fft=512,
            hop_length=len(x) // 256,
            win_length=512,
            window="hann",
            n_mels=n_mels,
            power=1.0,
            fmin=150,
            fmax=12000,
            htk=False,
            norm="slaney",
        )[:, :256]
    else:
        s = np.abs(librosa.stft(x, n_fft=512, hop_length=len(x) // 256, win_length=512, window="hann"))[:256, :256]
    if compression == "sqrt":
        s = np.sqrt(s)
    elif compression == "log":
        s = np.log(np.maximum(s, s.max() * 10 ** (-ref.LOG_FLOOR_DB / 20)))
    expected = (s - s.min()) / (s.max() - s.min() + 1e-10)
    got = ref.spectrogram_input(x, 24000, 512, 256, n_mels=n_mels, compression=compression)
    assert got.dtype == np.float32
    assert np.abs(got - expected).max() < 1e-5


def test_short_time_energy_vectorized_matches_the_loop():
    x = _signal(5, samples=48123)
    frames = 1 + (len(x) - 1024) // 512
    loop = np.array([np.mean(x[i * 512 : i * 512 + 1024] ** 2) for i in range(frames)], dtype=np.float32)
    np.testing.assert_allclose(_short_time_energy(x), loop, rtol=1e-5)
    np.testing.assert_allclose(_short_time_energy(x[:500]), [np.mean(x[:500] ** 2)], rtol=1e-5)


@pytest.mark.parametrize(("n_mels", "compression"), [(0, "none"), (0, "sqrt"), (64, "none"), (64, "log")])
def test_documented_reference_code_matches_the_package(n_mels, compression):
    """docs/dev/spectrogram-input.md carries a copy-ready implementation; it must not drift."""
    import re
    from pathlib import Path

    page = (Path(__file__).resolve().parent.parent / "docs/dev/spectrogram-input.md").read_text()
    code = re.search(r"## Reference implementation.*?```python\n(.*?)```", page, re.S).group(1)
    namespace: dict = {}
    exec(code, namespace)  # noqa: S102 - trusted repository documentation
    x = _signal(6)
    x[:5000] = 0.0  # silence at the edge exercises padding and the log floor
    documented = namespace["spectrogram_input"](x, 24000, 512, 256, n_mels, compression)
    assert np.abs(documented - ref.spectrogram_input(x, 24000, 512, 256, n_mels, compression)).max() < 1e-5
