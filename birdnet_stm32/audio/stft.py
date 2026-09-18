"""Reference implementation of the spectrogram inputs (hybrid and precomputed mel).

This module is the definition, not a convenience wrapper: the training, the
evaluation, the conversion calibration and the firmware (``firmware/Src/
audio_stft.c``, ``fft.c``, ``audio_mel.c``) all compute exactly this, and
``docs/dev/spectrogram-input.md`` specifies it step by step for anyone
re-implementing it on another device. It needs numpy; ``scipy.fft`` is used
when installed because it transforms float32 without widening to float64.

Arithmetic is float32 throughout, as on the device. The float64 result differs
by ~1e-7 relative, far below one INT8 input step.

Pipeline for one chunk of ``N`` samples, with ``W = spec_width`` frames:

1. ``hop = N // W``.
2. Pad ``n_fft // 2`` zeros on both sides (centered frames).
3. Frame ``t`` covers padded samples ``[t * hop, t * hop + n_fft)``, ``t < W``.
4. Multiply by a **periodic** Hann window of length ``n_fft``.
5. ``|rfft|``, keeping bins ``0 .. n_fft // 2 - 1`` (Nyquist dropped).
6. Precomputed mel only: multiply by the Slaney mel filterbank (below).
7. Optional compression: ``sqrt``, or ``log`` with a floor 80 dB below the
   chunk's peak.
8. Min-max normalize the whole chunk to ``[0, 1]`` with ``1e-10`` in the
   denominator.

Output layout is ``[rows, W]``: rows are frequency bins (hybrid, ``n_fft // 2``)
or mel bands (precomputed), columns are frames.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

try:  # float32-native and ~2x faster than numpy's float64 transform
    from scipy.fft import rfft as _rfft
except ImportError:  # pragma: no cover - numpy fallback, same values to ~1e-7
    from numpy.fft import rfft as _rfft

VALID_INPUT_COMPRESSIONS = ("none", "sqrt", "log")
# Dynamic range kept by "log" compression, below each chunk's peak magnitude.
LOG_FLOOR_DB = 80.0
# Lowest mel band edge of the precomputed frontend; its upper edge is Nyquist.
MEL_FMIN_HZ = 150.0

# Slaney mel scale: linear below 1 kHz (200/3 Hz per mel), logarithmic above
# (27 mels per factor 6.4). Identical to librosa's default (htk=False).
_MEL_F_SP = 200.0 / 3.0
_MEL_MIN_LOG_HZ = 1000.0
_MEL_MIN_LOG_MEL = _MEL_MIN_LOG_HZ / _MEL_F_SP  # 15.0
_MEL_LOGSTEP = np.log(6.4) / 27.0


@lru_cache(maxsize=8)
def hann_window(n_fft: int) -> np.ndarray:
    """Periodic Hann window, ``0.5 - 0.5 cos(2 pi n / n_fft)`` for ``n < n_fft``.

    Periodic (divide by ``n_fft``), not symmetric (``n_fft - 1``): the
    symmetric window is a different input by up to ~0.6% per sample.
    """
    n = np.arange(n_fft, dtype=np.float64)
    window = (0.5 - 0.5 * np.cos(2.0 * np.pi * n / n_fft)).astype(np.float32)
    window.setflags(write=False)
    return window


def stft_magnitude(audio: np.ndarray, n_fft: int, spec_width: int) -> np.ndarray:
    """Centered STFT magnitude, Nyquist dropped: ``[n_fft // 2, frames]`` float32.

    ``frames`` is ``spec_width`` whenever the chunk is long enough, which every
    fixed-length model chunk is; a shorter signal yields every complete frame.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    hop = len(audio) // spec_width
    if hop < 1:
        raise ValueError(f"{len(audio)} samples cannot hold {spec_width} frames")
    padded = np.pad(audio, n_fft // 2)
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop][:spec_width]
    spectrum = np.abs(_rfft(frames * hann_window(n_fft), axis=1))
    return np.ascontiguousarray(spectrum[:, : n_fft // 2].T, dtype=np.float32)


def hz_to_mel(hz):
    """Slaney mel of a frequency in Hz."""
    hz = np.asarray(hz, dtype=np.float64)
    return np.where(
        hz >= _MEL_MIN_LOG_HZ,
        _MEL_MIN_LOG_MEL + np.log(np.maximum(hz, _MEL_MIN_LOG_HZ) / _MEL_MIN_LOG_HZ) / _MEL_LOGSTEP,
        hz / _MEL_F_SP,
    )


def mel_to_hz(mel):
    """Frequency in Hz of a Slaney mel value."""
    mel = np.asarray(mel, dtype=np.float64)
    return np.where(
        mel >= _MEL_MIN_LOG_MEL,
        _MEL_MIN_LOG_HZ * np.exp(_MEL_LOGSTEP * (np.maximum(mel, _MEL_MIN_LOG_MEL) - _MEL_MIN_LOG_MEL)),
        mel * _MEL_F_SP,
    )


def mel_frequencies(n_points: int, fmin: float, fmax: float) -> np.ndarray:
    """``n_points`` frequencies equally spaced on the Slaney mel scale, in Hz."""
    return mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_points))


@lru_cache(maxsize=8)
def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Slaney-normalized triangular mel filters over bins ``0 .. n_fft // 2 - 1``.

    Returns ``[n_mels, n_fft // 2]`` float32. Band ``m`` rises linearly from
    edge ``m`` to edge ``m + 1`` and falls to edge ``m + 2`` of
    ``mel_frequencies(n_mels + 2, fmin, fmax)``, and is scaled by
    ``2 / (edge[m + 2] - edge[m])`` so every band has the same area.

    The Nyquist bin is excluded, as in the STFT. With ``fmax`` at Nyquist the
    top band's weight there is exactly zero, so nothing is lost.
    """
    edges = mel_frequencies(n_mels + 2, fmin, fmax)
    bin_hz = np.arange(n_fft // 2, dtype=np.float64) * sample_rate / n_fft
    lower, center, upper = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    rising = (bin_hz[None, :] - lower) / (center - lower)
    falling = (upper - bin_hz[None, :]) / (upper - center)
    weights = np.maximum(0.0, np.minimum(rising, falling)) * (2.0 / (upper - lower))
    weights = weights.astype(np.float32)
    weights.setflags(write=False)
    return weights


def compress(spectrogram: np.ndarray, compression: str) -> np.ndarray:
    """Input compression applied before normalization: ``none``, ``sqrt`` or ``log``."""
    if compression == "none":
        return spectrogram
    if compression == "sqrt":
        return np.sqrt(spectrogram)
    if compression == "log":
        peak = float(np.max(spectrogram)) if spectrogram.size else 0.0
        floor = max(peak, 1e-10) * 10.0 ** (-LOG_FLOOR_DB / 20.0)
        return np.log(np.maximum(spectrogram, np.float32(floor)))
    raise ValueError(f"Invalid input compression: '{compression}'. Valid options: {VALID_INPUT_COMPRESSIONS}")


def minmax_normalize(spectrogram: np.ndarray) -> np.ndarray:
    """``(S - min) / (max - min + 1e-10)`` over the whole chunk."""
    low, high = spectrogram.min(), spectrogram.max()
    return np.asarray((spectrogram - low) / (high - low + np.float32(1e-10)))


def spectrogram_input(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int,
    spec_width: int,
    n_mels: int = 0,
    compression: str = "none",
) -> np.ndarray:
    """The model input for one chunk: hybrid (``n_mels=0``) or precomputed mel.

    Returns ``[n_fft // 2 or n_mels, spec_width]`` float32 in ``[0, 1]``.
    """
    spectrogram = stft_magnitude(audio, n_fft, spec_width)
    if n_mels > 0:
        spectrogram = (
            mel_filterbank(int(sample_rate), int(n_fft), int(n_mels), MEL_FMIN_HZ, sample_rate / 2.0) @ spectrogram
        )
    return minmax_normalize(compress(spectrogram, compression)).astype(np.float32)
