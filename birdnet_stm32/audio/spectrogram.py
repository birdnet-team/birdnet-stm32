"""Spectrogram model inputs for the hybrid and precomputed-mel frontends.

The arithmetic is defined in :mod:`birdnet_stm32.audio.stft`, which the firmware
reproduces and ``docs/dev/spectrogram-input.md`` specifies. This module adds the
host-only options (the host-side PWL curves and paired uncompressed output).
"""

import numpy as np

from birdnet_stm32.audio.stft import (
    LOG_FLOOR_DB,  # noqa: F401 - re-exported
    MEL_FMIN_HZ,
    VALID_INPUT_COMPRESSIONS,
    compress,
    mel_filterbank,
    minmax_normalize,
    stft_magnitude,
)


def normalize(S: np.ndarray) -> np.ndarray:
    """Normalize a spectrogram to [0, 1] per sample.

    Args:
        S: Spectrogram array.

    Returns:
        Normalized spectrogram, same shape as input.
    """
    return minmax_normalize(S)


def get_spectrogram_from_audio(
    audio: np.ndarray,
    sample_rate: int = 24000,
    n_fft: int = 512,
    mel_bins: int = 64,
    spec_width: int = 256,
    mag_scale: str = "none",
    mode: str = "mel",
    compression: str = "none",
    with_uncompressed: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute a magnitude spectrogram with optional scaling and normalization.

    Modes:
        - 'mel': Standard mel spectrogram.
        - 'linear': Linear STFT magnitude (when mel_bins <= 0).

    Behavior by mag_scale (applied only in 'mel' and 'linear' modes):
        - 'none': Magnitude mel (power=1.0), then normalize to [0, 1].
        - 'pwl': Magnitude mel, pre-normalize, piecewise compression, normalize.
    Args:
        audio: 1D audio array (mono).
        sample_rate: Sampling rate (Hz).
        n_fft: FFT size for STFT.
        mel_bins: Number of mel bands, or <=0 for linear STFT bins (magnitude).
        spec_width: Target number of time frames (columns).
        mag_scale: 'none' | 'pwl'.
        mode: 'mel' | 'linear'.
        compression: Fixed compression applied to the magnitude before
            ``mag_scale`` and normalization: 'none', 'sqrt', or 'log' (natural
            log with a floor ``LOG_FLOOR_DB`` below the chunk's peak). It runs
            where the spectrogram is computed -- the host, or the Cortex-M55 on
            device -- so the model's first INT8 tensor already holds compressed
            values. Compressing inside the graph cannot do that: the linear
            magnitude is rounded onto the INT8 grid first.
        with_uncompressed: Also return the same spectrogram computed with
            ``compression='none'``, so callers that rank chunks by content
            (activity-based selection) rank on the representation they always
            ranked on, and the compression changes only the model input.

    Returns:
        Spectrogram array (mel_bins or fft_bins, spec_width), values in [0, 1];
        a ``(compressed, uncompressed)`` pair when ``with_uncompressed``.
    """
    if mode not in ("mel", "linear"):
        raise ValueError(f"Invalid spectrogram mode: '{mode}'. Valid options: ('mel', 'linear')")
    if mag_scale not in ("none", "pwl"):
        raise ValueError(f"Invalid magnitude scale: '{mag_scale}'. Valid options: ('none', 'pwl')")
    if compression not in VALID_INPUT_COMPRESSIONS:
        raise ValueError(f"Invalid input compression: '{compression}'. Valid options: {VALID_INPUT_COMPRESSIONS}")

    rows = n_fft // 2 if (mel_bins <= 0 or mode == "linear") else mel_bins
    S = stft_magnitude(audio, n_fft, spec_width)
    if rows != n_fft // 2:
        S = mel_filterbank(int(sample_rate), int(n_fft), int(mel_bins), MEL_FMIN_HZ, sample_rate / 2.0) @ S

    if with_uncompressed:
        return _scale(S, mag_scale, compression), _scale(S, mag_scale, "none")
    return _scale(S, mag_scale, compression)


def _scale(S: np.ndarray, mag_scale: str, compression: str) -> np.ndarray:
    """Apply input compression, the host-side magnitude curve, and normalization."""
    S = compress(S, compression)

    if mag_scale == "pwl":
        Smin, Smax = S.min(), S.max()
        Snorm = (S - Smin) / (Smax - Smin + 1e-10)
        t1, t2, t3 = 0.10, 0.35, 0.65
        # The in-graph layer's initial curve.
        k0, k1, k2, k3 = 0.40, 0.25, 0.15, 0.08
        relu = lambda z: np.maximum(z, 0.0)  # noqa: E731
        S = k0 * Snorm + k1 * relu(Snorm - t1) + k2 * relu(Snorm - t2) + k3 * relu(Snorm - t3)

    return normalize(S)
