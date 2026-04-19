"""Spectrogram computation, magnitude scaling, and normalization.

Supports mel spectrograms, MFCC, linear STFT, and multiple magnitude compression
modes (none, pwl, pcen, db). All scaling is designed to be quantization-friendly
for INT8 deployment on the STM32N6 NPU.
"""

import librosa
import numpy as np


def normalize(S: np.ndarray) -> np.ndarray:
    """Normalize a spectrogram to [0, 1] per sample.

    Args:
        S: Spectrogram array.

    Returns:
        Normalized spectrogram, same shape as input.
    """
    return (S - S.min()) / (S.max() - S.min() + 1e-10)


def get_spectrogram_from_audio(
    audio: np.ndarray,
    sample_rate: int = 24000,
    n_fft: int = 512,
    mel_bins: int = 64,
    spec_width: int = 256,
    mag_scale: str = "none",
    mode: str = "mel",
    n_mfcc: int = 20,
) -> np.ndarray:
    """Compute a magnitude spectrogram with optional scaling and normalization.

    Modes:
        - 'mel': Standard mel spectrogram.
        - 'mfcc': Mel-frequency cepstral coefficients (mel -> DCT -> truncate).
        - 'log_mel': Log-scaled mel spectrogram (log1p, quantization-friendly).
        - 'linear': Linear STFT magnitude (when mel_bins <= 0).

    Behavior by mag_scale (applied only in 'mel' and 'linear' modes):
        - 'none': Magnitude mel (power=1.0), then normalize to [0, 1].
        - 'pcen': Magnitude mel, scale to 32-bit PCM range, librosa.pcen, normalize.
        - 'pwl': Magnitude mel, pre-normalize, piecewise compression, normalize.
        - 'db': Magnitude mel, amplitude_to_db(ref=max), normalize.

    Args:
        audio: 1D audio array (mono).
        sample_rate: Sampling rate (Hz).
        n_fft: FFT size for STFT.
        mel_bins: Number of mel bands, or <=0 for linear STFT bins (magnitude).
        spec_width: Target number of time frames (columns).
        mag_scale: 'none' | 'db' | 'pcen' | 'pwl'.
        mode: 'mel' | 'mfcc' | 'log_mel' | 'linear'.
        n_mfcc: Number of MFCC coefficients (only used when mode='mfcc').

    Returns:
        Spectrogram array (mel_bins or n_mfcc or fft_bins, spec_width), values in [0, 1].
    """
    hop_length = (len(audio) // spec_width) if spec_width > 0 else n_fft // 2

    if mode == "mfcc":
        S_mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window="hann",
            n_mels=mel_bins,
            power=2.0,
            fmin=150,
            fmax=sample_rate // 2,
            htk=False,
            norm="slaney",
        )
        S_log = librosa.power_to_db(S_mel, ref=np.max)
        S = librosa.feature.mfcc(
            S=S_log,
            n_mfcc=n_mfcc,
            norm="ortho",
        )
        S = S[:, :spec_width]
        return normalize(S)

    if mode == "log_mel":
        S = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window="hann",
            n_mels=mel_bins,
            power=1.0,
            fmin=150,
            fmax=sample_rate // 2,
            htk=False,
            norm="slaney",
        )
        S = S[:, :spec_width]
        S = np.log1p(S)
        return normalize(S)

    if mel_bins <= 0 or mode == "linear":
        S = np.abs(
            librosa.stft(
                y=audio,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window="hann",
            )
        )
    else:
        S = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window="hann",
            n_mels=mel_bins,
            power=1.0,
            fmin=150,
            fmax=sample_rate // 2,
            htk=False,
            norm="slaney",
        )

    # Ensure fixed width
    S = S[:, :spec_width]

    if mag_scale == "pcen":
        S = librosa.pcen(S * (2.0**31), sr=sample_rate, hop_length=hop_length, axis=1)

    elif mag_scale == "pwl":
        Smin, Smax = S.min(), S.max()
        Snorm = (S - Smin) / (Smax - Smin + 1e-10)
        t1, t2, t3 = 0.10, 0.35, 0.65
        k0, k1, k2, k3 = 0.40, 0.25, 0.15, 0.08
        relu = lambda z: np.maximum(z, 0.0)  # noqa: E731
        S = k0 * Snorm + k1 * relu(Snorm - t1) + k2 * relu(Snorm - t2) + k3 * relu(Snorm - t3)

    elif mag_scale == "db":
        S = librosa.amplitude_to_db(S, ref=np.max)

    return normalize(S)
