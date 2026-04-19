"""Activity detection and signal-to-noise sorting for audio chunks.

Provides heuristics to rank audio chunks by signal content and filter out
low-activity (silent or noise-only) segments before training or evaluation.
"""

import numpy as np


def get_s2n_from_spectrogram(spectrogram: np.ndarray) -> float:
    """Compute a simple signal-to-noise proxy from a spectrogram (mean / std).

    Args:
        spectrogram: 2D spectrogram array.

    Returns:
        SNR-like scalar value.
    """
    signal = np.mean(spectrogram)
    noise = np.std(spectrogram)
    return signal / (noise + 1e-10)


def get_s2n_from_audio(audio: np.ndarray) -> float:
    """Compute a simple signal-to-noise proxy from raw audio (mean / std).

    Args:
        audio: 1D audio signal.

    Returns:
        SNR-like scalar value.
    """
    signal = np.mean(audio)
    noise = np.std(audio)
    return signal / (noise + 1e-10)


def sort_by_s2n(samples: list[np.ndarray], threshold: float = 0.1) -> list[np.ndarray]:
    """Sort samples by SNR proxy and filter out low-SNR ones. Keeps at least one.

    Args:
        samples: List of 2D spectrograms or 1D audio arrays.
        threshold: Minimum normalized SNR to keep (in [0, 1]).

    Returns:
        Sorted (descending by SNR) and filtered samples.
    """
    if len(samples[0].shape) == 2:
        s2n_values = np.array([get_s2n_from_spectrogram(spec) for spec in samples])
    elif len(samples[0].shape) == 1:
        s2n_values = np.array([get_s2n_from_audio(audio) for audio in samples])
    else:
        raise ValueError("Samples must be 1D or 2D arrays (raw audio or spectrograms).")

    s2n_values /= s2n_values.max() + 1e-10

    sorted_indices = np.argsort(s2n_values)[::-1]
    sorted_samples = [samples[i] for i in sorted_indices]

    filtered = [s for s, v in zip(sorted_samples, s2n_values[sorted_indices], strict=False) if v >= threshold]
    if len(filtered) == 0:
        filtered = [sorted_samples[0]]
    return filtered


def get_activity_ratio(x: np.ndarray, k: float = 2.0, max_active: float = 0.8, subsample: int = 512) -> float:
    """Compute the fraction of units above median + k * MAD, capped to avoid broadband noise.

    Args:
        x: 1D or 2D array (audio or spectrogram).
        k: MAD multiplier for threshold.
        max_active: Max allowed fraction of active units (returns 0.0 if exceeded).
        subsample: Number of points to use for median/MAD computation.

    Returns:
        Activity ratio in [0, 1].
    """
    x = np.abs(x)
    flat = x.ravel()
    n = flat.size
    if n > subsample:
        idx = np.linspace(0, n - 1, subsample, dtype=int)
        flat = flat[idx]
    med = np.median(flat)
    mad = np.median(np.abs(flat - med)) + 1e-10
    thresh = med + k * mad
    active = np.count_nonzero(x > thresh)
    total = x.size
    ratio = float(active) / float(total)
    if ratio > max_active:
        return 0.0
    return ratio


def sort_by_activity(samples: list[np.ndarray], threshold: float = 0.25) -> list[np.ndarray]:
    """Sort samples by activity ratio and filter low-activity ones. Keeps at least one.

    Args:
        samples: List of 1D or 2D arrays.
        threshold: Minimum activity ratio to keep.

    Returns:
        Sorted and filtered samples.
    """
    activity = np.array([get_activity_ratio(s) for s in samples])
    sorted_idx = np.argsort(activity)[::-1]
    sorted_samples = [samples[i] for i in sorted_idx]
    filtered = [s for s, a in zip(sorted_samples, activity[sorted_idx], strict=False) if a >= threshold]
    if not filtered:
        filtered = [sorted_samples[0]]
    return filtered


def pick_random_samples(
    samples: list[np.ndarray],
    num_samples: int = 1,
    pick_first: bool = False,
) -> list[np.ndarray] | np.ndarray:
    """Randomly select one or more samples from a list.

    When ``pick_first=True`` and ``num_samples > 1``, the first sample is
    always included and the remaining are drawn randomly from the rest.

    Args:
        samples: List of samples (spectrograms or raw audio).
        num_samples: Number of samples to select.
        pick_first: If True and num_samples == 1, always return the first sample.
            If True and num_samples > 1, include the first sample plus random picks.

    Returns:
        Selected samples. A list if num_samples > 1, otherwise a single ndarray.
    """
    if len(samples) == 0:
        return []
    if num_samples > len(samples):
        num_samples = len(samples)

    if pick_first:
        if num_samples == 1:
            return samples[0]
        # Always include first, randomly pick remaining from the rest
        rest_count = min(num_samples - 1, len(samples) - 1)
        if rest_count > 0:
            rest_indices = np.random.choice(len(samples) - 1, size=rest_count, replace=False) + 1
            return [samples[0]] + [samples[i] for i in rest_indices]
        return [samples[0]]

    indices = np.random.choice(len(samples), size=num_samples, replace=False)
    return [samples[i] for i in indices] if num_samples > 1 else samples[indices[0]]
