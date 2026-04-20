"""Activity detection and signal-to-noise sorting for audio chunks.

Provides heuristics to rank audio chunks by signal content and filter out
low-activity (silent or noise-only) segments before training or evaluation.
Includes a smart crop strategy for weakly-labeled long recordings that
identifies the most salient segments using short-time energy analysis.
"""

import numpy as np


def _short_time_energy(audio: np.ndarray, frame_length: int = 1024, hop_length: int = 512) -> np.ndarray:
    """Compute short-time energy (STE) for an audio signal.

    Args:
        audio: 1D audio signal.
        frame_length: Analysis frame length in samples.
        hop_length: Hop between frames in samples.

    Returns:
        1D array of per-frame energy values.
    """
    n = audio.shape[0]
    n_frames = max(1, 1 + (n - frame_length) // hop_length)
    energy = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start : start + frame_length]
        energy[i] = np.mean(frame**2)
    return energy


def smart_crop(
    audio: np.ndarray,
    sample_rate: int,
    chunk_duration: float,
    max_chunks: int = 5,
    energy_percentile: float = 75.0,
) -> list[np.ndarray]:
    """Extract the most salient chunks from a long audio recording.

    Uses short-time energy (STE) to identify regions with the highest
    vocal activity.  This is critical for weakly-labeled recordings where
    a long file may contain sparse vocalizations mixed with silence or
    background noise.

    Strategy:
        1. Compute STE profile over the entire recording.
        2. Set a threshold at the given percentile of the energy distribution.
        3. Find contiguous regions above the threshold.
        4. For each region, extract one chunk centered on the energy peak.
        5. Return up to *max_chunks* chunks, ranked by peak energy.

    Args:
        audio: 1D float32 audio signal (mono, pre-normalized).
        sample_rate: Sampling rate (Hz).
        chunk_duration: Desired chunk length (seconds).
        max_chunks: Maximum number of chunks to return.
        energy_percentile: Percentile of STE used as activity threshold
            (higher = stricter, keeps only the loudest regions).

    Returns:
        List of 1D float32 audio chunks, sorted by descending energy.
        Falls back to a single center crop if no salient region is found.
    """
    chunk_size = int(sample_rate * chunk_duration)
    n = audio.shape[0]

    if n <= chunk_size:
        # File shorter than one chunk — pad and return
        padded = np.pad(audio, (0, max(0, chunk_size - n)))
        return [padded[:chunk_size].astype(np.float32)]

    # Compute STE
    frame_len = min(1024, chunk_size // 4)
    hop = frame_len // 2
    ste = _short_time_energy(audio, frame_length=frame_len, hop_length=hop)

    if ste.max() < 1e-10:
        # Silent recording — return center crop
        mid = n // 2
        start = max(0, mid - chunk_size // 2)
        return [audio[start : start + chunk_size].astype(np.float32)]

    threshold = np.percentile(ste, energy_percentile)
    above = ste >= threshold

    # Find contiguous active regions
    regions: list[tuple[int, int]] = []
    in_region = False
    region_start = 0
    for i, val in enumerate(above):
        if val and not in_region:
            region_start = i
            in_region = True
        elif not val and in_region:
            regions.append((region_start, i))
            in_region = False
    if in_region:
        regions.append((region_start, len(above)))

    if not regions:
        # Fallback: center crop
        mid = n // 2
        start = max(0, mid - chunk_size // 2)
        return [audio[start : start + chunk_size].astype(np.float32)]

    # For each region, find the peak energy frame and center a chunk there
    candidates: list[tuple[float, int]] = []
    for rs, re in regions:
        peak_frame = rs + int(np.argmax(ste[rs:re]))
        peak_sample = peak_frame * hop
        start = max(0, min(peak_sample - chunk_size // 2, n - chunk_size))
        peak_energy = float(ste[peak_frame])
        candidates.append((peak_energy, start))

    # Sort by energy (descending) and deduplicate overlapping chunks
    candidates.sort(key=lambda x: x[0], reverse=True)
    selected_starts: list[int] = []
    for _energy, start in candidates:
        # Skip if too close to an already-selected chunk
        if any(abs(start - s) < chunk_size // 2 for s in selected_starts):
            continue
        selected_starts.append(start)
        if len(selected_starts) >= max_chunks:
            break

    chunks = [audio[s : s + chunk_size].astype(np.float32) for s in selected_starts]
    return chunks if chunks else [audio[:chunk_size].astype(np.float32)]


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
