"""Data augmentation for audio and spectrograms.

Implements mixup (multi-source additive mixing for soundscape realism)
and SpecAugment (frequency/time masking).
"""

import numpy as np


def apply_mixup(
    batch_samples: np.ndarray,
    batch_labels: np.ndarray,
    alpha: float = 0.2,
    probability: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply realistic multi-source mixup to a batch of samples and labels.

        Emulates natural soundscapes with multiple birds vocalizing at the same
        time. Instead of a single Beta-distributed lambda that biases toward one
        source, this draws mixing gains from a Dirichlet distribution so each
        source contributes a meaningful proportion. Each mixed sample blends
        2–3 sources (randomly chosen), and labels are merged via element-wise
        max so all contributing species remain active in the target.

    Args:
        batch_samples: Input batch [B, ...].
        batch_labels: One-hot labels [B, C].
        alpha: Dirichlet concentration parameter.  Lower values produce more
            varied gain distributions; higher values produce more uniform
            mixing (all sources contribute equally).  ``alpha=0.5`` is a
            good default for bird soundscape emulation.
        probability: Fraction of the batch to apply mixup to.

    Returns:
        Tuple of (mixed_samples, mixed_labels) with same shapes as inputs.
    """
    if alpha <= 0 or probability <= 0:
        return batch_samples, batch_labels

    B = batch_samples.shape[0]
    num_mix = int(B * probability)
    if num_mix <= 0:
        return batch_samples, batch_labels

    mix_indices = np.random.choice(B, size=num_mix, replace=False)

    for idx in mix_indices:
        # Randomly pick 2 or 3 sources (including the original)
        n_sources = np.random.choice([2, 3])
        partners = np.random.choice(B, size=n_sources - 1, replace=False)
        source_indices = np.concatenate([[idx], partners])

        # Draw mixing gains from Dirichlet distribution
        gains = np.random.dirichlet([alpha] * n_sources).astype(np.float32)
        gains_shaped = gains.reshape((n_sources,) + (1,) * (batch_samples.ndim - 1))

        # Additive mix of audio
        batch_samples[idx] = np.sum(gains_shaped * batch_samples[source_indices], axis=0)

        batch_labels[idx] = np.maximum.reduce(batch_labels[source_indices])

    return batch_samples, batch_labels


def apply_time_mask(
    waveform: np.ndarray,
    sample_rate: int,
    num_masks: int = 8,
    max_width_ms: float = 30.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Zero random spans of a waveform: SpecAugment time masking, in the sample domain.

    The ``raw`` frontend never sees a spectrogram in the loader, so it cannot
    use :func:`apply_spec_augment` and has historically trained with mixup
    alone. Zeroing a span of samples is equivalent: the learned filterbank
    turns it into zeroed columns of the time axis, exactly what a spectrogram
    time mask produces.

    There is no waveform equivalent of *frequency* masking — removing a band
    needs a filter, not a mask, which is what random equalization does.

    Widths are given in milliseconds so the setting is independent of the frame
    rate. At 256 frames over 2.5 s one frame is ~9.8 ms, so the 30 ms default
    is about three frames: the thin-band regime that keeps narrow calls intact
    rather than erasing them.

    Args:
        waveform: 1D float32 waveform. Masking is applied after any
            normalization, so a mask cannot change the peak the signal was
            scaled by.
        sample_rate: Sampling rate (Hz).
        num_masks: Number of masks to apply.
        max_width_ms: Maximum width of each mask (milliseconds).
        rng: Optional generator, for tests that need a fixed draw.

    Returns:
        Masked copy of the waveform, same shape and dtype.
    """
    if num_masks <= 0 or max_width_ms <= 0:
        return waveform
    rng = rng if rng is not None else np.random.default_rng()
    out = waveform.copy()
    n = out.shape[0]
    max_width = max(1, int(sample_rate * max_width_ms / 1000.0))
    for _ in range(num_masks):
        w = int(rng.integers(0, max_width + 1))
        if w <= 0 or w >= n:
            continue
        start = int(rng.integers(0, n - w))
        out[start : start + w] = 0.0
    return out


def apply_spec_augment(
    spectrogram: np.ndarray,
    freq_mask_max: int = 8,
    time_mask_max: int = 25,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
) -> np.ndarray:
    """Apply SpecAugment (frequency and time masking) to a spectrogram.

    Zeroes out random contiguous bands along the frequency and time axes.
    Operates on a single spectrogram of shape [F, T] or [F, T, 1].

    Reference: Park et al., "SpecAugment", 2019.

    Args:
        spectrogram: Input spectrogram [F, T] or [F, T, 1].
        freq_mask_max: Maximum width of each frequency mask (bins).
        time_mask_max: Maximum width of each time mask (frames).
        num_freq_masks: Number of frequency masks to apply.
        num_time_masks: Number of time masks to apply.

    Returns:
        Augmented spectrogram with same shape as input.
    """
    spec = spectrogram.copy()
    squeeze = False
    if spec.ndim == 3 and spec.shape[-1] == 1:
        spec = spec[:, :, 0]
        squeeze = True

    F, T = spec.shape

    # Frequency masks
    for _ in range(num_freq_masks):
        f = np.random.randint(0, max(1, min(freq_mask_max, F)))
        f0 = np.random.randint(0, max(1, F - f))
        spec[f0 : f0 + f, :] = 0.0

    # Time masks
    for _ in range(num_time_masks):
        t = np.random.randint(0, max(1, min(time_mask_max, T)))
        t0 = np.random.randint(0, max(1, T - t))
        spec[:, t0 : t0 + t] = 0.0

    if squeeze:
        spec = spec[:, :, np.newaxis]
    return spec
