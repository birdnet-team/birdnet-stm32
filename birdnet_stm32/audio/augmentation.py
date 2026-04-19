"""Data augmentation for audio and spectrograms.

Implements mixup (uniform or Beta distribution) and SpecAugment (frequency/time masking).
"""

import numpy as np


def apply_mixup(
    batch_samples: np.ndarray,
    batch_labels: np.ndarray,
    alpha: float = 0.2,
    probability: float = 0.25,
    use_beta: bool = False,
    label_smoothing: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply mixup augmentation to a batch of samples and labels.

    A fraction of samples in the batch are mixed with random partners. Audio
    is blended by a weighted average; labels are merged via element-wise max
    (multi-label OR).

    Args:
        batch_samples: Input batch [B, ...].
        batch_labels: One-hot labels [B, C].
        alpha: Mixup strength. For uniform: range [alpha, 1 - alpha].
            For Beta: concentration parameter (Beta(alpha, alpha)).
        probability: Fraction of the batch to apply mixup to.
        use_beta: If True, sample lambda from Beta(alpha, alpha) instead of
            uniform. Beta distribution provides more diversity in mixing ratios.
        label_smoothing: If > 0, smooth labels after mixup by
            ``(1 - eps) * label + eps / C`` where ``eps = label_smoothing``.

    Returns:
        Tuple of (mixed_samples, mixed_labels) with same shapes as inputs.
    """
    if alpha <= 0 or probability <= 0:
        return batch_samples, batch_labels

    num_mix = int(batch_samples.shape[0] * probability)
    if num_mix <= 0:
        return batch_samples, batch_labels

    mix_indices = np.random.choice(batch_samples.shape[0], size=num_mix, replace=False)
    permuted_indices = np.random.permutation(batch_samples.shape[0])

    if use_beta:
        lam = np.random.beta(alpha, alpha, size=(num_mix,)).astype(np.float32)
    else:
        lam = np.random.uniform(alpha, 1 - alpha, size=(num_mix,)).astype(np.float32)

    lam_inp = lam.reshape((num_mix,) + (1,) * (batch_samples.ndim - 1))

    # Audio: weighted mix
    batch_samples[mix_indices] = (
        lam_inp * batch_samples[mix_indices] + (1 - lam_inp) * batch_samples[permuted_indices[mix_indices]]
    )
    # Labels: element-wise OR (multi-label union)
    batch_labels[mix_indices] = np.maximum(batch_labels[mix_indices], batch_labels[permuted_indices[mix_indices]])

    # Optional label smoothing
    if label_smoothing > 0 and batch_labels.shape[-1] > 1:
        C = batch_labels.shape[-1]
        batch_labels = (1.0 - label_smoothing) * batch_labels + label_smoothing / C

    return batch_samples, batch_labels


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
