"""Data augmentation for audio and spectrograms.

Currently implements mixup augmentation. Designed to be extended with
SpecAugment, time-stretch, pitch-shift, and other augmentation techniques.
"""

import numpy as np


def apply_mixup(
    batch_samples: np.ndarray,
    batch_labels: np.ndarray,
    alpha: float = 0.2,
    probability: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply mixup augmentation to a batch of samples and labels.

    A fraction of samples in the batch are mixed with random partners. Audio
    is blended by a weighted average; labels are merged via element-wise max
    (multi-label OR).

    Args:
        batch_samples: Input batch [B, ...].
        batch_labels: One-hot labels [B, C].
        alpha: Mixup strength (uniform [alpha, 1 - alpha]). 0 disables.
        probability: Fraction of the batch to apply mixup to.

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

    lam = np.random.uniform(alpha, 1 - alpha, size=(num_mix,))
    lam_inp = lam.reshape((num_mix,) + (1,) * (batch_samples.ndim - 1))

    # Audio: weighted mix
    batch_samples[mix_indices] = (
        lam_inp * batch_samples[mix_indices] + (1 - lam_inp) * batch_samples[permuted_indices[mix_indices]]
    )
    # Labels: element-wise OR (multi-label union)
    batch_labels[mix_indices] = np.maximum(batch_labels[mix_indices], batch_labels[permuted_indices[mix_indices]])

    return batch_samples, batch_labels
