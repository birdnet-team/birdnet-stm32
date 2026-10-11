"""Data augmentation for audio and spectrograms.

Implements SpecAugment (frequency/time masking). Mixup works on whole batches in
TensorFlow ops: ``birdnet_stm32.data.generator.mixup_batch``.
"""

import numpy as np


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
