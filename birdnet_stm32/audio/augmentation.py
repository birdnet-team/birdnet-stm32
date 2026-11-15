"""Data augmentation for audio and spectrograms.

Implements SpecAugment (frequency/time masking): ``apply_spec_augment`` on the loader's
spectrograms (librosa, hybrid), ``spec_augment_tf`` in the graph after the raw frontend,
which computes its spectrogram inside the model. Mixup works on whole batches in
TensorFlow ops: ``birdnet_stm32.data.generator.mixup_batch``.
"""

import numpy as np
import tensorflow as tf


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


def spec_augment_tf(
    spec: tf.Tensor,
    freq_mask_max: int = 8,
    time_mask_max: int = 25,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
) -> tf.Tensor:
    """SpecAugment in TensorFlow ops on a batch ``[B, F, T, C]``, each sample masked independently.

    The raw frontend computes its spectrogram inside the model, so the loader's
    ``apply_spec_augment`` never sees it (every raw model up to 2026-11 trained without
    SpecAugment). This runs on the frontend output during training only.

    Masked cells take their band's median over time in the same sample (per channel), not
    zero: the raw frontend maps silence to a band-dependent floor (median ~0.23 on
    N6-B0), so zeros would be a pattern no real input produces. Mask widths are drawn
    uniformly from ``[0, max)``, as in ``apply_spec_augment``.
    """
    shape = tf.shape(spec)
    b, f, t = shape[0], shape[1], shape[2]
    fill = tf.sort(spec, axis=2)[:, :, t // 2 : t // 2 + 1, :]  # [B, F, 1, C]: the band's median over time

    def _band_mask(n, length, max_width):
        pos = tf.range(length)[None, None, :]  # [1, 1, L]
        width = tf.random.uniform([b, n, 1], 0, tf.maximum(1, tf.minimum(max_width, length)), dtype=tf.int32)
        start = tf.cast(tf.random.uniform([b, n, 1]) * tf.cast(tf.maximum(1, length - width), tf.float32), tf.int32)
        return tf.reduce_any((pos >= start) & (pos < start + width), axis=1)  # [B, L]

    fmask = _band_mask(num_freq_masks, f, freq_mask_max)[:, :, None, None]
    tmask = _band_mask(num_time_masks, t, time_mask_max)[:, None, :, None]
    return tf.where(fmask | tmask, fill, spec)
