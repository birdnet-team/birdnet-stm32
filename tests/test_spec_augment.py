"""Unit tests for SpecAugment augmentation."""

import numpy as np

from birdnet_stm32.audio.augmentation import apply_spec_augment


class TestApplySpecAugment:
    """Tests for apply_spec_augment."""

    def test_shape_preserved_2d(self):
        """Output shape should match 2-D input shape."""
        spec = np.ones((64, 128), dtype=np.float32)
        out = apply_spec_augment(spec)
        assert out.shape == spec.shape

    def test_shape_preserved_3d(self):
        """Output shape should match 3-D input shape [F, T, 1]."""
        spec = np.ones((64, 128, 1), dtype=np.float32)
        out = apply_spec_augment(spec)
        assert out.shape == spec.shape

    def test_contains_zeros(self):
        """Masking should zero out at least some values."""
        np.random.seed(42)
        spec = np.ones((64, 128), dtype=np.float32)
        out = apply_spec_augment(spec, freq_mask_max=8, time_mask_max=25)
        assert np.any(out == 0.0)

    def test_does_not_modify_input(self):
        """apply_spec_augment should not modify the original array."""
        spec = np.ones((64, 128), dtype=np.float32)
        original = spec.copy()
        apply_spec_augment(spec)
        np.testing.assert_array_equal(spec, original)

    def test_zero_masks_noop(self):
        """With zero masks, output should equal input."""
        spec = np.ones((64, 128), dtype=np.float32)
        out = apply_spec_augment(spec, num_freq_masks=0, num_time_masks=0)
        np.testing.assert_array_equal(out, spec)
