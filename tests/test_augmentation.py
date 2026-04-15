"""Unit tests for mixup augmentation."""

import numpy as np

from birdnet_stm32.audio.augmentation import apply_mixup


class TestApplyMixup:
    """Tests for apply_mixup."""

    def test_shape_preserved(self):
        """Output shapes should match input shapes."""
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((16, 64, 128, 1)).astype(np.float32)
        labels = np.eye(10, dtype=np.float32)[rng.integers(0, 10, size=16)]
        out_s, out_l = apply_mixup(samples.copy(), labels.copy(), alpha=0.2, probability=0.5)
        assert out_s.shape == samples.shape
        assert out_l.shape == labels.shape

    def test_labels_are_multi_label_or(self):
        """Mixed labels should be the element-wise max (OR) of the pair."""
        samples = np.ones((4, 10), dtype=np.float32)
        labels = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
            ],
            dtype=np.float32,
        )
        np.random.seed(0)
        _, out_labels = apply_mixup(samples.copy(), labels.copy(), alpha=0.2, probability=1.0)
        # After mixup with OR, any mixed sample should have >= as many 1s as original
        for i in range(4):
            assert out_labels[i].sum() >= labels[i].sum() or np.allclose(out_labels[i], labels[i])

    def test_alpha_zero_noop(self):
        """alpha=0 should return inputs unchanged."""
        samples = np.ones((4, 10), dtype=np.float32)
        labels = np.eye(3, dtype=np.float32)[:3]
        labels = np.vstack([labels, labels[:1]])  # shape (4, 3)
        out_s, out_l = apply_mixup(samples.copy(), labels.copy(), alpha=0.0, probability=1.0)
        np.testing.assert_array_equal(out_s, samples)
        np.testing.assert_array_equal(out_l, labels)

    def test_probability_zero_noop(self):
        """probability=0 should return inputs unchanged."""
        rng = np.random.default_rng(1)
        samples = rng.standard_normal((8, 10)).astype(np.float32)
        labels = np.eye(5, dtype=np.float32)[: 8 % 5]
        labels = np.tile(labels, (2, 1))[:8]
        out_s, out_l = apply_mixup(samples.copy(), labels.copy(), alpha=0.3, probability=0.0)
        np.testing.assert_array_equal(out_s, samples)
        np.testing.assert_array_equal(out_l, labels)

    def test_values_in_range(self):
        """Mixed samples should stay within the range [min, max] of originals."""
        rng = np.random.default_rng(7)
        samples = rng.uniform(0, 1, (16, 10)).astype(np.float32)
        labels = np.eye(5, dtype=np.float32)[rng.integers(0, 5, 16)]
        out_s, _ = apply_mixup(samples.copy(), labels.copy(), alpha=0.2, probability=1.0)
        assert out_s.min() >= samples.min() - 1e-6
        assert out_s.max() <= samples.max() + 1e-6
