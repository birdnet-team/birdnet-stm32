"""Unit tests for batch mixup (TensorFlow ops, run inside tf.data)."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from birdnet_stm32.data.generator import mixup_batch  # noqa: E402


def run(samples, labels, **kwargs):
    s, lab, m = mixup_batch(tf.constant(samples), tf.constant(labels), **kwargs)
    return s.numpy(), lab.numpy(), m.numpy()


class TestMixupBatch:
    def test_shapes_are_kept(self):
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((16, 64, 128, 1)).astype(np.float32)
        labels = np.eye(10, dtype=np.float32)[rng.integers(0, 10, size=16)]
        s, lab, m = run(samples, labels, alpha=0.2, probability=0.5)
        assert s.shape == samples.shape and lab.shape == labels.shape and m.shape == (16,)

    def test_mixes_exactly_the_requested_share_and_leaves_the_rest(self):
        tf.random.set_seed(0)
        samples = np.random.default_rng(0).standard_normal((16, 10)).astype(np.float32)
        labels = np.eye(16, dtype=np.float32)
        s, _, m = run(samples, labels, alpha=0.5, probability=0.25)
        assert m.sum() == 4
        np.testing.assert_array_equal(s[m == 0], samples[m == 0])

    def test_mixed_labels_keep_every_source_species(self):
        tf.random.set_seed(1)
        labels = np.eye(8, dtype=np.float32)
        _, lab, m = run(np.ones((8, 4), np.float32), labels, alpha=0.2, probability=1.0)
        assert m.sum() == 8
        assert np.all(lab[np.arange(8), np.arange(8)] == 1.0)  # its own species stays
        assert np.all(lab.sum(axis=1) >= 1.0) and np.all(lab <= 1.0)

    def test_blends_stay_within_the_sources_range(self):
        tf.random.set_seed(2)
        rng = np.random.default_rng(7)
        samples = rng.uniform(0, 1, (16, 10)).astype(np.float32)
        labels = np.eye(5, dtype=np.float32)[rng.integers(0, 5, 16)]
        s, _, _ = run(samples, labels, alpha=0.2, probability=1.0)
        assert np.isfinite(s).all()
        assert s.min() >= samples.min() - 1e-5 and s.max() <= samples.max() + 1e-5

    @pytest.mark.parametrize("kwargs", [dict(alpha=0.0, probability=1.0), dict(alpha=0.3, probability=0.0)])
    def test_no_mixing_is_a_no_op(self, kwargs):
        samples = np.random.default_rng(1).standard_normal((8, 10)).astype(np.float32)
        labels = np.eye(8, dtype=np.float32)
        s, lab, m = run(samples, labels, **kwargs)
        np.testing.assert_array_equal(s, samples)
        np.testing.assert_array_equal(lab, labels)
        assert m.sum() == 0
