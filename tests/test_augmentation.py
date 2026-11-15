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


class TestMixupCooccurrence:
    def test_partners_only_from_classes_that_can_co_occur(self):
        # Two groups of classes that never co-occur: every mixed sample's labels stay inside its group.
        tf.random.set_seed(1)
        cls = np.arange(32) % 8
        labels = np.eye(8, dtype=np.float32)[cls]
        cooc = np.zeros((8, 8), np.float32)
        cooc[:4, :4] = cooc[4:, 4:] = 1.0
        samples = np.random.default_rng(1).standard_normal((32, 10)).astype(np.float32)
        for _ in range(20):
            _, lab, m = run(samples, labels, alpha=0.5, probability=1.0, cooccurrence=tf.constant(cooc))
            assert m.sum() == 32
            group = cls < 4
            assert np.all(lab[group][:, 4:] == 0) and np.all(lab[~group][:, :4] == 0)

    def test_unlabelled_samples_pair_with_anything(self):
        tf.random.set_seed(2)
        labels = np.zeros((16, 4), np.float32)
        labels[:8, 0] = 1.0  # class 0 co-occurs with nothing else
        cooc = np.eye(4, dtype=np.float32)
        samples = np.random.default_rng(2).standard_normal((16, 10)).astype(np.float32)
        seen = np.zeros(4)
        for _ in range(20):
            _, lab, _ = run(samples, labels, alpha=0.5, probability=1.0, cooccurrence=tf.constant(cooc))
            seen += lab[8:].sum(0)
        assert seen[0] > 0  # the unlabelled half picked up class-0 partners


class TestSpecAugmentTF:
    def test_masks_with_the_band_median_and_keeps_the_rest(self):
        from birdnet_stm32.audio.augmentation import spec_augment_tf

        tf.random.set_seed(3)
        rng = np.random.default_rng(3)
        x = (rng.random((4, 16, 40, 2)) + np.arange(16)[None, :, None, None]).astype(np.float32)
        y = spec_augment_tf(tf.constant(x), freq_mask_max=4, time_mask_max=8).numpy()
        assert y.shape == x.shape
        changed = y != x
        assert changed.any()
        median = np.sort(x, axis=2)[:, :, 20:21, :]
        fill = np.broadcast_to(median, x.shape)
        np.testing.assert_array_equal(y[changed], fill[changed])
        # at most 2 x 3 bands and 2 x 7 frames masked per sample
        for i in range(4):
            bands = changed[i].all(axis=(1, 2)).sum()
            assert bands <= 6

    def test_raw_frontend_masks_only_when_armed_and_training(self):
        from birdnet_stm32.models.frontend import AudioFrontendLayer

        layer = AudioFrontendLayer(mode="raw", mel_bins=32, spec_width=64, sample_rate=16000, chunk_duration=1.0)
        x = tf.constant(np.random.default_rng(4).standard_normal((2, 16000, 1)).astype(np.float32) * 0.1)
        plain = layer(x, training=True).numpy()  # training mode: the band BN uses batch statistics
        layer.train_spec_augment = (8, 16)
        assert "train_spec_augment" not in layer.get_config()
        tf.random.set_seed(4)
        masked = layer(x, training=True).numpy()
        changed = masked != plain
        assert changed.any()
        fill = np.broadcast_to(np.sort(plain, axis=2)[:, :, 32:33, :], plain.shape)
        np.testing.assert_allclose(masked[changed], fill[changed], rtol=1e-6)
        armed_inference = layer(x, training=False).numpy()
        layer.train_spec_augment = None
        np.testing.assert_array_equal(armed_inference, layer(x, training=False).numpy())
