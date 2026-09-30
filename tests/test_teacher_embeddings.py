"""Teacher embedding distillation: cache lookup, loader plumbing, loss and wrapper."""

import numpy as np
import pytest
import tensorflow as tf
from test_teacher_targets import CLASSES, TestWorkerIntegration, write_cache

from birdnet_stm32.audio.augmentation import apply_mixup
from birdnet_stm32.data.teacher import TeacherTargets
from birdnet_stm32.models.dscnn import build_dscnn_model
from birdnet_stm32.training.embedding_distillation import (
    EmbeddingDistilledModel,
    fit_projector,
    masked_cosine_distance,
)

DIM = 6


def write_embedded_cache(root, emb=None, **kwargs):
    """write_cache plus an emb.npy row-aligned with its three default windows."""
    root = write_cache(root, **kwargs)
    if emb is None:
        emb = np.arange(3 * DIM, dtype=np.float32).reshape(3, DIM) + 1.0
    np.save(root / "emb.npy", np.asarray(emb, dtype=np.float16))
    return root


class TestEmbeddingLookup:
    def test_returns_the_row_of_the_nearest_window(self, tmp_path):
        t = TeacherTargets(write_embedded_cache(tmp_path / "c"), CLASSES, embeddings=True)
        assert t.embedding_dim == DIM
        # Chunk 1.25-3.75 s is centred on window 1 (1.25-4.25 s, centre 2.75 s), as for scores.
        np.testing.assert_array_equal(t.embedding("rec1", 1.25, 2.5), np.arange(DIM) + 1.0 + DIM)
        np.testing.assert_array_equal(t.lookup("rec1", 1.25, 2.5), np.float16([0.1, 0.8, 0.0]).astype(np.float32))

    def test_missing_recording_and_non_finite_rows_give_none(self, tmp_path):
        emb = np.ones((3, DIM), np.float32)
        emb[0, 2] = np.nan
        t = TeacherTargets(write_embedded_cache(tmp_path / "c", emb=emb), CLASSES, embeddings=True)
        assert t.embedding("nope", 0.0, 2.5) is None
        assert t.embedding("rec1", 0.0, 2.5) is None
        assert t.embedding("rec1", 2.5, 2.5) is not None

    def test_embeddings_must_be_requested_and_present(self, tmp_path):
        with pytest.raises(RuntimeError):
            TeacherTargets(write_embedded_cache(tmp_path / "c"), CLASSES).embedding("rec1", 0.0, 2.5)
        with pytest.raises(FileNotFoundError):
            TeacherTargets(write_cache(tmp_path / "plain"), CLASSES, embeddings=True)

    def test_row_count_must_match_the_scores(self, tmp_path):
        with pytest.raises(ValueError, match="rows"):
            TeacherTargets(write_embedded_cache(tmp_path / "c", emb=np.ones((2, DIM))), CLASSES, embeddings=True)


class TestWorkerEmbeddings:
    def _cfg(self, cache):
        cfg = TestWorkerIntegration()._cfg(cache, 0.5)
        cfg["teacher_embeddings"] = True
        return cfg

    def test_emits_the_embedding_and_a_valid_flag(self, tmp_path):
        from birdnet_stm32.data import worker

        cache = write_embedded_cache(tmp_path / "cache")
        path = TestWorkerIntegration()._write_audio(tmp_path)
        worker._init_worker(self._cfg(cache))
        try:
            ((sample, target, emb, valid),) = worker._process_file(str(path))
        finally:
            worker._init_worker(TestWorkerIntegration()._cfg(None, 0.0))
        np.testing.assert_array_equal(emb, np.arange(DIM) + 1.0)
        assert emb.dtype == np.float16 and valid == 1.0
        np.testing.assert_allclose(target, [0.95, 0.0, 0.0], atol=1e-3)  # blending unchanged

    def test_uncached_recording_is_marked_invalid(self, tmp_path):
        from birdnet_stm32.data import worker

        cache = write_embedded_cache(
            tmp_path / "cache", recordings={"other": ([0.0], [[0.0, 1.0, 0.0]])}, emb=np.ones((1, DIM))
        )
        path = TestWorkerIntegration()._write_audio(tmp_path)
        worker._init_worker(self._cfg(cache))
        try:
            ((_, target, emb, valid),) = worker._process_file(str(path))
        finally:
            worker._init_worker(TestWorkerIntegration()._cfg(None, 0.0))
        assert valid == 0.0 and not emb.any()
        np.testing.assert_array_equal(target, [1.0, 0.0, 0.0])

    def test_default_worker_output_is_unchanged(self, tmp_path):
        from birdnet_stm32.data import worker

        cache = write_embedded_cache(tmp_path / "cache")
        path = TestWorkerIntegration()._write_audio(tmp_path)
        worker._init_worker(TestWorkerIntegration()._cfg(cache, 0.5))
        try:
            (item,) = worker._process_file(str(path))
        finally:
            worker._init_worker(TestWorkerIntegration()._cfg(None, 0.0))
        assert len(item) == 2


class TestMixupMask:
    def test_marks_exactly_the_mixed_samples(self):
        np.random.seed(0)
        x = np.random.randn(16, 10).astype(np.float32)
        y = np.eye(16, dtype=np.float32)
        before = x.copy()
        _, _, mixed = apply_mixup(x, y, alpha=0.5, probability=0.25, return_mixed=True)
        assert mixed.sum() == 4
        changed = ~np.all(np.isclose(x, before), axis=1)
        assert not changed[~mixed].any()

    def test_default_return_is_unchanged(self):
        out = apply_mixup(np.zeros((4, 3), np.float32), np.zeros((4, 2), np.float32), probability=0.0)
        assert len(out) == 2


class TestMaskedCosine:
    def test_mean_distance_over_valid_rows_only(self):
        a = tf.constant([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        b = tf.constant([[2.0, 0.0], [1.0, 0.0], [5.0, 5.0]])
        loss, cos = masked_cosine_distance(a, b, tf.constant([1.0, 0.0, 1.0]))
        np.testing.assert_allclose(cos.numpy(), [1.0, 0.0, 1.0], atol=1e-6)
        assert abs(float(loss)) < 1e-6
        loss, _ = masked_cosine_distance(a, b, tf.constant([0.0, 1.0, 0.0]))
        assert float(loss) == pytest.approx(1.0)

    def test_no_valid_rows_is_zero_not_nan(self):
        loss, _ = masked_cosine_distance(tf.ones((2, 3)), tf.ones((2, 3)), tf.zeros(2))
        assert float(loss) == 0.0


def tiny_student(num_classes=3):
    return build_dscnn_model(
        num_mels=16,
        spec_width=32,
        sample_rate=16000,
        chunk_duration=1,
        embeddings_size=16,
        num_classes=num_classes,
        audio_frontend="librosa",
        alpha=0.25,
        depth_multiplier=1,
        mag_scale="none",
    )


class TestEmbeddingDistilledModel:
    def test_is_the_student_when_called(self):
        student = tiny_student()
        wrapped = EmbeddingDistilledModel(student, DIM, 1.0)
        x = np.random.default_rng(0).random((2, 16, 32, 1)).astype(np.float32)
        np.testing.assert_allclose(wrapped(x, training=False), student(x, training=False), atol=1e-6)

    def test_trains_the_student_and_the_projector_once_each(self):
        student = tiny_student()
        wrapped = EmbeddingDistilledModel(student, DIM, 1.0)
        ids = [id(v) for v in wrapped.trainable_weights]
        assert len(ids) == len(set(ids))
        assert {id(v) for v in student.trainable_weights} | {id(v) for v in wrapped.projector.trainable_weights} == set(
            ids
        )

    def test_train_step_lowers_the_embedding_loss_and_reports_it(self):
        tf.keras.utils.set_random_seed(0)
        student = tiny_student()
        wrapped = EmbeddingDistilledModel(student, DIM, 1.0)
        wrapped.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss="binary_crossentropy")
        rng = np.random.default_rng(0)
        x = rng.random((8, 16, 32, 1)).astype(np.float32)
        y = np.eye(3, dtype=np.float32)[rng.integers(0, 3, 8)]
        teacher = rng.random((8, DIM)).astype(np.float32)
        valid = np.array([1, 1, 1, 1, 1, 1, 0, 0], np.float32)
        data = tf.data.Dataset.from_tensors((x, (y, teacher, valid))).repeat()
        val = tf.data.Dataset.from_tensors((x, y)).repeat()
        first = wrapped.fit(
            data, steps_per_epoch=1, epochs=1, verbose=0, validation_data=val, validation_steps=1
        ).history
        assert "val_teacher_cosine" not in first
        last = wrapped.fit(data, steps_per_epoch=30, epochs=1, verbose=0).history
        assert "teacher_cosine" in first
        assert last["teacher_cosine"][0] > first["teacher_cosine"][0]
        assert wrapped.projector.name not in [layer.name for layer in student.layers]

    def test_projector_fit_recovers_a_linear_map(self):
        student = tiny_student()
        wrapped = EmbeddingDistilledModel(student, DIM, 1.0)
        rng = np.random.default_rng(0)
        x = rng.random((256, 16, 32, 1)).astype(np.float32)
        _, feats = wrapped.embedder(x, training=False)
        w = rng.standard_normal((feats.shape[-1], DIM)).astype(np.float32)
        teacher = np.asarray(feats) @ w + 3.0
        valid = np.ones(256, np.float32)
        batches = [(x[i : i + 64], (None, teacher[i : i + 64], valid[i : i + 64])) for i in range(0, 256, 64)]
        fit = fit_projector(wrapped, batches, ridges=(1e-6, 1e-1))
        assert fit["cosine_holdout"] > 0.99 and fit["ridge"] == 1e-6 and fit["rows"] == 256


class TestLoaderAndTrainer:
    def _dataset(self, tmp_path, **kwargs):
        import soundfile as sf

        from birdnet_stm32.data.generator import load_dataset

        rng = np.random.default_rng(0)
        paths = []
        for label in ("a", "b"):
            d = tmp_path / "audio" / label
            d.mkdir(parents=True, exist_ok=True)
            p = d / "rec1.wav" if label == "a" else d / "rec2.wav"
            sf.write(p, (rng.standard_normal(int(24000 * 2.5)) * 0.1).astype(np.float32), 24000)
            paths.append(str(p))
        cache = write_embedded_cache(tmp_path / "cache")
        return load_dataset(
            paths,
            CLASSES,
            audio_frontend="raw",
            batch_size=4,
            num_workers=0,
            sample_rate=24000,
            chunk_duration=2.5,
            teacher_cache=str(cache),
            teacher_weight=0.5,
            teacher_embeddings=True,
            **kwargs,
        )

    def test_batches_carry_embeddings_and_mixup_clears_their_flag(self, tmp_path):
        ds = self._dataset(tmp_path, mixup_alpha=0.5, mixup_probability=0.5)
        x, (y, emb, valid) = next(iter(ds))
        assert x.shape == (4, 60000, 1) and y.shape == (4, 3) and emb.shape == (4, DIM) and valid.shape == (4,)
        # rec1 is cached, rec2 is not; two of four samples are mixed.
        assert float(tf.reduce_sum(valid)) <= 2.0
        unmixed = self._dataset(tmp_path / "u", mixup_alpha=0.0, mixup_probability=0.0)
        _, (_, emb, valid) = next(iter(unmixed))
        assert float(tf.reduce_sum(valid)) == 2.0
        np.testing.assert_array_equal(emb.numpy()[valid.numpy() > 0][0], np.arange(DIM) + 1.0)

    def test_train_model_checkpoints_the_student_without_the_projector(self, tmp_path):
        from birdnet_stm32.models.runners import load_keras_model
        from birdnet_stm32.training.trainer import train_model

        rng = np.random.default_rng(0)
        x = rng.random((8, 16, 32, 1)).astype(np.float32)
        y = np.eye(3, dtype=np.float32)[np.arange(8) % 3]
        teacher = rng.random((8, DIM)).astype(np.float32)
        train = tf.data.Dataset.from_tensors((x, (y, teacher, np.ones(8, np.float32)))).repeat()
        val = tf.data.Dataset.from_tensors((x, y)).repeat()
        student = tiny_student()
        path = str(tmp_path / "m.keras")
        train_model(
            student,
            train,
            val,
            epochs=2,
            steps_per_epoch=2,
            val_steps=1,
            checkpoint_path=path,
            training_wrapper=lambda m: EmbeddingDistilledModel(m, DIM, 1.0),
        )
        saved = load_keras_model(path)
        assert "teacher_projector" not in [layer.name for layer in saved.layers]
        assert saved.count_params() == student.count_params()
        np.testing.assert_allclose(saved(x, training=False), student(x, training=False), atol=1e-5)

    @pytest.mark.parametrize("workers", [0, 2])
    def test_sample_chunks_draws_one_training_chunk_per_file(self, tmp_path, workers):
        from birdnet_stm32.data.generator import sample_chunks

        self._dataset(tmp_path)  # writes the audio and the cache
        paths = sorted(str(p) for p in (tmp_path / "audio").rglob("*.wav"))
        chunks = sample_chunks(
            paths,
            CLASSES,
            audio_frontend="raw",
            num_workers=workers,
            sample_rate=24000,
            chunk_duration=2.5,
            teacher_cache=str(tmp_path / "cache"),
            teacher_weight=0.5,
            teacher_embeddings=True,
        )
        assert len(chunks) == 2 and all(len(c) == 4 for c in chunks)
        assert [float(c[3]) for c in chunks] == [1.0, 0.0]  # rec1 cached, rec2 not


class TestEmbeddingWeightDefault:
    """The embedding loss is part of the recipe: on whenever the cache can feed it."""

    def _args(self, monkeypatch, tmp_path, *extra):
        import sys

        from birdnet_stm32.cli import train

        monkeypatch.setattr(sys, "argv", ["train", "--data_path_train", str(tmp_path), *extra])
        return train.get_args()

    def test_on_by_default_with_an_embedding_cache(self, monkeypatch, tmp_path):
        from birdnet_stm32.cli.train import TEACHER_EMBEDDING_WEIGHT

        cache = write_embedded_cache(tmp_path / "c")
        assert self._args(monkeypatch, tmp_path, "--teacher_cache", str(cache)).teacher_embedding_weight == (
            TEACHER_EMBEDDING_WEIGHT
        )
        assert (
            self._args(
                monkeypatch, tmp_path, "--teacher_cache", str(cache), "--teacher_embedding_weight", "0"
            ).teacher_embedding_weight
            == 0.0
        )

    def test_off_without_embeddings_and_outside_float_training(self, monkeypatch, tmp_path):
        plain = write_cache(tmp_path / "plain")
        assert self._args(monkeypatch, tmp_path).teacher_embedding_weight == 0.0
        assert self._args(monkeypatch, tmp_path, "--teacher_cache", str(plain)).teacher_embedding_weight == 0.0
        cache = write_embedded_cache(tmp_path / "c")
        assert self._args(monkeypatch, tmp_path, "--teacher_cache", str(cache), "--qat").teacher_embedding_weight == 0.0

    def test_asking_for_it_without_embeddings_is_an_error(self, monkeypatch, tmp_path):
        plain = write_cache(tmp_path / "plain")
        with pytest.raises(SystemExit):
            self._args(monkeypatch, tmp_path, "--teacher_cache", str(plain), "--teacher_embedding_weight", "0.2")
