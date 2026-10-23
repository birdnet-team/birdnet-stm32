"""Online distillation from a larger frozen model (2.5): the KL term, the wrappers and the train flags."""

import numpy as np
import pytest
import tensorflow as tf
from test_teacher_embeddings import DIM, tiny_student, write_embedded_cache

from birdnet_stm32.models.dscnn import build_dscnn_model
from birdnet_stm32.training.distillation import OnlineDistilledModel, bernoulli_kl
from birdnet_stm32.training.embedding_distillation import EmbeddingDistilledModel


def tiny_teacher(num_classes=3):
    """A wider model on the student's input: what 2.4 is to a 2.5 student."""
    return build_dscnn_model(
        num_mels=16,
        spec_width=32,
        sample_rate=16000,
        chunk_duration=1,
        embeddings_size=32,
        num_classes=num_classes,
        audio_frontend="librosa",
        alpha=0.5,
        depth_multiplier=1,
        mag_scale="none",
    )


def batch(n=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, 16, 32, 1)).astype(np.float32), np.eye(3, dtype=np.float32)[np.arange(n) % 3]


class TestBernoulliKL:
    def test_zero_when_equal_positive_otherwise(self):
        p = tf.constant([[0.1, 0.9, 0.5]])
        assert float(bernoulli_kl(p, p)) == pytest.approx(0.0, abs=1e-6)
        assert float(bernoulli_kl(p, tf.constant([[0.5, 0.5, 0.5]]))) > 0.05

    def test_float16_inputs_are_computed_in_float32(self):
        p, q = tf.constant([[0.2, 0.7]]), tf.constant([[0.3, 0.6]])
        assert bernoulli_kl(tf.cast(p, tf.float16), tf.cast(q, tf.float16)).dtype == tf.float32
        assert float(bernoulli_kl(tf.cast(p, tf.float16), q)) == pytest.approx(
            float(bernoulli_kl(p, q)), rel=1e-2
        )  # fp16 rounds the inputs


class TestOnlineDistilledModel:
    def test_is_the_student_when_called(self):
        student = tiny_student()
        wrapped = OnlineDistilledModel(student, tiny_teacher(), 1.0)
        x, _ = batch(2)
        np.testing.assert_allclose(wrapped(x, training=False), student(x, training=False), atol=1e-6)

    def test_trains_only_the_student(self):
        student, teacher = tiny_student(), tiny_teacher()
        wrapped = OnlineDistilledModel(student, teacher, 1.0)
        assert {id(v) for v in wrapped.trainable_weights} == {id(v) for v in student.trainable_weights}
        assert not teacher.trainable

    def test_train_step_moves_the_student_toward_the_teacher(self):
        tf.keras.utils.set_random_seed(0)
        student, teacher = tiny_student(), tiny_teacher()
        before = [v.numpy().copy() for v in teacher.weights]
        wrapped = OnlineDistilledModel(student, teacher, 1.0)
        wrapped.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss="binary_crossentropy")
        x, y = batch()
        data = tf.data.Dataset.from_tensors((x, y)).repeat()
        first = wrapped.fit(
            data, steps_per_epoch=1, epochs=1, verbose=0, validation_data=data, validation_steps=1
        ).history
        assert "distill_kl" in first and "val_distill_kl" not in first
        last = wrapped.fit(data, steps_per_epoch=40, epochs=1, verbose=0).history
        assert last["distill_kl"][0] < first["distill_kl"][0]
        for a, b in zip(before, teacher.weights, strict=True):
            np.testing.assert_array_equal(a, b.numpy())

    def test_rejects_a_teacher_on_another_input_or_class_count(self):
        other_input = build_dscnn_model(
            num_mels=16,
            spec_width=48,
            sample_rate=16000,
            chunk_duration=1,
            embeddings_size=16,
            num_classes=3,
            audio_frontend="librosa",
            alpha=0.25,
            depth_multiplier=1,
            mag_scale="none",
        )
        with pytest.raises(ValueError, match="input"):
            OnlineDistilledModel(tiny_student(), other_input, 1.0)
        with pytest.raises(ValueError, match="classes"):
            OnlineDistilledModel(tiny_student(), tiny_teacher(num_classes=4), 1.0)
        with pytest.raises(ValueError, match="positive"):
            OnlineDistilledModel(tiny_student(), tiny_teacher(), 0.0)

    def test_train_model_checkpoints_the_student_without_the_teacher(self, tmp_path):
        from birdnet_stm32.models.runners import load_keras_model
        from birdnet_stm32.training.trainer import train_model

        x, y = batch()
        data = tf.data.Dataset.from_tensors((x, y)).repeat()
        student, teacher = tiny_student(), tiny_teacher()
        path = str(tmp_path / "m.keras")
        train_model(
            student,
            data,
            data,
            epochs=1,
            steps_per_epoch=2,
            val_steps=1,
            checkpoint_path=path,
            training_wrapper=lambda m: OnlineDistilledModel(m, teacher, 1.0),
        )
        saved = load_keras_model(path)
        assert saved.count_params() == student.count_params() < teacher.count_params()
        np.testing.assert_allclose(saved(x, training=False), student(x, training=False), atol=1e-5)


class TestWithEmbeddingDistillation:
    def test_both_terms_train_and_report(self):
        tf.keras.utils.set_random_seed(0)
        student, teacher = tiny_student(), tiny_teacher()
        wrapped = EmbeddingDistilledModel(student, DIM, 1.0, output_teacher=teacher, output_weight=1.0)
        assert {id(v) for v in wrapped.trainable_weights} == {id(v) for v in student.trainable_weights} | {
            id(v) for v in wrapped.projector.trainable_weights
        }
        wrapped.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss="binary_crossentropy")
        x, y = batch()
        emb = np.random.default_rng(1).random((8, DIM)).astype(np.float32)
        data = tf.data.Dataset.from_tensors((x, (y, emb, np.ones(8, np.float32)))).repeat()
        val = tf.data.Dataset.from_tensors((x, y)).repeat()
        first = wrapped.fit(
            data, steps_per_epoch=1, epochs=1, verbose=0, validation_data=val, validation_steps=1
        ).history
        assert {"distill_kl", "teacher_cosine"} <= set(first) and "val_distill_kl" not in first
        last = wrapped.fit(data, steps_per_epoch=40, epochs=1, verbose=0).history
        assert last["distill_kl"][0] < first["distill_kl"][0]

    def test_without_an_output_teacher_nothing_changes(self):
        wrapped = EmbeddingDistilledModel(tiny_student(), DIM, 1.0)
        assert wrapped.output_teacher is None and not hasattr(wrapped, "distill_metric")


class TestDistillFlags:
    def _args(self, monkeypatch, tmp_path, *extra):
        import sys

        from birdnet_stm32.cli import train

        monkeypatch.setattr(sys, "argv", ["train", "--data_path_train", str(tmp_path), *extra])
        return train.get_args()

    def test_default_weight_and_off_by_default(self, monkeypatch, tmp_path):
        ckpt = tmp_path / "t.keras"
        ckpt.write_bytes(b"")
        assert self._args(monkeypatch, tmp_path).distill_from is None
        assert self._args(monkeypatch, tmp_path, "--distill_from", str(ckpt)).distill_weight == 1.0
        assert (
            self._args(monkeypatch, tmp_path, "--distill_from", str(ckpt), "--distill_weight", "0.5").distill_weight
            == 0.5
        )

    def test_misuse_is_an_error(self, monkeypatch, tmp_path):
        ckpt = tmp_path / "t.keras"
        ckpt.write_bytes(b"")
        cache = write_embedded_cache(tmp_path / "c")
        for extra in (
            ["--distill_weight", "0.5"],
            ["--distill_from", str(tmp_path / "missing.keras")],
            ["--distill_from", str(ckpt), "--distill_weight", "0"],
            ["--distill_from", str(ckpt), "--qat", "--teacher_cache", str(cache)],
        ):
            with pytest.raises(SystemExit):
                self._args(monkeypatch, tmp_path, *extra)
