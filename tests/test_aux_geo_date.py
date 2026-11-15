"""Location/date auxiliary head: target table, loader plumbing, wrapper loss and head fit."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from test_teacher_embeddings import DIM, TestLoaderAndTrainer, tiny_student

from birdnet_stm32.data.aux_targets import AuxTargets, build, day_angle, geo_unit
from birdnet_stm32.training.embedding_distillation import EmbeddingDistilledModel, fit_aux_head


def write_manifest(path):
    pd.DataFrame(
        {
            "sample_id": ["rec1", "rec2", "rec3", "rec4", "rec5"],
            "latitude": [42.0, np.nan, np.nan, 0.0, 95.0],
            "longitude": [-76.0, np.nan, np.nan, 0.0, 10.0],
            "event_date": ["2020-06-01", "26 Apr 1977", "", "2021-01-01", "bad"],
            "source": ["XC", "ML", "Soundscape:SSW", "XC", "XC"],
        }
    ).to_csv(path, index=False)
    return path


class TestTable:
    def test_targets_flags_and_site_fill(self, tmp_path):
        meta = build(write_manifest(tmp_path / "m.csv"), tmp_path / "aux", {"Soundscape:SSW": [42.4768, -76.4527]})
        assert meta["rows"] == 5 and meta["location"] == 2 and meta["date"] == 3
        t = AuxTargets(tmp_path / "aux")
        a, v = t.lookup("rec1")
        np.testing.assert_allclose(a[:3], geo_unit(np.array([42.0]), np.array([-76.0]))[0], atol=1e-3)
        np.testing.assert_allclose(a[3:], day_angle([153])[0], atol=1e-3)  # 2020-06-01 is day 153
        np.testing.assert_array_equal(v, [1, 1])
        _, v2 = t.lookup("rec2")  # no location, a date in Macaulay's "26 Apr 1977" form
        np.testing.assert_array_equal(v2, [0, 1])
        a3, v3 = t.lookup("rec3")  # single-site collection: location filled, no date
        np.testing.assert_array_equal(v3, [1, 0])
        assert np.isclose(np.linalg.norm(a3[:3]), 1.0, atol=1e-3)
        np.testing.assert_array_equal(t.lookup("rec4")[1], [0, 1])  # 0/0 is a missing location, not the equator
        np.testing.assert_array_equal(t.lookup("rec5")[1], [0, 0])  # latitude out of range, unparseable date
        a, v = t.lookup("unknown")
        assert not a.any() and not v.any() and "unknown" not in t and "rec1" in t


class TestWorkerStaysTensorFlowFree:
    def test_importing_the_worker_does_not_load_tensorflow(self):
        code = "import sys, birdnet_stm32.data.worker; sys.exit('tensorflow' in sys.modules)"
        assert subprocess.run([sys.executable, "-c", code]).returncode == 0


class TestLoader:
    @pytest.mark.parametrize("workers", [0, 2])
    def test_batches_carry_aux_targets_and_mixup_clears_their_flags(self, tmp_path, workers):
        aux = tmp_path / "aux"
        build(write_manifest(tmp_path / "m.csv"), aux)
        ds = TestLoaderAndTrainer()._dataset(
            tmp_path, mixup_alpha=0.0, mixup_probability=0.0, num_workers=workers, aux_table=str(aux)
        )
        x, (y, emb, valid, aux_t, aux_v) = next(iter(ds))
        assert aux_t.shape == (4, 5) and aux_v.shape == (4, 2)
        # rec1 has both, rec2 only a date.
        rows = {tuple(r) for r in aux_v.numpy().tolist()}
        assert rows and rows <= {(1.0, 1.0), (0.0, 1.0)}
        mixed = TestLoaderAndTrainer()._dataset(
            tmp_path / "m", mixup_alpha=0.5, mixup_probability=1.0, num_workers=workers, aux_table=str(aux)
        )
        *_, mixed_v = next(iter(mixed))[1]
        assert float(tf.reduce_sum(mixed_v)) < float(tf.reduce_sum(aux_v))

    def test_needs_teacher_embeddings(self, tmp_path):
        from birdnet_stm32.data.generator import load_dataset

        build(write_manifest(tmp_path / "m.csv"), tmp_path / "aux")
        with pytest.raises(ValueError, match="teacher_embeddings"):
            load_dataset(["x/a/rec1.wav"], ["a"], audio_frontend="raw", aux_table=str(tmp_path / "aux"))


class TestWrapper:
    def _data(self, rng, n=8):
        x = rng.random((n, 16, 32, 1)).astype(np.float32)
        y = np.eye(3, dtype=np.float32)[rng.integers(0, 3, n)]
        teacher = rng.random((n, DIM)).astype(np.float32)
        lat, lon = rng.uniform(-60, 60, n), rng.uniform(-180, 180, n)
        aux_t = np.concatenate([geo_unit(lat, lon), day_angle(rng.integers(1, 366, n))], axis=1).astype(np.float32)
        aux_v = np.ones((n, 2), np.float32)
        aux_v[-2:, 0] = 0
        return x, y, teacher, np.ones(n, np.float32), aux_t, aux_v

    def test_train_step_reports_and_raises_the_aux_cosines(self):
        tf.keras.utils.set_random_seed(0)
        student = tiny_student()
        wrapped = EmbeddingDistilledModel(student, DIM, 1.0, aux_weight=1.0)
        wrapped.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss="binary_crossentropy")
        x, y, teacher, valid, aux_t, aux_v = self._data(np.random.default_rng(0))
        data = tf.data.Dataset.from_tensors((x, (y, teacher, valid, aux_t, aux_v))).repeat()
        val = tf.data.Dataset.from_tensors((x, y)).repeat()
        first = wrapped.fit(data, steps_per_epoch=1, epochs=1, verbose=0, validation_data=val, validation_steps=1)
        first = first.history
        assert "aux_geo_cosine" in first and "aux_date_cosine" in first and "val_aux_geo_cosine" not in first
        last = wrapped.fit(data, steps_per_epoch=60, epochs=1, verbose=0).history
        assert last["aux_geo_cosine"][0] > first["aux_geo_cosine"][0]
        assert wrapped.aux_head.name not in [layer.name for layer in student.layers]
        assert len(student.outputs) == 1

    def test_without_aux_weight_there_is_no_head(self):
        assert EmbeddingDistilledModel(tiny_student(), DIM, 1.0).aux_head is None

    def test_head_fit_recovers_a_linear_map(self):
        wrapped = EmbeddingDistilledModel(tiny_student(), DIM, 1.0, aux_weight=1.0)
        rng = np.random.default_rng(0)
        x = rng.random((256, 16, 32, 1)).astype(np.float32)
        _, feats = wrapped.embedder(x, training=False)
        w = rng.standard_normal((feats.shape[-1], 5))
        aux_t = (np.asarray(feats, np.float64) @ w + 0.5).astype(np.float32)
        aux_v = np.ones((256, 2), np.float32)
        aux_v[::4, 1] = 0
        report = fit_aux_head(
            wrapped, [(x[i : i + 64], aux_t[i : i + 64], aux_v[i : i + 64]) for i in range(0, 256, 64)], ridges=(1e-6,)
        )
        assert report["geo"]["rows"] == 256 and report["date"]["rows"] == 192
        assert report["geo"]["cosine_holdout"] > 0.99 and report["date"]["cosine_holdout"] > 0.99
