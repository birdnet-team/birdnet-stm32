"""Resume after a stopped run: last checkpoint every epoch, best value carried over, rewind without a last checkpoint.

2026-11-09: the host-memory guard stopped a 50-epoch run five epochs after its best-only checkpoint; resuming needed a
hand rewind of the train state and the history. Stefan: "always save the best and last checkpoint so we can resume
model training should it fail".
"""

import csv
import json
import os

import tensorflow as tf
from test_online_distillation import batch, tiny_student

from birdnet_stm32.training.trainer import train_model


def _rows(path):
    with open(path, newline="") as f:
        return [r for r in csv.reader(f)][1:]


def _train(path, epochs, resume=False):
    x, y = batch()
    data = tf.data.Dataset.from_tensors((x, y)).repeat()
    return train_model(tiny_student(), data, data, epochs=epochs, steps_per_epoch=2, val_steps=1,
                       checkpoint_path=path, resume=resume, warmup_epochs=0)


def test_every_epoch_writes_the_last_checkpoint_and_the_state(tmp_path):
    path = str(tmp_path / "m.keras")
    _train(path, epochs=3)
    state = json.loads((tmp_path / "m_train_state.json").read_text())
    assert os.path.isfile(path) and os.path.isfile(tmp_path / "m_last.keras")
    assert state["epoch"] == state["last_epoch"] == 3
    assert 1 <= state["best_epoch"] <= 3 and isinstance(state["best"], float)


def test_resume_continues_from_the_last_epoch(tmp_path):
    path = str(tmp_path / "m.keras")
    _train(path, epochs=3)
    history = _train(path, epochs=5, resume=True)
    assert history.epoch == [3, 4]
    assert json.loads((tmp_path / "m_train_state.json").read_text())["epoch"] == 5
    assert [int(r[0]) for r in _rows(tmp_path / "m_history.csv")] == [1, 2, 3, 4, 5]


def test_resume_without_a_last_checkpoint_rewinds_to_the_best_epoch(tmp_path):
    path = str(tmp_path / "m.keras")
    _train(path, epochs=4)
    state = json.loads((tmp_path / "m_train_state.json").read_text())
    state.update(best_epoch=2)  # as if epochs 3-4 never improved on epoch 2
    (tmp_path / "m_train_state.json").write_text(json.dumps(state))
    os.remove(tmp_path / "m_last.keras")
    history = _train(path, epochs=4, resume=True)
    assert history.epoch == [2, 3]
    assert [int(r[0]) for r in _rows(tmp_path / "m_history.csv")] == [1, 2, 3, 4]
    assert os.path.isfile(str(tmp_path / "m_history.csv") + ".ep4_rewound")
