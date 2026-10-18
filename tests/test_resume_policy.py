"""A resumed run keeps the dtype policy it was trained with (mixed precision stays mixed)."""

import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow required")

from birdnet_stm32.models.runners import load_keras_model
from birdnet_stm32.training.trainer import load_for_resume


def test_resume_keeps_mixed_precision_while_evaluation_loads_float32(tmp_path):
    previous = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        inputs = tf.keras.Input(shape=(8,))
        x = tf.keras.layers.Dense(4, name="hidden")(inputs)
        outputs = tf.keras.layers.Dense(2, activation="sigmoid", name="pred", dtype="float32")(x)
        path = tmp_path / "model.keras"
        tf.keras.Model(inputs, outputs).save(path)
    finally:
        tf.keras.mixed_precision.set_global_policy(previous)
    assert load_for_resume(str(path)).get_layer("hidden").dtype_policy.name == "mixed_float16"
    assert load_keras_model(str(path)).get_layer("hidden").dtype_policy.name == "float32"
