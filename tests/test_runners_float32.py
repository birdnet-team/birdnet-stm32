"""Checkpoints trained with mixed precision load as float32 models (they must calibrate and convert)."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from birdnet_stm32.models.runners import load_keras_model  # noqa: E402


def test_mixed_precision_checkpoint_loads_in_float32(tmp_path):
    previous = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        inputs = tf.keras.Input((16, 8, 1))
        x = tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu")(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(3, activation="sigmoid", dtype="float32")(x)
        mixed = tf.keras.Model(inputs, outputs)
    finally:
        tf.keras.mixed_precision.set_global_policy(previous)
    path = tmp_path / "mixed.keras"
    mixed.save(path)

    loaded = load_keras_model(str(path))
    assert {layer.dtype_policy.name for layer in loaded.layers} == {"float32"}
    x = np.random.default_rng(0).random((2, 16, 8, 1)).astype(np.float32)
    np.testing.assert_allclose(loaded(x), np.asarray(mixed(x), np.float32), atol=2e-3)
    tf.lite.TFLiteConverter.from_keras_model(loaded).convert()  # float16 compute would be rejected here


def test_float32_checkpoint_is_returned_as_loaded(tmp_path):
    inputs = tf.keras.Input((4,))
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(2)(inputs))
    path = tmp_path / "f32.keras"
    model.save(path)
    loaded = load_keras_model(str(path))
    np.testing.assert_allclose(loaded.get_weights()[0], model.get_weights()[0])
