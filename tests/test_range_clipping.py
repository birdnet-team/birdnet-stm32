"""Percentile activation ranges in the PTQ conversion."""

import numpy as np
import pytest
import tensorflow as tf

from birdnet_stm32.conversion.quantize import _calibrated_ranges, clip_activation_ranges, convert_to_tflite


def _model():
    x = tf.keras.Input((16,))
    h = tf.keras.layers.Dense(32, activation="relu", name="hidden")(x)
    y = tf.keras.layers.Dense(4, name="logits")(h)
    return tf.keras.Model(x, y)


def _samples(n=64, outlier=True):
    rng = np.random.default_rng(0)
    xs = [rng.normal(size=(1, 16)).astype(np.float32) for _ in range(n)]
    if outlier:
        xs[0] *= 50.0  # one rare loud input stretches every min/max range
    return [[x] for x in xs]


def _calibrated(model, samples):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: iter(samples)
    converter._experimental_new_quantizer = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    debugger = tf.lite.experimental.QuantizationDebugger(converter=converter, debug_dataset=lambda: iter(samples[:1]))
    return bytes(debugger.calibrated_model)


def test_ranges_only_shrink_and_io_keeps_min_max():
    model, samples = _model(), _samples()
    calibrated = _calibrated(model, samples)
    before = _calibrated_ranges(calibrated)
    clipped = clip_activation_ranges(calibrated, samples, percentile=99.0)
    after = _calibrated_ranges(clipped)
    interp = tf.lite.Interpreter(model_content=calibrated)
    io = {interp.get_input_details()[0]["index"], interp.get_output_details()[0]["index"]}
    assert before.keys() == after.keys()
    shrunk = 0
    for i, (lo, hi) in before.items():
        new_lo, new_hi = after[i]
        assert lo <= new_lo <= new_hi <= hi
        if i in io:
            assert (new_lo, new_hi) == (lo, hi)
        shrunk += (new_hi - new_lo) < 0.9 * (hi - lo)
    assert shrunk > 0  # the outlier's stretch is cut somewhere inside the graph


def test_convert_keeps_float_io_and_tracks_float(tmp_path):
    model, samples = _model(), _samples(outlier=False)
    data = convert_to_tflite(model, lambda: iter(samples), str(tmp_path / "m.tflite"))
    interp = tf.lite.Interpreter(model_content=data)
    interp.allocate_tensors()
    assert interp.get_input_details()[0]["dtype"] == np.float32
    assert interp.get_output_details()[0]["dtype"] == np.float32
    x = samples[1][0]
    interp.set_tensor(interp.get_input_details()[0]["index"], x)
    interp.invoke()
    out = interp.get_tensor(interp.get_output_details()[0]["index"])
    ref = model(x).numpy()
    cos = float(np.sum(out * ref) / (np.linalg.norm(out) * np.linalg.norm(ref)))
    assert cos > 0.98


def test_a_range_too_narrow_to_bin_is_left_at_min_max():
    from birdnet_stm32.conversion.quantize import _binnable

    assert _binnable(-1.0, 1.0)
    assert not _binnable(0.5, 0.5)
    # a saturated gate: a range too narrow for 8,192 distinct float32 edges, which np.histogram refuses
    assert not _binnable(0.9999, 1.0)
    assert _binnable(0.999, 1.0)
    with pytest.raises(ValueError):
        np.histogram(np.ones(1, np.float32), bins=8192, range=(np.float32(0.9999), np.float32(1.0)))


def test_frontend_tensors_take_their_own_percentile():
    from birdnet_stm32.conversion.quantize import (
        ACTIVATION_RANGE_PERCENTILE,
        FRONTEND_RANGE_PERCENTILE,
        range_percentile,
    )

    assert FRONTEND_RANGE_PERCENTILE < ACTIVATION_RANGE_PERCENTILE
    assert range_percentile("dscnn_audio_1/audio_frontend_1/audio_frontend_mag_1/pwl_add_1") == FRONTEND_RANGE_PERCENTILE
    assert range_percentile("dscnn_audio_1/block2a_expand_act/Relu6") == ACTIVATION_RANGE_PERCENTILE


def test_a_percentile_function_cuts_tensors_by_name():
    x = tf.keras.Input((16,))
    fe = tf.keras.layers.Dense(32, activation="relu", name="audio_frontend_dense")(x)
    h = tf.keras.layers.Dense(32, activation="relu", name="hidden")(fe)
    model = tf.keras.Model(x, tf.keras.layers.Dense(4, name="logits")(h))
    samples = _samples()
    calibrated = _calibrated(model, samples)
    uniform = _calibrated_ranges(clip_activation_ranges(calibrated, samples, percentile=100.0))
    by_name = _calibrated_ranges(
        clip_activation_ranges(calibrated, samples, percentile=lambda n: 99.0 if "audio_frontend" in n else 100.0)
    )
    names = {d["index"]: d["name"] for d in tf.lite.Interpreter(model_content=calibrated).get_tensor_details()}
    fe = [i for i in uniform if "audio_frontend" in names[i]]
    assert fe and all(by_name[i][1] < uniform[i][1] for i in fe)
    assert all(by_name[i] == uniform[i] for i in uniform if i not in fe)
