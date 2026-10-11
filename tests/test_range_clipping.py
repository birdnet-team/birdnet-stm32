"""Percentile activation ranges in the PTQ conversion."""

import numpy as np
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
