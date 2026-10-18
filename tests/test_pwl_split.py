"""The two-exposure PWL rewrite the INT8 conversion applies to the raw frontend."""

import numpy as np
import pytest
import tensorflow as tf

from birdnet_stm32.conversion.pwl_split import can_split, split_points, split_pwl
from birdnet_stm32.models.dscnn import build_dscnn_model


def _model(**kwargs):
    kwargs.setdefault("raw_exposure_gain", 16.0)
    return build_dscnn_model(
        num_mels=16,
        spec_width=128,
        sample_rate=16000,
        chunk_duration=1,
        embeddings_size=32,
        num_classes=3,
        audio_frontend="raw",
        alpha=0.25,
        stage_widths=(32, 64, 128, 256),
        **kwargs,
    )


def _perturb(model, seed=0):
    """Trained-looking frontend parameters: hinges at different points per band, negative slopes, BN statistics."""
    rng = np.random.default_rng(seed)
    f = model.get_layer("audio_frontend")
    pwl = f.mag_layer
    k0 = pwl._pwl_k0_dw.get_weights()[0]
    pwl._pwl_k0_dw.set_weights([rng.uniform(0.1, 1.3, k0.shape).astype(np.float32)])
    for s, k in zip(pwl._pwl_shift_dws, pwl._pwl_k_dws, strict=True):
        w, b = s.get_weights()
        s.set_weights([rng.uniform(0.7, 2.4, w.shape).astype(np.float32), rng.uniform(-5.0, 0.5, b.shape).astype(np.float32)])
        (kw,) = k.get_weights()
        k.set_weights([rng.uniform(-0.5, 2.4, kw.shape).astype(np.float32)])
    g, b, m, v = f.band_bn.get_weights()
    f.band_bn.set_weights([rng.uniform(0.5, 2.0, g.shape), rng.uniform(-0.5, 0.5, b.shape), rng.uniform(0.0, 1.0, m.shape),
                           rng.uniform(0.2, 2.0, v.shape)])


def test_split_is_float_exact():
    model = _model()
    _perturb(model)
    assert can_split(model)
    split = split_pwl(model)
    x = (np.random.default_rng(1).standard_normal((4, 16000, 1)) * 0.3).astype(np.float32)
    np.testing.assert_allclose(split.predict(x, verbose=0), model.predict(x, verbose=0), atol=1e-5)
    names = {layer.name for layer in split.layers}
    assert {"audio_frontend_split_mix_low", "audio_frontend_split_mix_high", "audio_frontend_split_sum"} <= names
    assert "audio_frontend" not in names
    assert split.get_layer("pred") is model.get_layer("pred")  # the backbone is reused, not copied


def test_split_points_reach_the_target_output():
    k0 = np.array([0.5, 1.0], np.float32).reshape(1, 1, 2, 1)
    shifts = [(np.ones((1, 1, 2, 1), np.float32), np.array([-1.0, -10.0], np.float32))] * 3
    ks = [np.ones((1, 1, 2, 1), np.float32)] * 3
    xt = split_points(k0, shifts, ks, yt=2.0)
    # band 0: 0.5 x + 3 relu(x - 1) = 2 at x = 10 / 7; band 1: 1.0 x = 2 at x = 2
    np.testing.assert_allclose(xt, [10 / 7, 2.0], atol=2e-3)


def test_only_the_release_frontend_is_split():
    assert not can_split(_model(raw_exposure_gain=1.0))  # one exposure: no mix convolution
    assert not can_split(_model(mag_scale="none"))
    with pytest.raises(ValueError, match="split_pwl"):
        split_pwl(_model(mag_scale="none"))
    model = _model()
    w, b = model.get_layer("audio_frontend").mag_layer._pwl_shift_dws[0].get_weights()
    model.get_layer("audio_frontend").mag_layer._pwl_shift_dws[0].set_weights([-w, b])
    assert not can_split(model)  # a falling hinge breaks the high path's rewrite


def test_converted_split_model_is_int8_with_float_io(tmp_path):
    from birdnet_stm32.conversion.quantize import convert_to_tflite

    model = _model()
    _perturb(model)
    samples = [[(np.random.default_rng(i).standard_normal((1, 16000, 1)) * 0.3).astype(np.float32)] for i in range(8)]
    convert_to_tflite(split_pwl(model), lambda: iter(samples), str(tmp_path / "split.tflite"))
    interp = tf.lite.Interpreter(model_path=str(tmp_path / "split.tflite"))
    interp.allocate_tensors()
    assert interp.get_input_details()[0]["dtype"] == np.float32
    names = [d["name"] for d in interp.get_tensor_details()]
    assert any("audio_frontend_split_low" in n for n in names) and any("audio_frontend_split_high" in n for n in names)
