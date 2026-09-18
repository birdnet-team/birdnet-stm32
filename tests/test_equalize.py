"""Tests for raw-frontend per-band equalization."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow required for equalization tests")

from birdnet_stm32.conversion.equalize import equalize_raw_frontend
from birdnet_stm32.models.frontend import AudioFrontendLayer


def _raw_model(mag_scale="pwl"):
    inputs = tf.keras.Input((2000, 1))
    frontend = AudioFrontendLayer(
        mode="raw",
        mel_bins=8,
        spec_width=8,
        sample_rate=8000,
        chunk_duration=0.25,
        mag_scale=mag_scale,
        name="audio_frontend",
    )
    features = tf.keras.layers.GlobalAveragePooling2D()(frontend(inputs))
    return tf.keras.Model(inputs, tf.keras.layers.Dense(3, activation="sigmoid")(features))


def _inputs(seed, count):
    rng = np.random.default_rng(seed)
    t = np.arange(2000) / 8000.0
    out = []
    for i in range(count):
        # Loud low tones against quiet broadband noise: one band dominates, as on real audio.
        x = 0.9 * np.sin(2 * np.pi * (300 + 20 * i) * t) + 0.02 * rng.standard_normal(2000)
        out.append(x.astype(np.float32).reshape(1, 2000, 1))
    return out


def _make_bands_uneven(model):
    """Push the BN statistics to a trained-like state so every stage has work to do."""
    fe = model.get_layer("audio_frontend")
    probe = tf.keras.Model(model.inputs, fe.output)
    for x in _inputs(9, 4):
        probe(x, training=True)
    gamma, beta, mean, var = fe.band_bn.get_weights()
    rng = np.random.default_rng(3)
    fe.band_bn.set_weights([gamma * rng.uniform(0.5, 2.0, gamma.shape), beta, mean, var + 1e-3])


def test_output_is_unchanged_and_bands_are_balanced():
    model = _raw_model()
    _make_bands_uneven(model)
    report = equalize_raw_frontend(model, _inputs(0, 16), _inputs(1, 4))
    assert report["max_abs_output_diff"] < 1e-4
    assert report["stages"] == ["fb", "hinge", "pwl_in"]
    for name in ("audio_frontend_fb_re", "audio_frontend_band_relu"):
        assert report["after"][name] < report["before"][name]
        assert report["after"][name] == pytest.approx(1.0, abs=0.05)


def test_single_stage_touches_only_the_filterbank():
    model = _raw_model()
    _make_bands_uneven(model)
    mag = model.get_layer("audio_frontend").mag_layer
    k0_before = mag._pwl_k0_dw.get_weights()[0].copy()  # noqa: SLF001
    report = equalize_raw_frontend(model, _inputs(0, 16), _inputs(1, 4), stages=["fb"])
    assert "fb_gain_range" in report and "pwl_in_gain_range" not in report
    np.testing.assert_array_equal(mag._pwl_k0_dw.get_weights()[0], k0_before)  # noqa: SLF001


@pytest.mark.parametrize("stages", [[], ["fb", "bogus"]])
def test_unknown_stages_are_rejected(stages):
    with pytest.raises(ValueError, match="stages"):
        equalize_raw_frontend(_raw_model(), _inputs(0, 2), _inputs(1, 1), stages=stages)


def test_requires_a_raw_pwl_frontend():
    with pytest.raises(ValueError, match="raw frontend"):
        equalize_raw_frontend(_raw_model(mag_scale="none"), _inputs(0, 2), _inputs(1, 1))
