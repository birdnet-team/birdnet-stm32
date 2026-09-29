"""A two-exposure raw bank: high copy = G x low copy, clamped, emitted as two channels."""

import numpy as np
import pytest
import tensorflow as tf

from birdnet_stm32.conversion.equalize import equalize_raw_frontend, normalize_exposure_bank
from birdnet_stm32.models.dscnn import build_dscnn_model
from birdnet_stm32.models.frontend import AudioFrontendLayer

G = 16.0


def _model(gain=G, bank="fused", axis="taps", mode="channels"):
    return build_dscnn_model(
        num_mels=16,
        spec_width=64,
        sample_rate=8000,
        chunk_duration=1.0,
        audio_frontend="raw",
        num_classes=3,
        alpha=0.25,
        embeddings_size=16,
        raw_magnitude="l1",
        raw_bank=bank,
        raw_split_axis=axis,
        raw_exposure_gain=gain,
        raw_exposure_mode=mode,
    )


class _Grab:
    def __init__(self):
        self.values = {}

    def activation(self, name, x):
        self.values[name] = np.asarray(x)
        return x

    def kernel(self, layer, x):
        return layer(x)


def _frontend_tensors(model, x, clamp=True):
    fe = model.get_layer("audio_frontend")
    grab = _Grab()
    fe._exposure_clamp = clamp
    fe.set_quantization_hook(grab)
    try:
        out = tf.keras.Model(model.inputs, fe.output)(x, training=False).numpy()
    finally:
        fe.set_quantization_hook(None)
        fe._exposure_clamp = True
    return grab.values, out


def _audio(seed=0, n=2):
    return (0.3 * np.random.default_rng(seed).standard_normal((n, 8000, 1))).astype(np.float32)


def test_shapes_and_stem():
    model = _model()
    fe = model.get_layer("audio_frontend")
    assert fe.exposures == 2 and fe.band_channels == 32
    assert tuple(fe.output.shape[1:]) == (16, 64, 2)
    assert model.get_layer("stem_conv").kernel.shape[2] == 2
    assert model(_audio(), training=False).shape == (2, 3)


def test_high_copy_is_gain_times_low_until_the_clamp():
    model = _model()
    m = 16
    values, _ = _frontend_tensors(model, _audio(), clamp=False)
    re = values["audio_frontend_fb_re"]
    assert np.allclose(re[..., m:], G * re[..., :m], rtol=1e-4, atol=1e-5)
    values, _ = _frontend_tensors(model, _audio(), clamp=True)
    for name, v in values.items():
        if "_fb_" in name:
            assert np.abs(v).max() <= 1.0 + 1e-6, name


def test_kernel_layout_and_scaling_keep_the_ratio():
    fe = _model().get_layer("audio_frontend")
    re_k, im_k = fe.full_filterbank()
    fe.set_full_filterbank(2 * re_k, im_k)
    assert np.allclose(fe.full_filterbank()[0], 2 * re_k)
    fe.scale_filterbank_bands(np.linspace(0.5, 2.0, 16).astype(np.float32))
    w = fe.fb[0].get_weights()[0]
    assert np.allclose(w[..., 16:32], G * w[..., :16], rtol=1e-5)
    assert np.allclose(w[..., 48:64], G * w[..., 32:48], rtol=1e-5)


def test_config_round_trip_and_legacy_default():
    fe = _model().get_layer("audio_frontend")
    cfg = fe.get_config()
    assert cfg["raw_exposure_gain"] == G
    assert AudioFrontendLayer.from_config(cfg).exposures == 2
    cfg.pop("raw_exposure_gain")
    assert AudioFrontendLayer.from_config(cfg).exposures == 1


@pytest.mark.parametrize("bank,axis", [("pair", "taps"), ("fused", "channels")])
def test_needs_fused_bank_and_tap_split(bank, axis):
    with pytest.raises(ValueError, match="fused bank and a tap split"):
        _model(bank=bank, axis=axis)


def test_normalization_fills_the_grid_and_keeps_the_low_path():
    model = _model()
    x = _audio(1, 4)
    _, before = _frontend_tensors(model, x, clamp=False)
    report = normalize_exposure_bank(model, [x[i : i + 1] for i in range(4)])
    assert report["exposure_norm_range"][0] > 0
    values, after = _frontend_tensors(model, x, clamp=False)
    fb = [v for k, v in values.items() if "_fb_part_" in k or "_fb_sum_" in k]
    # Every band's low copy (re: channels 0-15, im: 32-47) peaks at exactly 1.
    band_peak = np.max(
        [np.maximum(np.abs(v[..., :16]).max((0, 1, 2)), np.abs(v[..., 32:48]).max((0, 1, 2))) for v in fb], 0
    )
    active = band_peak > 1e-6
    assert np.allclose(band_peak[active], 1.0, atol=1e-5)
    # The low exposure's path is unchanged (band_bn absorbed the gain), clamps on.
    _, clamped = _frontend_tensors(model, x, clamp=True)
    assert np.allclose(before[..., 0], after[..., 0], atol=1e-4)
    assert np.allclose(clamped[..., 0], after[..., 0], atol=1e-4)


def test_equalize_skips_the_bank_but_keeps_the_function():
    model = _model()
    x = _audio(2, 6)
    normalize_exposure_bank(model, [x[i : i + 1] for i in range(4)])
    report = equalize_raw_frontend(model, [x[i : i + 1] for i in range(4)], [x[4:6]])
    assert "fb_skipped" in report
    assert report["max_abs_output_diff"] < 1e-3


def test_compress_mode_is_one_knee_compressed_channel():
    model = _model(mode="compress")
    fe = model.get_layer("audio_frontend")
    assert (fe.band_channels, fe.calib_channels, fe.out_channels) == (32, 16, 1)
    assert tuple(fe.output.shape[1:]) == (16, 64, 1)
    assert model.get_layer("stem_conv").kernel.shape[2] == 1
    (w,) = fe.exposure_mix.get_weights()
    assert w.shape == (1, 1, 32, 16)
    assert np.allclose(np.diag(w[0, 0, :16]), 1.0) and np.allclose(np.diag(w[0, 0, 16:]), 0.5)
    assert np.count_nonzero(w) == 32
    x = _audio(3, 4)
    normalize_exposure_bank(model, [x[i : i + 1] for i in range(4)])
    report = equalize_raw_frontend(model, [x[i : i + 1] for i in range(3)], [x[3:4]])
    assert "fb_skipped" in report and report["max_abs_output_diff"] < 1e-3
    cfg = fe.get_config()
    assert AudioFrontendLayer.from_config(cfg).out_channels == 1


def test_compress_mode_needs_its_mode_name():
    with pytest.raises(ValueError, match="raw_exposure_mode"):
        _model(mode="bypass")


def test_equalize_without_pwl_keeps_only_the_bank_stage():
    model = build_dscnn_model(
        num_mels=16,
        spec_width=64,
        sample_rate=8000,
        chunk_duration=1.0,
        audio_frontend="raw",
        num_classes=3,
        alpha=0.25,
        embeddings_size=16,
        raw_magnitude="l1",
        mag_scale="none",
    )
    x = _audio(4, 5)
    report = equalize_raw_frontend(model, [x[i : i + 1] for i in range(4)], [x[4:5]])
    assert report["stages"] == ["fb"] and report["stages_dropped_without_pwl"] == ["hinge", "pwl_in"]
    assert report["max_abs_output_diff"] < 1e-3


def test_add_exposure_switches_a_trained_model_to_the_compressor(tmp_path):
    from birdnet_stm32.conversion.exposure import add_exposure
    from birdnet_stm32.models.runners import load_keras_model

    src = _model(gain=1.0)
    cfg = {"embeddings_size": 16, "alpha": 0.25, "dw_kernel_size": 3, "dropout_rate": 0.5}
    x = _audio(5, 4)
    new, report = add_exposure(src, cfg, [x[i : i + 1] for i in range(4)], gain=G)
    fe, sfe = new.get_layer("audio_frontend"), src.get_layer("audio_frontend")
    assert (fe.exposures, fe.raw_exposure_mode, fe.out_channels) == (2, "compress", 1)
    assert report["layers_copied"] > 5 and report["gain"] == G
    # The low copy is the source bank up to the per-band normalization.
    (lo_re, _), (src_re, _) = fe.full_filterbank(), sfe.full_filterbank()
    lo, ref = lo_re.reshape(-1, lo_re.shape[-1]), src_re.reshape(-1, src_re.shape[-1])
    scale = (lo * ref).sum(axis=0) / (ref * ref).sum(axis=0)
    assert np.all(scale > 0) and np.allclose(lo, ref * scale, atol=1e-6)
    for name in ("stem_conv", "emb_conv"):
        for a, b in zip(new.get_layer(name).get_weights(), src.get_layer(name).get_weights(), strict=True):
            assert np.array_equal(a, b)
    path = tmp_path / "m.keras"
    new.save(path)
    assert np.allclose(load_keras_model(str(path))(x, training=False), new(x, training=False), atol=1e-5)
    with pytest.raises(ValueError, match="one-exposure"):
        add_exposure(new, cfg, [x[:1]], gain=G)


def test_add_exposure_can_drop_the_pwl():
    from birdnet_stm32.conversion.exposure import add_exposure

    src = _model(gain=1.0)
    cfg = {"embeddings_size": 16, "alpha": 0.25, "dw_kernel_size": 3, "dropout_rate": 0.5}
    x = _audio(6, 3)
    new, report = add_exposure(src, cfg, [x[i : i + 1] for i in range(3)], gain=G, mag_scale="none")
    assert report["mag_scale"] == "none" and new.get_layer("audio_frontend").mag_scale == "none"
    assert new(x, training=False).shape == (3, 3)
