"""The tap split computes the same filterbank as the channel split."""

import numpy as np
import pytest

from birdnet_stm32.models.dscnn import build_dscnn_model


def _model(axis, bank):
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
    )


@pytest.mark.parametrize("bank", ["fused", "pair"])
def test_tap_split_is_the_same_function(bank):
    chan, taps = _model("channels", bank), _model("taps", bank)
    old, new = chan.get_layer("audio_frontend"), taps.get_layer("audio_frontend")
    rng = np.random.default_rng(0)
    re_k, im_k = old.full_filterbank()
    old.set_full_filterbank(re_k + 0.01 * rng.standard_normal(re_k.shape), im_k)  # not just the seed
    new.set_full_filterbank(*old.full_filterbank())
    for name in ("band_smooth", "band_bn", "mag_layer"):
        getattr(new, name).set_weights(getattr(old, name).get_weights())
    for layer in taps.layers:
        if layer.name != "audio_frontend" and layer.weights:
            layer.set_weights(chan.get_layer(layer.name).get_weights())
    x = (0.3 * rng.standard_normal((2, 8000, 1))).astype(np.float32)
    assert np.allclose(chan(x, training=False).numpy(), taps(x, training=False).numpy(), atol=1e-5)
    assert len(new.filterbank_convs()) == len(old.filterbank_convs())


def test_full_filterbank_round_trips():
    layer = _model("taps", "fused").get_layer("audio_frontend")
    re_k, im_k = layer.full_filterbank()
    layer.set_full_filterbank(2 * re_k, im_k)
    assert np.allclose(layer.full_filterbank()[0], 2 * re_k)
