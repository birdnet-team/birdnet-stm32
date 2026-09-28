"""Add a second, compressing exposure to a trained raw model (the 1.7 raw recipe).

A raw model trained from scratch with ``raw_exposure_mode="compress"`` loses
float accuracy; one trained on the linear envelope and then switched keeps it
and gains the INT8 robustness (measured: WABAD INT8 recall 0.255 -> 0.309 on
the 1.6 raw model after a 20-epoch fine-tune). This module does the switch:

1. rebuild the model with a two-exposure, compressing frontend at gain G;
2. copy every weight: the bank keeps its filters as the low copy and gets them
   times G as the high copy, the smoother is duplicated per copy, the mix starts
   at low + 0.5 * high, and band_bn, the PWL, the stem and the backbone are
   copied unchanged;
3. normalize the bank so each band's low copy spans +-1 (this sets the knee);
4. re-estimate band_bn's statistics on the compressed envelope.

The result is not the source function -- the backbone now sees a compressed
input -- so it must be fine-tuned (``train --init_checkpoint``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import tensorflow as tf

from birdnet_stm32.conversion.equalize import normalize_exposure_bank
from birdnet_stm32.models.dscnn import build_dscnn_model

# Calibration windows whose single batch re-estimates band_bn (momentum 0).
BN_BATCH = 256


def add_exposure(
    model: tf.keras.Model, config: dict, tensors: Sequence[np.ndarray], gain: float = 16.0
) -> tuple[tf.keras.Model, dict]:
    """Return a compressing two-exposure copy of a trained one-exposure raw model.

    Args:
        model: Trained float model with a raw, fused, tap-split frontend.
        config: Its model config dict (architecture fields).
        tensors: Calibration inputs, each ``[1, samples, 1]`` (peak-normalized).
        gain: The high exposure's gain G (> 1).

    Returns:
        The new model and a report.

    Raises:
        ValueError: The source frontend cannot take a second exposure.
    """
    src = model.get_layer("audio_frontend")
    if src.mode != "raw" or src.raw_bank != "fused" or src.raw_split_axis != "taps" or src.exposures != 1:
        raise ValueError("add_exposure needs a one-exposure raw model with a fused, tap-split bank")
    if gain <= 1.0:
        raise ValueError(f"gain must be > 1, got {gain}")
    new = build_dscnn_model(
        num_mels=int(src.mel_bins),
        spec_width=int(src.spec_width),
        sample_rate=int(src.sample_rate),
        chunk_duration=float(src.chunk_duration),
        embeddings_size=int(config["embeddings_size"]),
        num_classes=int(model.output_shape[-1]),
        audio_frontend="raw",
        alpha=float(config["alpha"]),
        depth_multiplier=int(config.get("depth_multiplier", 1)),
        fft_length=int(src.fft_length),
        mag_scale=src.mag_scale,
        raw_magnitude=src.raw_magnitude,
        raw_overlap=int(src.raw_overlap),
        raw_bank=src.raw_bank,
        raw_split_axis=src.raw_split_axis,
        raw_exposure_gain=float(gain),
        raw_exposure_mode="compress",
        frontend_trainable=bool(config.get("frontend_trainable", False)),
        dropout_rate=float(config.get("dropout_rate", 0.5)),
        head_pooling=config.get("head_pooling", "gap"),
        dw_kernel_size=int(config.get("dw_kernel_size", 3)),
        stage_widths=config.get("stage_widths"),
    )
    fe = new.get_layer("audio_frontend")
    fe.set_full_filterbank(*src.full_filterbank())
    (smooth,) = src.band_smooth.get_weights()
    fe.band_smooth.set_weights([np.concatenate([smooth, smooth], axis=2)])
    fe.band_bn.set_weights(src.band_bn.get_weights())
    fe.mag_layer.set_weights(src.mag_layer.get_weights())
    copied = 0
    for layer in model.layers:
        if layer.name == "audio_frontend" or not layer.weights:
            continue
        new.get_layer(layer.name).set_weights(layer.get_weights())
        copied += 1

    report = {"gain": float(gain), "layers_copied": copied}
    report.update(normalize_exposure_bank(new, tensors))
    probe = tf.keras.Model(new.inputs, fe.output)
    momentum = fe.band_bn.momentum
    fe.band_bn.momentum = 0.0
    try:
        probe(np.concatenate(list(tensors[:BN_BATCH])), training=True)
    finally:
        fe.band_bn.momentum = momentum
    return new, report
