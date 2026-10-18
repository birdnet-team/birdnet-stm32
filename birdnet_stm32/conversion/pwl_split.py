"""Two-exposure PWL: a float-exact rewrite of the raw frontend's compressor for INT8 (2.2).

The raw frontend's band magnitude x = relu(bn(mix(smooth(|filterbank|)))) and
its PWL compressor y = k0 x + sum_i k_i relu(s_i x + b_i) are per-tensor INT8
tensors whose ranges are set by loud focal calls, while soundscape bins sit a
few INT8 steps above zero (2.2's N6-B0 on WABAD windows: band magnitude median
0.54 of a 0..16 range, PWL output median level 3 of 255). Their roundings
compound along the hinge chain. With every s_i > 0, per band b and a split
point xT_b where PWL_b reaches ``yt``,

    PWL(x) = PWL(min(x, xT)) + [PWL(x) - PWL(min(x, xT))]

exactly, and both terms run on the NPU with grids of their own:

* low:  x_s = relu6(g * bn(mix(.))), g = 6 / xT, is g * min(x, xT) on a [0, 6]
  grid; y_low = PWL(x_s / g), the PWL with k0 / g and s_i / g. Every tensor of
  this path spans the quiet range only.
* high: u = relu(bn(mix(.)) - xT); y_high = k0 u + sum_i k_i relu(s_i u + c_i),
  c_i = b_i + s_i xT where the hinge t_i = -b_i / s_i >= xT, else 0 (that hinge
  is linear above xT). y_high is exactly 0 on quiet bins.

y = y_low + y_high feeds the stem as before. The mix convolution and the PWL
are duplicated; the filterbank and the smoother are shared. On 2.2's N6-B0 over
61 WABAD sites, INT8 window AUPRC 0.4366 -> 0.4409 (float 0.4653) at 34.6 ms
on the board (33.7 without), every op still on the NPU.

The rewrite only feeds the INT8 conversion: the float model, its checkpoint and
the ONNX export are unchanged, and the converted model is validated against the
float model as always.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

SPLIT_YT = 2.0  # the PWL output where each band's low path ends (bench: 2 and 4 alike, 1 worse)
_X_MAX = 64.0  # band magnitudes searched for the split point (2.2's N6-B0: max 38 on calibration audio)


def can_split(model: tf.keras.Model) -> bool:
    """Whether the model has the frontend this rewrite targets: raw, fused bank, compress mix, 3-hinge PWL, s_i > 0."""
    try:
        frontend = model.get_layer("audio_frontend")
    except ValueError:
        return False
    if getattr(frontend, "mode", None) != "raw" or getattr(frontend, "raw_bank", None) != "fused":
        return False
    if getattr(frontend, "exposure_mix", None) is None or getattr(frontend, "out_channels", None) != 1:
        return False
    if getattr(frontend, "raw_split_axis", None) != "taps":
        return False
    pwl = getattr(frontend, "mag_layer", None)
    if pwl is None or getattr(pwl, "method", None) != "pwl" or pwl._pwl_k0_dw is None or len(pwl._pwl_shift_dws) != 3:
        return False
    return all(float(np.min(s.get_weights()[0])) > 0 for s in pwl._pwl_shift_dws)


class _Envelope(layers.Layer):
    """The frontend up to its smoothed band envelope (shared by both paths)."""

    def __init__(self, frontend, **kwargs):
        super().__init__(**kwargs)
        self.frontend = frontend

    def call(self, inputs):
        f, g = self.frontend, self.frontend.geom
        y = tf.reshape(inputs[:, : g.crop, :], [-1, 1, g.crop // g.fold, g.fold])
        parts = [f._exposure_clip(conv(y[:, :, i : i + g.stride * (g.frames - 1) + 1, :])) for i, conv in enumerate(f.fb)]
        total = parts[0]
        for part in parts[1:]:
            total = f._exposure_clip(total + part)
        m = f.band_channels
        return f.band_smooth(f._band_envelope(total[:, :, :, :m], total[:, :, :, m:]))


class _ToSpec(layers.Layer):
    """[B, 1, T, bands] -> [B, bands, spec_width, 1], as the frontend ends."""

    def __init__(self, spec_width: int, **kwargs):
        super().__init__(**kwargs)
        self.spec_width = int(spec_width)

    def call(self, y):
        return tf.transpose(y[:, :, : self.spec_width, :], [0, 3, 2, 1])


def split_points(k0: np.ndarray, shifts: list, ks: list, yt: float) -> np.ndarray:
    """Per band, the first band magnitude where the PWL output reaches ``yt`` (``_X_MAX`` if it never does)."""
    bands = k0.size
    grid = np.linspace(0.0, _X_MAX, 64001, dtype=np.float32)[:, None]
    y = grid * k0.reshape(1, bands)
    for (sw, sb), kw in zip(shifts, ks, strict=True):
        y = y + kw.reshape(1, bands) * np.maximum(grid * sw.reshape(1, bands) + sb.reshape(1, bands), 0.0)
    reach = y >= yt
    xt = np.where(reach.any(axis=0), grid[np.argmax(reach, axis=0), 0], _X_MAX)
    return np.maximum(xt, 1e-3).astype(np.float32)


def split_pwl(model: tf.keras.Model, yt: float = SPLIT_YT) -> tf.keras.Model:
    """The model with its frontend's band magnitude and PWL split in two exposures (float-identical).

    Layers outside the frontend are reused, so the result shares their weights;
    the frontend's filterbank and smoother are shared too.
    """
    if not can_split(model):
        raise ValueError("split_pwl needs a raw frontend with a fused bank, a compress mix and a 3-hinge PWL")
    f = model.get_layer("audio_frontend")
    pwl = f.mag_layer
    (wm,) = f.exposure_mix.get_weights()
    gamma, beta, mean, var = f.band_bn.get_weights()
    sc = gamma / np.sqrt(var + f.band_bn.epsilon)
    bias = beta - mean * sc
    k0 = pwl._pwl_k0_dw.get_weights()[0]
    shifts = [s.get_weights() for s in pwl._pwl_shift_dws]
    ks = [k.get_weights()[0] for k in pwl._pwl_k_dws]
    xt = split_points(k0, shifts, ks, yt)
    gain = (6.0 / xt).astype(np.float32)

    low_shifts = [(sw / gain.reshape(sw.shape), sb) for sw, sb in shifts]
    high_shifts = []
    for sw, sb in shifts:
        t = -sb / sw.reshape(sb.shape)
        high_shifts.append((sw, np.where(t >= xt, sb + sw.reshape(sb.shape) * xt, 0.0).astype(np.float32)))

    pending: list[tuple[layers.Layer, list[np.ndarray]]] = []

    def dw(name, weights):
        layer = layers.DepthwiseConv2D(1, use_bias=len(weights) == 2, name=name)
        pending.append((layer, weights))
        return layer

    def hinge_sum(x, tag, k0w, shift_params):
        y = dw(f"audio_frontend_split_{tag}_k0", [k0w])(x)
        for i, ((sw, sb), kw) in enumerate(zip(shift_params, ks, strict=True), start=1):
            hinge = layers.ReLU(name=f"audio_frontend_split_{tag}_relu{i}")(dw(f"audio_frontend_split_{tag}_shift{i}", [sw, sb])(x))
            y = layers.Add(name=f"audio_frontend_split_{tag}_add{i}")([y, dw(f"audio_frontend_split_{tag}_k{i}", [kw])(hinge)])
        return y

    mix_low = layers.Conv2D(wm.shape[-1], 1, use_bias=True, name="audio_frontend_split_mix_low")
    mix_high = layers.Conv2D(wm.shape[-1], 1, use_bias=True, name="audio_frontend_split_mix_high")
    pending += [(mix_low, [wm * (sc * gain), bias * gain]), (mix_high, [wm * sc, bias - xt])]

    def frontend_split(inputs):
        s = _Envelope(f, name="audio_frontend_split_envelope")(inputs)
        x_low = layers.ReLU(max_value=6.0, name="audio_frontend_split_low_x")(mix_low(s))
        u = layers.ReLU(name="audio_frontend_split_high_u")(mix_high(s))
        y_low = hinge_sum(x_low, "low", k0 / gain.reshape(k0.shape), low_shifts)
        y_high = hinge_sum(u, "high", k0, high_shifts)
        y = layers.Add(name="audio_frontend_split_sum")([y_low, y_high])
        return _ToSpec(f.spec_width, name="audio_frontend_split_out")(y)

    # Replay the functional graph layer by layer, the frontend swapped for its split.
    first = model.layers[0]
    new_input = tf.keras.Input(model.input.shape[1:], dtype=model.input.dtype, name=first.name if isinstance(first, layers.InputLayer) else None)
    tensors = {id(model.input): new_input}
    for layer in model.layers:
        if isinstance(layer, layers.InputLayer):
            continue
        ins = [tensors[id(t)] for t in layer._inbound_nodes[0].input_tensors]
        out = frontend_split(ins[0]) if layer is f else layer(ins if len(ins) > 1 else ins[0])
        tensors[id(layer.output)] = out
    split = tf.keras.Model(new_input, tensors[id(model.output)], name=model.name)
    for layer, weights in pending:
        layer.set_weights(weights)
    return split
