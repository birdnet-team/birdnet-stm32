"""Equalize per-band INT8 ranges in the raw frontend without changing its function.

Every internal frontend tensor is quantized per tensor across all bands, and on
the v1.2 raw model a single band sets each range: the median band's typical
filterbank value is below one INT8 step. Three per-band rescalings are exact in
float, so they redistribute the grid without retraining:

1. **Filterbank gain** ``s`` (stage ``fb``). Multiply band c's quadrature
   kernels by ``s_c``. ``|x|``, the magnitude approximation and the per-band
   smoothing are all positively homogeneous per band, so every tensor up to
   ``band_bn`` scales by ``s_c``; ``band_bn`` absorbs it exactly (mean ``*s``,
   variance ``*s^2``, gamma corrected for epsilon).
2. **PWL input gain** ``t`` (stage ``pwl_in``). Scale ``band_bn`` gamma and beta
   by ``t_c`` (so ``band_relu`` scales by ``t_c``) and divide the ``k0`` and
   hinge-shift input weights by ``t_c``.
3. **Hinge gain** ``a`` (stage ``hinge``). Scale hinge i's shift weight and bias
   by ``a_i,c`` (its ReLU output scales by ``a_i,c``) and divide its slope
   ``k_i`` by it.

Per-channel symmetric weight quantization is invariant to a per-output-channel
scale, so the INT8 filterbank kernels are unchanged; what moves is how much of
each activation grid a band gets. The PWL output tensors feed the backbone as
spatial rows, where no per-band freedom exists, and are left alone.

Gains bring every band to the median band's p99.9 magnitude on calibration
inputs. Measured on the v1.2 raw model: +0.028 validation cMAP after
post-training quantization, +0.017 catalog cMAP after QAT, float output
unchanged to 6e-7. See ``docs/dev/int8-parity-plan.md``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import tensorflow as tf

STAGES = ("fb", "pwl_in", "hinge")
PERCENTILE = 99.9
# Largest float output change accepted as "function-preserving".
MAX_OUTPUT_DIFF = 1e-3


class _Collector:
    """Frontend quantization hook that records per-band values of named tensors."""

    def __init__(self, names: set[str]):
        self.names = names
        self.values: dict[str, list[np.ndarray]] = {}

    def activation(self, name, x):
        if name in self.names:
            a = np.asarray(x)
            self.values.setdefault(name, []).append(a.reshape(-1, a.shape[-1]))
        return x

    def kernel(self, layer, x):
        return layer(x)


def band_percentiles(
    model: tf.keras.Model, frontend, tensors: Iterable[np.ndarray], names: set[str]
) -> dict[str, np.ndarray]:
    """Per-band ``PERCENTILE`` of ``|x|`` for each named frontend tensor."""
    collector = _Collector(names)
    frontend.set_quantization_hook(collector)
    probe = tf.keras.Model(model.inputs, frontend.output)
    try:
        for x in tensors:
            probe(x, training=False)
    finally:
        frontend.set_quantization_hook(None)
    return {n: np.percentile(np.abs(np.concatenate(v)), PERCENTILE, axis=0) for n, v in collector.values.items()}


def spread(p: np.ndarray) -> float:
    """Tensor range over the median band's percentile: what equalization shrinks."""
    return float(p.max() / max(np.median(p), 1e-12))


def gains(p: np.ndarray) -> np.ndarray:
    """Bring every active band to the median band's percentile; leave dead bands alone."""
    target = np.median(p[p > 0]) if np.any(p > 0) else 1.0
    return np.where(p > 1e-9, target / np.maximum(p, 1e-12), 1.0).astype(np.float64)


def equalize_raw_frontend(
    model: tf.keras.Model,
    tensors: Sequence[np.ndarray],
    check: Sequence[np.ndarray],
    stages: Iterable[str] = STAGES,
) -> dict:
    """Equalize ``model``'s raw frontend in place and return a report.

    Args:
        model: Float Keras model with a raw ``audio_frontend`` and a PWL magnitude.
        tensors: Calibration inputs (with batch dimension) that set the gains.
        check: Held-out inputs on which the float output must not change.
        stages: Subset of ``STAGES``.

    Raises:
        ValueError: The model is not a raw PWL frontend, or a stage is unknown.
        RuntimeError: The float output moved by more than ``MAX_OUTPUT_DIFF``;
            the weights are then already modified and must be discarded.
    """
    stages = set(stages)
    if not stages or not stages <= set(STAGES):
        raise ValueError(f"stages must be a non-empty subset of {STAGES}, got {sorted(stages)}")
    fe = model.get_layer("audio_frontend")
    if fe.mode != "raw" or fe.mag_scale != "pwl":
        raise ValueError("Equalization is written for the raw frontend with a PWL magnitude")

    mag = fe.mag_layer
    n = fe.name
    hinge_names = [f"{s.name}_relu" for s in mag._pwl_shift_dws]  # noqa: SLF001
    watch = {f"{n}_fb_re", f"{n}_fb_im", f"{n}_magnitude", fe.band_relu.name, *hinge_names}
    reference = np.concatenate([model(x, training=False).numpy() for x in check])
    before = band_percentiles(model, fe, tensors, watch)
    report: dict = {"before": {k: spread(v) for k, v in before.items()}, "stages": sorted(stages)}

    eps = float(fe.band_bn.epsilon)
    if "fb" in stages:
        # One gain per band for both quadrature banks, so the magnitude stays meaningful.
        s = gains(np.maximum(before[f"{n}_fb_re"], before[f"{n}_fb_im"]))
        for conv in (*fe.fb_re, *fe.fb_im):
            (kernel,) = conv.get_weights()
            conv.set_weights([(kernel * s[None, None, None, :]).astype(np.float32)])
        gamma, beta, mean, var = fe.band_bn.get_weights()
        new_var = var * s**2
        gamma = gamma * np.sqrt(new_var + eps) / (s * np.sqrt(var + eps))
        fe.band_bn.set_weights(
            [gamma.astype(np.float32), beta, (mean * s).astype(np.float32), new_var.astype(np.float32)]
        )
        report["fb_gain_range"] = [float(s.min()), float(s.max())]

    if "pwl_in" in stages:
        current = band_percentiles(model, fe, tensors, {fe.band_relu.name})[fe.band_relu.name]
        t = gains(current)
        gamma, beta, mean, var = fe.band_bn.get_weights()
        fe.band_bn.set_weights([(gamma * t).astype(np.float32), (beta * t).astype(np.float32), mean, var])
        (k0,) = mag._pwl_k0_dw.get_weights()  # noqa: SLF001
        mag._pwl_k0_dw.set_weights([(k0 / t[None, None, :, None]).astype(np.float32)])  # noqa: SLF001
        for shift in mag._pwl_shift_dws:  # noqa: SLF001
            w, b = shift.get_weights()
            shift.set_weights([(w / t[None, None, :, None]).astype(np.float32), b])
        report["pwl_in_gain_range"] = [float(t.min()), float(t.max())]

    if "hinge" in stages:
        current = band_percentiles(model, fe, tensors, set(hinge_names))
        for shift, slope, name in zip(mag._pwl_shift_dws, mag._pwl_k_dws, hinge_names, strict=True):  # noqa: SLF001
            a = gains(current[name])
            w, b = shift.get_weights()
            shift.set_weights([(w * a[None, None, :, None]).astype(np.float32), (b * a).astype(np.float32)])
            (k,) = slope.get_weights()
            slope.set_weights([(k / a[None, None, :, None]).astype(np.float32)])
            report[f"{name}_gain_range"] = [float(a.min()), float(a.max())]

    after = band_percentiles(model, fe, tensors, watch)
    report["after"] = {k: spread(v) for k, v in after.items()}
    output = np.concatenate([model(x, training=False).numpy() for x in check])
    report["max_abs_output_diff"] = float(np.max(np.abs(output - reference)))
    if report["max_abs_output_diff"] > MAX_OUTPUT_DIFF:
        raise RuntimeError(
            f"Equalization changed the float model (max output diff {report['max_abs_output_diff']:.2e})"
        )
    return report
