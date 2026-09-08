"""Serializable activation bounds shared by training and deployment graphs."""

import math

import tensorflow as tf


def validate_bounds(bounds):
    """Normalize named bounds and reject invalid calibration results."""
    result = {}
    for name, (lo, hi) in (bounds or {}).items():
        lo, hi = float(lo), float(hi)
        if not math.isfinite(lo) or not math.isfinite(hi) or lo > 0 or hi < 0 or lo >= hi:
            raise ValueError(f"Invalid activation bound for {name}: {(lo, hi)}")
        result[name] = (lo, hi)
    return result


def clip_activation(inputs, bounds, name):
    """Apply a real bound before fake quantization, including after export.

    Affine scaling around ReLU6 represents arbitrary bounds using ordinary
    operators. Compiler fusion and placement must still be measured.
    """
    if name not in bounds:
        return inputs
    lo, hi = bounds[name]
    scale = 6.0 / (hi - lo)
    return tf.nn.relu6((inputs - lo) * scale) / scale + lo
