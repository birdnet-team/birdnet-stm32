"""Model profiling: per-layer MACs, parameters, and activation memory.

Provides a simple profiling utility that inspects a Keras model and
reports per-layer statistics useful for estimating NPU/MCU cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

# Operations known to be supported by the STM32N6 NPU
N6_SUPPORTED_OPS = frozenset({
    "Conv2D",
    "DepthwiseConv2D",
    "Dense",
    "BatchNormalization",
    "ReLU",
    "Add",
    "Multiply",
    "GlobalAveragePooling2D",
    "AveragePooling2D",
    "MaxPooling2D",
    "Reshape",
    "Flatten",
    "Concatenate",
    "ZeroPadding2D",
    "Dropout",
    "SpatialDropout2D",
    "Activation",
    "Softmax",
    "Sigmoid",
    "InputLayer",
})

# Layer types that are unsupported or need manual verification
N6_WARN_OPS = frozenset({
    "Lambda",
    "LSTM",
    "GRU",
    "SimpleRNN",
    "Bidirectional",
    "MultiHeadAttention",
    "LayerNormalization",
    "GroupNormalization",
})


@dataclass
class LayerProfile:
    """Per-layer profiling result.

    Attributes:
        name: Layer name.
        layer_type: Keras layer class name.
        output_shape: Output shape as string.
        params: Number of trainable + non-trainable parameters.
        macs: Estimated multiply-accumulate operations.
        activation_bytes: Estimated activation memory in bytes (float32).
        n6_supported: Whether this layer type is known to be N6 NPU-compatible.
    """

    name: str
    layer_type: str
    output_shape: str
    params: int
    macs: int
    activation_bytes: int
    n6_supported: bool


def _estimate_macs(layer: tf.keras.layers.Layer) -> int:
    """Estimate MACs for a single layer."""
    try:
        out = layer.output_shape
    except AttributeError:
        return 0

    if isinstance(layer, tf.keras.layers.Conv2D) and not isinstance(layer, tf.keras.layers.DepthwiseConv2D):
        ks = layer.kernel_size
        if out is None or len(out) < 4:
            return 0
        _, H, W, C_out = out
        try:
            C_in = layer.input_shape[-1] if layer.input_shape else 0
        except AttributeError:
            return 0
        if any(v is None for v in (H, W, C_out, C_in)):
            return 0
        return int(H) * int(W) * int(C_out) * int(ks[0]) * int(ks[1]) * int(C_in)

    if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
        ks = layer.kernel_size
        if out is None or len(out) < 4:
            return 0
        _, H, W, C = out
        if any(v is None for v in (H, W, C)):
            return 0
        return int(H) * int(W) * int(C) * int(ks[0]) * int(ks[1])

    if isinstance(layer, tf.keras.layers.Dense):
        if out is None or len(out) < 2:
            return 0
        try:
            in_dim = layer.input_shape[-1] if layer.input_shape else 0
        except AttributeError:
            return 0
        out_dim = out[-1]
        if in_dim is None or out_dim is None:
            return 0
        return int(in_dim) * int(out_dim)

    return 0


def _activation_bytes(layer: tf.keras.layers.Layer) -> int:
    """Estimate activation memory in bytes (float32 output)."""
    try:
        out = layer.output_shape
    except AttributeError:
        return 0
    if out is None:
        return 0
    # Flatten shape, skip batch dim
    shape = out[1:] if isinstance(out, tuple) else out
    if isinstance(shape, list):
        # Multi-output — take first
        shape = shape[0][1:] if shape else ()
    elements = 1
    for dim in shape:
        if dim is None:
            return 0
        elements *= int(dim)
    return elements * 4  # float32


def profile_model(model: tf.keras.Model) -> list[LayerProfile]:
    """Profile a Keras model and return per-layer statistics.

    Args:
        model: Compiled or uncompiled Keras model.

    Returns:
        List of LayerProfile for each layer.
    """
    profiles = []
    for layer in model.layers:
        ltype = type(layer).__name__
        n6_ok = ltype in N6_SUPPORTED_OPS
        if ltype in N6_WARN_OPS:
            n6_ok = False

        out_shape = str(layer.output_shape) if hasattr(layer, "output_shape") else "?"
        try:
            params = layer.count_params()
        except ValueError:
            params = 0
        macs = _estimate_macs(layer)
        act_bytes = _activation_bytes(layer)

        profiles.append(LayerProfile(
            name=layer.name,
            layer_type=ltype,
            output_shape=out_shape,
            params=params,
            macs=macs,
            activation_bytes=act_bytes,
            n6_supported=n6_ok,
        ))
    return profiles


def print_profile(model: tf.keras.Model, warn_unsupported: bool = True) -> None:
    """Print a formatted profiling table for a Keras model.

    Args:
        model: Keras model to profile.
        warn_unsupported: Print warnings for layers not known to be N6-compatible.
    """
    profiles = profile_model(model)
    total_params = sum(p.params for p in profiles)
    total_macs = sum(p.macs for p in profiles)
    total_act = sum(p.activation_bytes for p in profiles)

    print(f"\n{'Layer':<35} {'Type':<25} {'Output Shape':<25} {'Params':>10} {'MACs':>12} {'N6':>4}")
    print("-" * 115)
    warnings = []
    for p in profiles:
        n6_str = "OK" if p.n6_supported else "?"
        if not p.n6_supported and p.layer_type not in ("InputLayer",):
            warnings.append(p)
        print(f"{p.name:<35} {p.layer_type:<25} {p.output_shape:<25} {p.params:>10,} {p.macs:>12,} {n6_str:>4}")

    print("-" * 115)
    print(f"{'Total':<35} {'':<25} {'':<25} {total_params:>10,} {total_macs:>12,}")
    print(f"Activation memory: {total_act / 1024:.1f} KB (float32)")
    print(f"Model size: ~{total_params * 4 / 1024:.1f} KB (float32), ~{total_params / 1024:.1f} KB (INT8)")

    if warn_unsupported and warnings:
        print(f"\nWARNING: {len(warnings)} layer(s) have unknown N6 NPU compatibility:")
        for p in warnings:
            print(f"  - {p.name} ({p.layer_type})")


def check_n6_compatibility(model: tf.keras.Model) -> list[LayerProfile]:
    """Check model for layers with unknown N6 NPU compatibility.

    Args:
        model: Keras model to check.

    Returns:
        List of LayerProfile for layers not known to be N6-compatible
        (excludes InputLayer).
    """
    profiles = profile_model(model)
    return [p for p in profiles if not p.n6_supported and p.layer_type != "InputLayer"]
