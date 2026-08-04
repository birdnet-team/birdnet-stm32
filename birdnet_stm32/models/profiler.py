"""Model profiling: per-layer MACs, parameters, and activation memory.

Provides a simple profiling utility that inspects a Keras model and
reports per-layer statistics useful for estimating NPU/MCU cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

# Operations known to be supported by the STM32N6 NPU
N6_SUPPORTED_OPS = frozenset(
    {
        "Conv1D",
        "Conv2D",
        "DepthwiseConv1D",
        "DepthwiseConv2D",
        "SeparableConv2D",
        "Dense",
        "BatchNormalization",
        "ReLU",
        "Add",
        "Multiply",
        "GlobalAveragePooling2D",
        "GlobalAveragePooling1D",
        "AveragePooling2D",
        "MaxPooling2D",
        "Reshape",
        "Permute",
        "Flatten",
        "Concatenate",
        "ZeroPadding2D",
        "Dropout",
        "SpatialDropout2D",
        "Activation",
        "Softmax",
        "Sigmoid",
        "Rescaling",
        "InputLayer",
        # Project-internal layers — their internal ops decompose to the
        # supported set above and have been verified end-to-end with stedgeai.
        "AudioFrontendLayer",
        "MagnitudeScalingLayer",
        "AttentionPooling",
    }
)

# Layer types that are unsupported or need manual verification
N6_WARN_OPS = frozenset(
    {
        "Lambda",
        "LSTM",
        "GRU",
        "SimpleRNN",
        "Bidirectional",
        "MultiHeadAttention",
        "LayerNormalization",
        "GroupNormalization",
    }
)


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


def _safe_shape(tensor_or_list) -> tuple | None:
    """Return ``tensor.shape`` as a plain tuple for the (first) output of a layer.

    Keras 3 removed ``Layer.output_shape``; ``layer.output`` returns a
    ``KerasTensor`` (or list thereof for multi-output layers) whose ``.shape``
    is a ``TensorShape``. This helper normalises both to a plain tuple, or
    returns ``None`` if the layer isn't connected (no inbound nodes yet).
    """
    if tensor_or_list is None:
        return None
    if isinstance(tensor_or_list, (list, tuple)) and tensor_or_list and not hasattr(tensor_or_list[0], "shape"):
        return None
    t = tensor_or_list[0] if isinstance(tensor_or_list, (list, tuple)) else tensor_or_list
    shape = getattr(t, "shape", None)
    if shape is None:
        return None
    return tuple(shape)


def _layer_output_shape(layer: tf.keras.layers.Layer) -> tuple | None:
    """Return the first output tensor's shape for ``layer``, or ``None``."""
    try:
        return _safe_shape(layer.output)
    except (AttributeError, ValueError):
        return None


def _layer_input_shape(layer: tf.keras.layers.Layer) -> tuple | None:
    """Return the first input tensor's shape for ``layer``, or ``None``."""
    try:
        return _safe_shape(layer.input)
    except (AttributeError, ValueError):
        return None


def _estimate_macs(layer: tf.keras.layers.Layer) -> int:
    """Estimate MACs for a single layer."""
    out = _layer_output_shape(layer)
    if out is None:
        return 0

    if isinstance(layer, tf.keras.layers.Conv2D) and not isinstance(layer, tf.keras.layers.DepthwiseConv2D):
        ks = layer.kernel_size
        if len(out) < 4:
            return 0
        _, H, W, C_out = out
        in_shape = _layer_input_shape(layer)
        C_in = in_shape[-1] if in_shape else None
        if any(v is None for v in (H, W, C_out, C_in)):
            return 0
        return int(H) * int(W) * int(C_out) * int(ks[0]) * int(ks[1]) * int(C_in)

    if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
        ks = layer.kernel_size
        if len(out) < 4:
            return 0
        _, H, W, C = out
        if any(v is None for v in (H, W, C)):
            return 0
        return int(H) * int(W) * int(C) * int(ks[0]) * int(ks[1])

    if isinstance(layer, tf.keras.layers.Dense):
        if len(out) < 2:
            return 0
        in_shape = _layer_input_shape(layer)
        in_dim = in_shape[-1] if in_shape else None
        out_dim = out[-1]
        if in_dim is None or out_dim is None:
            return 0
        return int(in_dim) * int(out_dim)

    return 0


def _activation_bytes(layer: tf.keras.layers.Layer) -> int:
    """Estimate activation memory in bytes (float32 output)."""
    out = _layer_output_shape(layer)
    if out is None:
        return 0
    # Skip batch dim
    shape = out[1:]
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

        shape = _layer_output_shape(layer)
        out_shape = str(shape) if shape is not None else "?"
        try:
            params = layer.count_params()
        except ValueError:
            params = 0
        macs = _estimate_macs(layer)
        act_bytes = _activation_bytes(layer)

        profiles.append(
            LayerProfile(
                name=layer.name,
                layer_type=ltype,
                output_shape=out_shape,
                params=params,
                macs=macs,
                activation_bytes=act_bytes,
                n6_supported=n6_ok,
            )
        )
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
