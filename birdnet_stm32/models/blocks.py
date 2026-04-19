"""Additional model building blocks for audio classification.

Provides N6 NPU-compatible building blocks:
- Squeeze-and-Excite (SE) channel attention
- MobileNetV2-style inverted residual blocks
- Lightweight attention pooling
"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers


def _make_divisible(v: int | float, divisor: int = 8) -> int:
    """Round channel count to the nearest multiple of divisor (minimum = divisor).

    Args:
        v: Target channel count.
        divisor: Alignment divisor (default 8 for NPU).

    Returns:
        Aligned channel count.
    """
    v = int(v + divisor / 2) // divisor * divisor
    return max(divisor, v)


def se_block(x: tf.Tensor, reduction: int = 4, name: str = "se") -> tf.Tensor:
    """Squeeze-and-Excite channel attention block.

    NPU-compatible: uses GlobalAveragePooling2D, Dense, Sigmoid, Multiply.

    Args:
        x: Input tensor [B, H, W, C].
        reduction: Channel reduction factor for the bottleneck.
        name: Base name for layers.

    Returns:
        Channel-reweighted tensor, same shape as input.
    """
    channels = x.shape[-1]
    se_ch = max(1, int(channels) // reduction)

    squeeze = layers.GlobalAveragePooling2D(keepdims=True, name=f"{name}_squeeze")(x)
    excite = layers.Dense(se_ch, activation="relu", use_bias=False, name=f"{name}_reduce")(squeeze)
    excite = layers.Dense(int(channels), activation="sigmoid", use_bias=False, name=f"{name}_expand")(excite)
    return layers.Multiply(name=f"{name}_scale")([x, excite])


def inverted_residual_block(
    x: tf.Tensor,
    out_ch: int,
    expansion: int = 6,
    stride_f: int = 1,
    stride_t: int = 1,
    use_se: bool = False,
    se_reduction: int = 4,
    weight_decay: float = 1e-4,
    drop_rate: float = 0.1,
    name: str = "ir",
) -> tf.Tensor:
    """MobileNetV2-style inverted residual block with optional SE attention.

    Structure: 1x1 expand -> BN -> ReLU6 -> 3x3 DW -> BN -> ReLU6 -> [SE] -> 1x1 project -> BN
    Residual connection when stride=1 and channels match.

    All ops are NPU-compatible (Conv2D, DepthwiseConv2D, Dense, Sigmoid, Multiply, Add).

    Args:
        x: Input tensor [B, H, W, C].
        out_ch: Output channels.
        expansion: Expansion factor for the hidden dimension.
        stride_f: Stride along frequency axis.
        stride_t: Stride along time axis.
        use_se: Whether to apply squeeze-and-excite attention.
        se_reduction: SE channel reduction factor.
        weight_decay: L2 regularization weight.
        drop_rate: Spatial dropout rate.
        name: Base name for layers.

    Returns:
        Output tensor [B, H', W', out_ch].
    """
    reg = regularizers.l2(weight_decay) if weight_decay and weight_decay > 0 else None
    in_ch = x.shape[-1]
    hidden_ch = _make_divisible(int(in_ch) * expansion, 8)

    # Expand
    y = layers.Conv2D(
        hidden_ch, (1, 1), padding="same", use_bias=False,
        kernel_regularizer=reg, name=f"{name}_expand",
    )(x)
    y = layers.BatchNormalization(name=f"{name}_expand_bn")(y)
    y = layers.ReLU(max_value=6, name=f"{name}_expand_relu")(y)

    # Depthwise
    y = layers.DepthwiseConv2D(
        (3, 3), strides=(stride_f, stride_t), padding="same", use_bias=False,
        depthwise_regularizer=reg, name=f"{name}_dw",
    )(y)
    y = layers.BatchNormalization(name=f"{name}_dw_bn")(y)
    y = layers.ReLU(max_value=6, name=f"{name}_dw_relu")(y)

    # Optional SE
    if use_se:
        y = se_block(y, reduction=se_reduction, name=f"{name}_se")

    # Project (no activation — linear bottleneck)
    y = layers.Conv2D(
        out_ch, (1, 1), padding="same", use_bias=False,
        kernel_regularizer=reg, name=f"{name}_project",
    )(y)
    y = layers.BatchNormalization(name=f"{name}_project_bn")(y)

    if drop_rate and drop_rate > 0:
        y = layers.SpatialDropout2D(drop_rate, name=f"{name}_drop")(y)

    # Residual connection
    if (stride_f == 1 and stride_t == 1) and (in_ch is not None and int(in_ch) == int(out_ch)):
        y = layers.Add(name=f"{name}_add")([x, y])

    return y


class AttentionPooling(layers.Layer):
    """Lightweight attention pooling over spatial dimensions.

    Replaces GlobalAveragePooling2D with a learned weighted average.
    Uses only Dense + Softmax + Multiply + ReduceSum — all NPU-compatible.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._score_dense = None

    def build(self, input_shape):
        self._score_dense = layers.Dense(1, use_bias=False, name="score")
        super().build(input_shape)

    def call(self, x):
        # x: [B, H, W, C]
        shape = tf.shape(x)
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]
        flat = tf.reshape(x, [B, H * W, C])  # [B, H*W, C]
        attn = self._score_dense(flat)  # [B, H*W, 1]
        attn = tf.nn.softmax(attn, axis=1)
        weighted = flat * attn  # [B, H*W, C]
        return tf.reduce_sum(weighted, axis=1)  # [B, C]


def attention_pooling(x: tf.Tensor, name: str = "attn_pool") -> tf.Tensor:
    """Lightweight attention pooling over spatial dimensions.

    Replaces GlobalAveragePooling2D with a learned weighted average.
    Uses only Dense + Softmax + Multiply + ReduceSum — all NPU-compatible.

    Args:
        x: Input tensor [B, H, W, C].
        name: Base name for layers.

    Returns:
        Pooled tensor [B, C].
    """
    return AttentionPooling(name=name)(x)
