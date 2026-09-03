"""Additional model building blocks for audio classification.

Provides N6 NPU-compatible building blocks:
- Channel alignment for the NPU
- Lightweight attention pooling

Squeeze-and-excite and inverted-residual blocks were removed: A1 showed that
backbone could not reach the release parity gates under INT8 (0.739 mean
cosine, unrecoverable by QAT), and every shipped model uses plain depthwise
separable blocks instead.
"""

import tensorflow as tf
from tensorflow.keras import layers


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
