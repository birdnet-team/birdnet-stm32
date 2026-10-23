"""N6-B0: an EfficientNet-B0 backbone for INT8 on the STM32N6 (2.2).

EfficientNet-B0's inverted-residual MBConv blocks (Tan & Le 2019) behind the
project's audio frontends, with the three changes that make B0 survive INT8 on
the Neural-ART NPU: ReLU6 instead of SiLU, no squeeze-and-excite, and a bounded
residual stream (ReLU6 after each projection and residual add, so the stream
stays in [0, 6] as in the DS-CNN blocks). B0 as published lost 14% of its WABAD
AP to post-training quantization; N6-B0 trained to 2.2's float WABAD-61 AUPRC
0.465 (2.1 Raw: 0.374), INT8 0.441.

The block table and stochastic depth follow B0 at width and depth 1.0. The top
1x1 conv emits ``embeddings_size`` channels (1280), pooled by the same ``gap``
layer as the DS-CNN so the embedding loss and the head fine-tune find their
layers.

For the 2.5 size study the stage table can be any table (``parse_blocks``), and
a channel-attention block (``ChannelAttention``, EdgeNeXt's transposed
attention) can follow the last block of chosen stages. On the STM32N6 one such
block after stage 7 of N6-B0 adds 4 ms (36.9 vs 32.9 ms on target), three after
stages 3, 5 and 7 add 11 ms: the per-head matmuls are split per head and the L2
norm runs in float on the Cortex-M55, but everything maps.
"""

import tensorflow as tf
from tensorflow.keras import layers

# (expand ratio, depthwise kernel, repeats, output channels, first stride)
B0_BLOCKS = (
    (1, 3, 1, 16, 1),
    (6, 3, 2, 24, 2),
    (6, 5, 2, 40, 2),
    (6, 3, 3, 80, 2),
    (6, 5, 3, 112, 1),
    (6, 5, 4, 192, 2),
    (6, 3, 1, 320, 1),
)
# N6-B0 with wider bottlenecks (Stefan, 2026-10-16; from the PCA of the 2.2 run: the residual stream needed 92-96% of
# its channels for 99% of the variance, the expansions of stages 2-3 about half, stage 7 and the top conv 84% and 69%):
# expansion 4 in stages 2-4, residual stream x2 in stages 4-6 (80 -> 160, 112 -> 224, 192 -> 384). 2.2.0 had x1.5.
B0_WIDE_BLOCKS = (
    (1, 3, 1, 16, 1),
    (4, 3, 2, 24, 2),
    (4, 5, 2, 40, 2),
    (4, 3, 3, 160, 2),
    (6, 5, 3, 224, 1),
    (6, 5, 4, 384, 2),
    (6, 3, 1, 320, 1),
)
DROP_CONNECT_RATE = 0.2
ATTENTION_HEAD_WIDTH = 32  # channels per attention head, as in EdgeNeXt


def parse_blocks(spec: str) -> tuple[tuple[int, int, int, int, int], ...]:
    """A stage table from ``"e,k,r,c,s/e,k,r,c,s/..."`` (expand, kernel, repeats, channels, first stride per stage).

    Raises:
        ValueError: On a stage that is not five positive integers, an even kernel, or a stride other than 1 or 2
            (larger strides fall back to the Cortex-M55 on the STM32N6).
    """
    blocks = []
    for stage in spec.replace(" ", "").split("/"):
        try:
            values = tuple(int(v) for v in stage.split(","))
        except ValueError as exc:
            raise ValueError(f"stage '{stage}' is not five integers") from exc
        if len(values) != 5 or min(values) < 1:
            raise ValueError(
                f"stage '{stage}' must be five positive integers (expand, kernel, repeats, channels, stride)"
            )
        if values[1] % 2 == 0 or values[4] not in (1, 2):
            raise ValueError(f"stage '{stage}': the kernel must be odd and the stride 1 or 2")
        blocks.append(values)
    if not blocks:
        raise ValueError("empty stage table")
    return tuple(blocks)


@tf.keras.utils.register_keras_serializable(package="birdnet_stm32")
class ChannelAttention(layers.Layer):
    """Transposed (cross-covariance) attention of XCiT and EdgeNeXt (arXiv 2206.10589) on ``[q, k, v]``.

    Each input is ``[B, F, T, C]``. Per head of ``C / heads`` channels, q and k
    are L2-normalized over the ``F * T`` positions, their ``C x C`` channel
    affinities, scaled by a learned temperature per head, go through a softmax,
    and the result mixes the channels of v. Cost grows with positions times
    channels squared, not positions squared. Shaped for TFLite: the
    normalization and the softmax run over the last axis, transposes stay 4-D.

    Args:
        heads: Number of heads; must divide the channel count.
    """

    def __init__(self, heads: int, **kwargs):
        super().__init__(**kwargs)
        self.heads = int(heads)

    def build(self, input_shape):
        self.temperature = self.add_weight(name="temperature", shape=(1, self.heads, 1, 1), initializer="ones")
        super().build(input_shape)

    def call(self, inputs):
        q, k, v = inputs
        f, t, c = int(q.shape[1]), int(q.shape[2]), int(q.shape[3])
        h, d = self.heads, c // self.heads

        def heads_first(z):  # [B, F, T, C] -> [B, h, d, F*T]
            return tf.transpose(tf.reshape(z, [-1, f * t, h, d]), [0, 2, 3, 1])

        q, k, v = heads_first(q), heads_first(k), heads_first(v)
        q, k = tf.math.l2_normalize(q, axis=-1), tf.math.l2_normalize(k, axis=-1)
        scores = tf.matmul(q, k, transpose_b=True) * tf.cast(self.temperature, q.dtype)  # [B, h, d, d]
        out = tf.matmul(tf.nn.softmax(scores, axis=-1), v)  # [B, h, d, F*T]
        return tf.reshape(tf.transpose(out, [0, 3, 1, 2]), [-1, f, t, c])

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        return {**super().get_config(), "heads": self.heads}


def channel_attention_block(x: tf.Tensor, name: str) -> tf.Tensor:
    """A bounded residual ChannelAttention block: BN, 1x1 q / k / v, attention, 1x1 projection, BN, add, ReLU6.

    Args:
        x: Input tensor [B, F, T, C].
        name: Base name for layers.

    Returns:
        Output tensor of the same shape.
    """
    c = int(x.shape[-1])
    heads = max(1, c // ATTENTION_HEAD_WIDTH)
    while c % heads:
        heads -= 1
    y = layers.BatchNormalization(name=f"{name}_bn")(x)
    q, k, v = (layers.Conv2D(c, 1, use_bias=False, name=f"{name}_{part}")(y) for part in "qkv")
    y = ChannelAttention(heads, name=f"{name}_attn")([q, k, v])
    y = layers.Conv2D(c, 1, use_bias=False, name=f"{name}_proj")(y)
    y = layers.BatchNormalization(name=f"{name}_proj_bn")(y)
    y = layers.Add(name=f"{name}_add")([x, y])
    return _relu6(f"{name}_add_relu")(y)


def _relu6(name: str) -> layers.Layer:
    return layers.ReLU(max_value=6, name=name)


def mbconv_block(
    x: tf.Tensor,
    out_ch: int,
    expand: int,
    kernel: int,
    stride: int,
    drop_rate: float,
    name: str,
) -> tf.Tensor:
    """MBConv without squeeze-and-excite: 1x1 expand, k x k depthwise, 1x1 project, bounded residual.

    Args:
        x: Input tensor [B, F, T, C].
        out_ch: Output channels of the projection.
        expand: Expansion ratio of the hidden width (1 skips the expand conv).
        kernel: Square depthwise kernel size.
        stride: Depthwise stride on both axes.
        drop_rate: Stochastic-depth rate on the residual branch.
        name: Base name for layers.

    Returns:
        Output tensor [B, F', T', out_ch].
    """
    in_ch = int(x.shape[-1])
    y = x
    if expand != 1:
        y = layers.Conv2D(in_ch * expand, 1, padding="same", use_bias=False, name=f"{name}_expand_conv")(y)
        y = layers.BatchNormalization(name=f"{name}_expand_bn")(y)
        y = _relu6(f"{name}_expand_act")(y)
    y = layers.DepthwiseConv2D(kernel, strides=stride, padding="same", use_bias=False, name=f"{name}_dw")(y)
    y = layers.BatchNormalization(name=f"{name}_dw_bn")(y)
    y = _relu6(f"{name}_dw_act")(y)
    y = layers.Conv2D(out_ch, 1, padding="same", use_bias=False, name=f"{name}_project_conv")(y)
    y = layers.BatchNormalization(name=f"{name}_project_bn")(y)
    # B0's linear bottleneck is unbounded, and its outliers set coarse per-tensor INT8 grids.
    y = _relu6(f"{name}_project_relu")(y)
    if stride == 1 and in_ch == out_ch:
        if drop_rate > 0:
            y = layers.Dropout(drop_rate, noise_shape=(None, 1, 1, 1), name=f"{name}_drop")(y)
        y = layers.Add(name=f"{name}_add")([x, y])
        y = _relu6(f"{name}_add_relu")(y)
    return y


def efficientnet_b0_backbone(
    x: tf.Tensor, embeddings_size: int, blocks: tuple = B0_BLOCKS, attention_stages: tuple | list = ()
) -> tf.Tensor:
    """N6-B0 stem, the MBConv stages and the top conv on a frontend's [B, F, T, C] output.

    Args:
        x: Frontend output.
        embeddings_size: Channels of the top 1x1 conv (the pooled embedding).
        blocks: The stage table, (expand, kernel, repeats, output channels, first stride) per stage; B0's by default.
        attention_stages: 1-based stages whose last block is followed by a ``channel_attention_block``.

    Returns:
        Feature map [B, F/32, T/32, embeddings_size] (for B0's four stride-2 stages after the stem).

    Raises:
        ValueError: On an attention stage outside the table.
    """
    if any(not 1 <= int(si) <= len(blocks) for si in attention_stages):
        raise ValueError(f"attention_stages {list(attention_stages)} outside stages 1..{len(blocks)}")
    x = layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False, name="stem_conv")(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = _relu6("stem_act")(x)

    total = sum(reps for _, _, reps, _, _ in blocks)
    done = 0
    for si, (expand, kernel, reps, out_ch, stride) in enumerate(blocks, start=1):
        for bi in range(reps):
            x = mbconv_block(
                x,
                out_ch,
                expand=expand,
                kernel=kernel,
                stride=stride if bi == 0 else 1,
                drop_rate=DROP_CONNECT_RATE * done / total,
                name=f"block{si}{chr(ord('a') + bi)}",
            )
            done += 1
        if si in {int(a) for a in attention_stages}:
            x = channel_attention_block(x, f"attn{si}")

    x = layers.Conv2D(int(embeddings_size), 1, padding="same", use_bias=False, name="top_conv")(x)
    x = layers.BatchNormalization(name="top_bn")(x)
    return _relu6("top_act")(x)
