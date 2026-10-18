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
# its channels for 99% of the variance, the expansions of stages 2-3 about half): expansion 4 in stages 2-4, residual
# stream x1.5 in stages 4-6 (80 -> 120, 112 -> 168, 192 -> 288).
B0_WIDE_BLOCKS = (
    (1, 3, 1, 16, 1),
    (4, 3, 2, 24, 2),
    (4, 5, 2, 40, 2),
    (4, 3, 3, 120, 2),
    (6, 5, 3, 168, 1),
    (6, 5, 4, 288, 2),
    (6, 3, 1, 320, 1),
)
DROP_CONNECT_RATE = 0.2


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


def efficientnet_b0_backbone(x: tf.Tensor, embeddings_size: int, blocks: tuple = B0_BLOCKS) -> tf.Tensor:
    """N6-B0 stem, seven MBConv stages and the top conv on a frontend's [B, F, T, C] output.

    Args:
        x: Frontend output.
        embeddings_size: Channels of the top 1x1 conv (the pooled embedding).
        blocks: The stage table, (expand, kernel, repeats, output channels, first stride) per stage; B0's by default.

    Returns:
        Feature map [B, F/32, T/32, embeddings_size].
    """
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

    x = layers.Conv2D(int(embeddings_size), 1, padding="same", use_bias=False, name="top_conv")(x)
    x = layers.BatchNormalization(name="top_bn")(x)
    return _relu6("top_act")(x)
