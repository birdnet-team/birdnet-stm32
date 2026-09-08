"""Additional model building blocks for audio classification.

Provides channel alignment for the NPU. Attention pooling lived here too and
was removed in 1.2.0; the head uses global average pooling.

Squeeze-and-excite and inverted-residual blocks were removed: A1 showed that
backbone could not reach the release parity gates under INT8 (0.739 mean
cosine, unrecoverable by QAT), and every shipped model uses plain depthwise
separable blocks instead.
"""


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
