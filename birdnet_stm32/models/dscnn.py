"""DS-CNN (depthwise-separable CNN) model architecture for audio classification.

The model consists of:
- An AudioFrontendLayer (from frontend.py) for feature extraction.
- A stem convolution to lift channels.
- Four stages of depthwise-separable blocks with stride-2 downsampling.
- A pooling head (global average, or frequency mean then time max + mean),
  dropout, and a dense classifier head.

Scaling is controlled via alpha (width multiplier) and depth_multiplier (block repeats).
All channel counts are aligned to multiples of 8 for NPU vectorization.
"""

import math

import tensorflow as tf
from tensorflow.keras import layers, regularizers

from birdnet_stm32.models.blocks import _make_divisible
from birdnet_stm32.models.frontend import (
    RELEASE_RAW_BANK,
    RELEASE_RAW_MAGNITUDE,
    AudioFrontendLayer,
    hybrid_fft_bins,
    normalize_frontend_name,
)

HEAD_POOLINGS = ("gap",)
STAGE_WIDTHS = (32, 64, 128, 256)
DW_KERNEL_SIZES = (3, 5)


def ds_conv_block(
    x: tf.Tensor,
    out_ch: int,
    stride_f: int = 1,
    stride_t: int = 1,
    name: str = "ds",
    weight_decay: float = 1e-4,
    drop_rate: float = 0.1,
    dw_kernel_size: int = 3,
) -> tf.Tensor:
    """Depthwise-separable block (k x k DW + 1x1 PW) with optional residual.

    Args:
        x: Input tensor [B, H, W, C].
        out_ch: Output channels for pointwise conv.
        stride_f: Stride along frequency axis.
        stride_t: Stride along time axis.
        name: Base name for layers.
        weight_decay: L2 regularization for DW/PW kernels.
        drop_rate: Spatial dropout rate after PW BN.
        dw_kernel_size: Square depthwise kernel size.

    Returns:
        Output tensor [B, H', W', out_ch].
    """
    reg = regularizers.l2(weight_decay) if weight_decay and weight_decay > 0 else None
    in_ch = x.shape[-1]

    y = layers.DepthwiseConv2D(
        kernel_size=(dw_kernel_size, dw_kernel_size),
        strides=(stride_f, stride_t),
        padding="same",
        use_bias=False,
        depthwise_regularizer=reg,
        name=f"{name}_dw",
    )(x)
    y = layers.BatchNormalization(name=f"{name}_dw_bn")(y)
    y = layers.ReLU(max_value=6, name=f"{name}_dw_relu")(y)

    y = layers.Conv2D(
        filters=out_ch,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        kernel_regularizer=reg,
        name=f"{name}_pw",
    )(y)
    y = layers.BatchNormalization(name=f"{name}_pw_bn")(y)

    if drop_rate and drop_rate > 0:
        y = layers.SpatialDropout2D(drop_rate, name=f"{name}_drop")(y)

    # Residual connection when dimensions match
    if (stride_f == 1 and stride_t == 1) and (in_ch is not None and int(in_ch) == int(out_ch)):
        y = layers.Add(name=f"{name}_add")([x, y])

    y = layers.ReLU(max_value=6, name=f"{name}_pw_relu")(y)
    return y


def pooling_head(x: tf.Tensor, head_pooling: str = "gap") -> tf.Tensor:
    """Collapse a [B, F, T, C] feature map into a [B, C] embedding vector.

    Global average pooling over frequency and time. (A max + mean over time
    was measured and lost; ``gap`` is the only head.)

    Args:
        x: Feature map with static frequency and time dimensions.
        head_pooling: One of ``HEAD_POOLINGS``.

    Returns:
        Embedding tensor [B, C].
    """
    if head_pooling == "gap":
        return layers.GlobalAveragePooling2D(name="gap")(x)
    raise ValueError(f"head_pooling '{head_pooling}' not in {HEAD_POOLINGS}")


def build_dscnn_model(
    num_mels: int,
    spec_width: int,
    sample_rate: int,
    chunk_duration: int,
    embeddings_size: int,
    num_classes: int,
    audio_frontend: str = "hybrid",
    alpha: float = 1.0,
    depth_multiplier: int = 1,
    fft_length: int = 512,
    mag_scale: str = "pwl",
    raw_magnitude: str = RELEASE_RAW_MAGNITUDE,
    raw_overlap: int = 2,
    raw_bank: str = RELEASE_RAW_BANK,
    frontend_trainable: bool = False,
    dropout_rate: float = 0.5,
    weight_decay: float = 1e-4,
    head_pooling: str = "gap",
    dw_kernel_size: int = 3,
    stage_widths: tuple[int, ...] | list[int] | None = None,
) -> tf.keras.Model:
    """Build a DS-CNN model with a selectable audio frontend.

    Args:
        num_mels: Number of mel bins.
        spec_width: Spectrogram width (frames).
        sample_rate: Sampling rate (Hz).
        chunk_duration: Chunk duration (seconds).
        embeddings_size: Channels in the final embeddings layer.
        num_classes: Number of output classes.
        audio_frontend: 'librosa' | 'hybrid' | 'raw'.
        alpha: Width multiplier for the backbone.
        depth_multiplier: Repeats multiplier for DS blocks per stage.
        fft_length: FFT size for hybrid/librosa paths.
        mag_scale: Magnitude scaling ('pwl' | 'none').
        raw_magnitude: Quadrature combination for the raw frontend
            ('alpha_max' | 'l1' | 'halfwave'); ignored by other frontends.
        raw_overlap: Raw analysis window as a multiple of its hop; raw only.
        raw_bank: 'pair' or 'fused' quadrature filterbank; raw only. Defaults
            to the release layout, as raw_magnitude does.
        frontend_trainable: Make frontend sub-layers trainable.
        dropout_rate: Dropout rate before the classifier head.
        weight_decay: L2 regularization weight for DS-CNN blocks.
        head_pooling: Pooling head, one of ``HEAD_POOLINGS``.
        dw_kernel_size: Depthwise kernel size in stages 2-4; stage 1, which
            carries the largest feature map, stays 3x3.
        stage_widths: Base output channels of the four stages before the
            alpha multiplier; defaults to ``STAGE_WIDTHS``. When the last
            stage's width equals ``embeddings_size``, there is no separate
            embedding conv.

    Returns:
        Uncompiled DS-CNN Keras model.

    Raises:
        ValueError: If raw frontend exceeds STM32N6 input size limit (65536),
            or on an unknown head_pooling or dw_kernel_size, or stage_widths
            that are not four positive widths.
    """
    audio_frontend = normalize_frontend_name(audio_frontend)
    if head_pooling not in HEAD_POOLINGS:
        raise ValueError(f"head_pooling '{head_pooling}' not in {HEAD_POOLINGS}")
    if dw_kernel_size not in DW_KERNEL_SIZES:
        raise ValueError(f"dw_kernel_size {dw_kernel_size} not in {DW_KERNEL_SIZES}")
    stage_widths = tuple(STAGE_WIDTHS if stage_widths is None else stage_widths)
    if len(stage_widths) != 4 or any(int(w) <= 0 for w in stage_widths):
        raise ValueError(f"stage_widths must be four positive widths, got {stage_widths}")

    # Enforce STM32N6 constraint for raw frontend
    if audio_frontend == "raw":
        T = int(sample_rate * chunk_duration)
        if T >= (1 << 16):
            raise ValueError(
                f"STM32N6 constraint: raw input length (sample_rate*chunk_duration={T}) must be < 65536.\n"
                f"Use --sample_rate 16000, --chunk_duration 2, or --audio_frontend hybrid/librosa."
            )

    # Select input shape and frontend mode
    if audio_frontend == "librosa":
        inputs = tf.keras.Input(shape=(num_mels, spec_width, 1), name="mel_spectrogram_input")
        x = AudioFrontendLayer(
            mode="precomputed",
            mel_bins=num_mels,
            spec_width=spec_width,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            fft_length=fft_length,
            mag_scale=mag_scale,
            is_trainable=frontend_trainable,
            name="audio_frontend",
        )(inputs)
    elif audio_frontend == "hybrid":
        fft_bins = hybrid_fft_bins(fft_length)
        inputs = tf.keras.Input(shape=(fft_bins, spec_width, 1), name="linear_spectrogram_input")
        x = AudioFrontendLayer(
            mode="hybrid",
            mel_bins=num_mels,
            spec_width=spec_width,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            fft_length=fft_length,
            mag_scale=mag_scale,
            is_trainable=frontend_trainable,
            name="audio_frontend",
        )(inputs)
    elif audio_frontend == "raw":
        inputs = tf.keras.Input(shape=(int(chunk_duration * sample_rate), 1), name="raw_audio_input")
        x = AudioFrontendLayer(
            mode="raw",
            mel_bins=num_mels,
            spec_width=spec_width,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            fft_length=fft_length,
            mag_scale=mag_scale,
            raw_magnitude=raw_magnitude,
            raw_overlap=raw_overlap,
            raw_bank=raw_bank,
            is_trainable=frontend_trainable,
            name="audio_frontend",
        )(inputs)
    else:
        raise ValueError(f"Invalid audio_frontend: {audio_frontend}")

    # Stem (3x3, stride 1x2) to lift channels
    stem_ch = _make_divisible(int(16 * alpha), 8)
    x = layers.Conv2D(stem_ch, (3, 3), strides=(1, 2), padding="same", use_bias=False, name="stem_conv")(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(max_value=6, name="stem_relu")(x)

    # Four stages: (base_filters, base_repeats, (stride_f, stride_t))
    base_filters = [int(w) for w in stage_widths]
    base_repeats = [2, 3, 4, 2]
    base_strides = [(2, 2), (2, 2), (2, 2), (2, 2)]

    for si, (bf, br, (sf, st)) in enumerate(zip(base_filters, base_repeats, base_strides, strict=True), start=1):
        out_ch = _make_divisible(int(bf * alpha), 8)
        reps = max(1, int(math.ceil(br * depth_multiplier)))
        k = 3 if si == 1 else dw_kernel_size

        x = ds_conv_block(
            x, out_ch, stride_f=sf, stride_t=st, name=f"stage{si}_ds1", weight_decay=weight_decay, dw_kernel_size=k
        )
        for bi in range(2, reps + 1):
            x = ds_conv_block(
                x, out_ch, stride_f=1, stride_t=1, name=f"stage{si}_ds{bi}", weight_decay=weight_decay, dw_kernel_size=k
            )

    # Final 1x1 conv to embeddings
    emb_ch = _make_divisible(int(embeddings_size), 8)
    if not (x.shape[-1] is not None and int(x.shape[-1]) == int(emb_ch)):
        x = layers.Conv2D(emb_ch, (1, 1), strides=(1, 1), padding="same", use_bias=False, name="emb_conv")(x)
        x = layers.BatchNormalization(name="emb_bn")(x)
        x = layers.ReLU(max_value=6, name="emb_relu")(x)

    # Head
    x = pooling_head(x, head_pooling)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    # Keep the head in float32: under a mixed_float16 policy a float16 sigmoid
    # saturates well before the loss does, which stalls training.
    outputs = layers.Dense(num_classes, activation="sigmoid", name="pred", dtype="float32")(x)
    return tf.keras.models.Model(inputs, outputs, name="dscnn_audio")
