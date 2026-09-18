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
from birdnet_stm32.models.frontend import AudioFrontendLayer, hybrid_fft_bins, normalize_frontend_name

HEAD_POOLINGS = ("gap", "freq_mean_time_maxmean")
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

    ``gap`` averages over frequency and time. ``freq_mean_time_maxmean``
    averages over frequency, then adds the max and the mean over time, so a
    call that fills a few frames of the window is not averaged away. It adds no
    trainable weights.

    The frequency mean is a frozen depthwise convolution with a constant
    1/F kernel rather than a pooling op: on the STM32N6 an average that
    collapses frequency but keeps time (``AveragePool`` or ``MEAN`` alike) falls
    back to the Cortex-M55, while the convolution stays on the NPU. For F = 4
    the kernel value 0.25 is exact on the per-channel INT8 grid.

    Args:
        x: Feature map with static frequency and time dimensions.
        head_pooling: One of ``HEAD_POOLINGS``.

    Returns:
        Embedding tensor [B, C].
    """
    if head_pooling == "gap":
        return layers.GlobalAveragePooling2D(name="gap")(x)
    if head_pooling == "freq_mean_time_maxmean":
        n_freq, n_time = int(x.shape[1]), int(x.shape[2])
        x = layers.DepthwiseConv2D(
            kernel_size=(n_freq, 1),
            padding="valid",
            use_bias=False,
            depthwise_initializer=tf.keras.initializers.Constant(1.0 / n_freq),
            trainable=False,
            name="pool_freq_mean",
        )(x)
        t_max = layers.MaxPooling2D(pool_size=(1, n_time), name="pool_time_max")(x)
        t_mean = layers.GlobalAveragePooling2D(keepdims=True, name="pool_time_mean")(x)
        x = layers.Add(name="pool_time_maxmean")([t_max, t_mean])
        return layers.Flatten(name="pool_flatten")(x)
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
    frontend_trainable: bool = False,
    dropout_rate: float = 0.5,
    weight_decay: float = 1e-4,
    head_pooling: str = "gap",
    dw_kernel_size: int = 3,
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
        frontend_trainable: Make frontend sub-layers trainable.
        dropout_rate: Dropout rate before the classifier head.
        weight_decay: L2 regularization weight for DS-CNN blocks.
        head_pooling: Pooling head, one of ``HEAD_POOLINGS``.
        dw_kernel_size: Depthwise kernel size in stages 2-4; stage 1, which
            carries the largest feature map, stays 3x3.

    Returns:
        Uncompiled DS-CNN Keras model.

    Raises:
        ValueError: If raw frontend exceeds STM32N6 input size limit (65536),
            or on an unknown head_pooling or dw_kernel_size.
    """
    audio_frontend = normalize_frontend_name(audio_frontend)
    if head_pooling not in HEAD_POOLINGS:
        raise ValueError(f"head_pooling '{head_pooling}' not in {HEAD_POOLINGS}")
    if dw_kernel_size not in DW_KERNEL_SIZES:
        raise ValueError(f"dw_kernel_size {dw_kernel_size} not in {DW_KERNEL_SIZES}")

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
    base_filters = [32, 64, 128, 256]
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
