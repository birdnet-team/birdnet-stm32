"""DS-CNN (depthwise-separable CNN) model architecture for audio classification.

The model consists of:
- An AudioFrontendLayer (from frontend.py) for feature extraction.
- A stem convolution to lift channels.
- Four stages of depthwise-separable blocks with stride-2 downsampling.
- Global average pooling, dropout, and a dense classifier head.

Scaling is controlled via alpha (width multiplier) and depth_multiplier (block repeats).
All channel counts are aligned to multiples of 8 for NPU vectorization.
"""

import math

import tensorflow as tf
from tensorflow.keras import layers, regularizers

from birdnet_stm32.models.frontend import AudioFrontendLayer, normalize_frontend_name


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


def ds_conv_block(
    x: tf.Tensor,
    out_ch: int,
    stride_f: int = 1,
    stride_t: int = 1,
    name: str = "ds",
    weight_decay: float = 1e-4,
    drop_rate: float = 0.1,
) -> tf.Tensor:
    """Depthwise-separable block (3x3 DW + 1x1 PW) with optional residual.

    Args:
        x: Input tensor [B, H, W, C].
        out_ch: Output channels for pointwise conv.
        stride_f: Stride along frequency axis.
        stride_t: Stride along time axis.
        name: Base name for layers.
        weight_decay: L2 regularization for DW/PW kernels.
        drop_rate: Spatial dropout rate after PW BN.

    Returns:
        Output tensor [B, H', W', out_ch].
    """
    reg = regularizers.l2(weight_decay) if weight_decay and weight_decay > 0 else None
    in_ch = x.shape[-1]

    y = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
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


def build_dscnn_model(
    num_mels: int,
    spec_width: int,
    sample_rate: int,
    chunk_duration: int,
    embeddings_size: int,
    num_classes: int,
    audio_frontend: str = "precomputed",
    alpha: float = 1.0,
    depth_multiplier: int = 1,
    fft_length: int = 512,
    mag_scale: str = "none",
    frontend_trainable: bool = False,
    class_activation: str = "softmax",
    dropout_rate: float = 0.5,
) -> tf.keras.Model:
    """Build a DS-CNN model with a selectable audio frontend.

    Args:
        num_mels: Number of mel bins.
        spec_width: Spectrogram width (frames).
        sample_rate: Sampling rate (Hz).
        chunk_duration: Chunk duration (seconds).
        embeddings_size: Channels in the final embeddings layer.
        num_classes: Number of output classes.
        audio_frontend: 'librosa' | 'hybrid' | 'raw' (deprecated: 'precomputed', 'tf').
        alpha: Width multiplier for the backbone.
        depth_multiplier: Repeats multiplier for DS blocks per stage.
        fft_length: FFT size for hybrid/librosa paths.
        mag_scale: Magnitude scaling ('pcen' | 'pwl' | 'db' | 'none').
        frontend_trainable: Make frontend sub-layers trainable.
        class_activation: 'softmax' or 'sigmoid' for the classifier head.
        dropout_rate: Dropout rate before the classifier head.

    Returns:
        Uncompiled DS-CNN Keras model.

    Raises:
        ValueError: If raw frontend exceeds STM32N6 input size limit (65536).
    """
    audio_frontend = normalize_frontend_name(audio_frontend)

    # Enforce STM32N6 constraint for raw frontend
    if audio_frontend == "raw":
        T = int(sample_rate) * int(chunk_duration)
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
        fft_bins = fft_length // 2 + 1
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
        x = ds_conv_block(x, out_ch, stride_f=sf, stride_t=st, name=f"stage{si}_ds1")
        for bi in range(2, reps + 1):
            x = ds_conv_block(x, out_ch, stride_f=1, stride_t=1, name=f"stage{si}_ds{bi}")

    # Final 1x1 conv to embeddings
    emb_ch = _make_divisible(int(embeddings_size), 8)
    if not (x.shape[-1] is not None and int(x.shape[-1]) == int(emb_ch)):
        x = layers.Conv2D(emb_ch, (1, 1), strides=(1, 1), padding="same", use_bias=False, name="emb_conv")(x)
        x = layers.BatchNormalization(name="emb_bn")(x)
        x = layers.ReLU(max_value=6, name="emb_relu")(x)

    # Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation=class_activation, name="pred")(x)
    return tf.keras.models.Model(inputs, outputs, name="dscnn_audio")
