"""Quantization-Aware Training (QAT) via shadow-weight fake-quantization.

Manual QAT for Keras 3 (tensorflow-model-optimization is incompatible).
Injects INT8 quantization noise during fine-tuning so the model learns
weights that survive Post-Training Quantization with minimal accuracy loss.

No FakeQuant ops remain in the saved model — full STM32N6 NPU compatibility.

Usage::

    python -m birdnet_stm32 train --data_path_train data/train \\
        --qat --checkpoint_path checkpoints/best_model.keras \\
        --epochs 10 --learning_rate 0.0001
"""

import math
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Layer types that receive weight fake-quantization during QAT.
QUANTIZABLE_TYPES = (layers.Conv2D, layers.DepthwiseConv2D, layers.Dense)

# Layer name prefixes to skip (custom/frontend layers).
SKIP_PREFIXES = ("audio_frontend",)


def fake_quantize_weights(
    w: np.ndarray,
    num_bits: int = 8,
    per_channel: bool = True,
    channel_axis: int = -1,
) -> np.ndarray:
    """Simulate INT8 quantization on a weight array (quantize then dequantize).

    Args:
        w: Weight tensor (float32).
        num_bits: Quantization bit width.
        per_channel: Per-channel (True) or per-tensor (False) quantization.
        channel_axis: Axis for per-channel quantization.

    Returns:
        Fake-quantized weight tensor (float32, same shape).
    """
    qmax = (1 << num_bits) - 1  # 255 for 8-bit

    if per_channel and w.ndim > 1:
        reduce_axes = tuple(i for i in range(w.ndim) if i != channel_axis)
        w_min = w.min(axis=reduce_axes, keepdims=True)
        w_max = w.max(axis=reduce_axes, keepdims=True)
    else:
        w_min = w.min()
        w_max = w.max()

    scale = (w_max - w_min) / qmax
    scale = np.maximum(scale, 1e-10)

    w_q = np.round((w - w_min) / scale) * scale + w_min
    return w_q.astype(np.float32)


def _channel_axis(layer: tf.keras.layers.Layer) -> int:
    """Get the per-channel quantization axis for a layer's kernel."""
    if isinstance(layer, layers.DepthwiseConv2D):
        return -2  # kernel shape: [H, W, C_in, depth_multiplier]
    return -1  # Conv2D: [H, W, C_in, C_out], Dense: [in, out]


def _is_quantizable(layer: tf.keras.layers.Layer) -> bool:
    """Check if a layer should be fake-quantized during QAT."""
    if not isinstance(layer, QUANTIZABLE_TYPES):
        return False
    if not layer.trainable_weights:
        return False
    return not any(layer.name.startswith(p) for p in SKIP_PREFIXES)


class QATCallback(tf.keras.callbacks.Callback):
    """Shadow-weight fake-quantization callback for QAT fine-tuning.

    Before each training step:

    1. Save full-precision (FP32) weight copies.
    2. Replace kernel weights with fake-quantized (INT8-simulated) versions.
       Biases are left at full precision (INT32 in TFLite).

    After each training step:

    1. Compute optimizer delta (post-update weight minus pre-update quantized weight).
    2. Apply delta to full-precision weights.

    This ensures gradients flow through quantized weights (approximate STE)
    while maintaining full-precision weight accumulation across steps.

    Args:
        num_bits: Quantization bit width (default: 8).
        per_channel: Per-channel (True) or per-tensor (False) quantization.
    """

    def __init__(self, num_bits: int = 8, per_channel: bool = True):
        super().__init__()
        self.num_bits = num_bits
        self.per_channel = per_channel
        self._qat_layers: list[tf.keras.layers.Layer] = []
        self._tracked: list[tuple[tf.Variable, np.ndarray, np.ndarray]] = []

    def on_train_begin(self, logs=None):
        """Identify quantizable layers and report statistics."""
        self._qat_layers = [l for l in self.model.layers if _is_quantizable(l)]
        n_params = sum(
            sum(int(np.prod(w.shape)) for w in l.trainable_weights if "bias" not in w.name)
            for l in self._qat_layers
        )
        print(
            f"[QAT] {len(self._qat_layers)} layers, {n_params:,} kernel params, "
            f"{self.num_bits}-bit, per_channel={self.per_channel}"
        )

    def on_train_batch_begin(self, batch, logs=None):
        """Save FP weights and inject fake-quantized versions for forward pass."""
        self._tracked = []
        for layer in self._qat_layers:
            axis = _channel_axis(layer)
            for var in layer.trainable_weights:
                if "bias" in var.name:
                    continue  # biases stay at full precision (INT32 in TFLite)
                fp = var.numpy().copy()
                fq = fake_quantize_weights(fp, self.num_bits, self.per_channel, axis)
                var.assign(fq)
                self._tracked.append((var, fp, fq))

    def on_train_batch_end(self, batch, logs=None):
        """Transfer optimizer's gradient update to full-precision weights."""
        for var, fp, q in self._tracked:
            # delta = what the optimizer changed = (post-update) - (pre-update quantized)
            delta = var.numpy() - q
            var.assign(fp + delta)


def freeze_batch_norm(model: tf.keras.Model) -> int:
    """Freeze all BatchNormalization layers (standard for QAT fine-tuning).

    Prevents BN running statistics from drifting due to quantization noise.

    Args:
        model: Keras model.

    Returns:
        Number of frozen BN layers.
    """
    count = 0
    for layer in model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            count += 1
    return count


def _detect_loss(model: tf.keras.Model) -> str:
    """Detect the appropriate loss function from the model's output activation."""
    last_layer = model.layers[-1]
    if isinstance(last_layer, layers.Dense):
        activation = last_layer.get_config().get("activation", "linear")
        if activation == "sigmoid":
            return "binary_crossentropy"
    return "categorical_crossentropy"


def run_qat(args) -> None:
    """Run QAT fine-tuning from CLI args.

    Loads a pretrained model, freezes BatchNorm layers, and fine-tunes
    with shadow-weight fake-quantization. Saves the QAT model as
    ``{checkpoint_path_stem}_qat.keras``.

    Args:
        args: Parsed CLI arguments (checkpoint_path, data_path_train,
              epochs, learning_rate, batch_size, etc.).
    """
    from birdnet_stm32.data.dataset import (
        load_file_paths_from_directory,
        upsample_minority_classes,
    )
    from birdnet_stm32.data.generator import load_dataset
    from birdnet_stm32.models.frontend import AudioFrontendLayer
    from birdnet_stm32.models.magnitude import MagnitudeScalingLayer
    from birdnet_stm32.training.config import ModelConfig
    from birdnet_stm32.training.trainer import train_model

    # --- Load pretrained model -----------------------------------------------
    if not os.path.isfile(args.checkpoint_path):
        raise FileNotFoundError(
            f"QAT requires a pretrained model. Not found: {args.checkpoint_path}"
        )

    print(f"[QAT] Loading pretrained model from {args.checkpoint_path}")
    model = tf.keras.models.load_model(
        args.checkpoint_path,
        compile=False,
        custom_objects={
            "AudioFrontendLayer": AudioFrontendLayer,
            "MagnitudeScalingLayer": MagnitudeScalingLayer,
        },
    )

    # --- Load model config ---------------------------------------------------
    cfg_path = os.path.splitext(args.checkpoint_path)[0] + "_model_config.json"
    if hasattr(args, "model_config") and args.model_config:
        cfg_path = args.model_config
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Model config not found: {cfg_path}")
    cfg = ModelConfig.load(cfg_path)

    # --- Freeze BatchNorm ----------------------------------------------------
    n_frozen = freeze_batch_norm(model)
    print(f"[QAT] Frozen {n_frozen} BatchNorm layers")

    # --- Detect loss function ------------------------------------------------
    loss_fn = _detect_loss(model)
    is_multilabel = loss_fn == "binary_crossentropy"
    print(f"[QAT] Loss: {loss_fn}, multilabel={is_multilabel}")

    # --- Prepare datasets ----------------------------------------------------
    file_paths, classes = load_file_paths_from_directory(args.data_path_train)
    split_idx = int(len(file_paths) * (1 - args.val_split))
    train_paths = file_paths[:split_idx]
    val_paths = file_paths[split_idx:]
    print(f"[QAT] Training on {len(train_paths)} files, validating on {len(val_paths)} files")

    if args.upsample_ratio and 0 < args.upsample_ratio < 1.0:
        train_paths = upsample_minority_classes(train_paths, classes, args.upsample_ratio)

    common_kwargs = dict(
        sample_rate=cfg.sample_rate,
        max_duration=args.max_duration,
        chunk_duration=cfg.chunk_duration,
        spec_width=cfg.spec_width,
        mel_bins=cfg.num_mels,
        fft_length=cfg.fft_length,
        mag_scale=cfg.mag_scale,
    )
    # No mixup or augmentation during QAT fine-tuning.
    train_dataset = load_dataset(
        train_paths,
        classes,
        audio_frontend=cfg.audio_frontend,
        batch_size=args.batch_size,
        mixup_alpha=0.0,
        mixup_probability=0.0,
        random_offset=True,
        snr_threshold=0.1,
        spec_augment=False,
        **common_kwargs,
    )
    val_dataset = load_dataset(
        val_paths,
        classes,
        audio_frontend=cfg.audio_frontend,
        batch_size=args.batch_size,
        mixup_alpha=0.0,
        mixup_probability=0.0,
        random_offset=False,
        snr_threshold=0.5,
        spec_augment=False,
        **common_kwargs,
    )

    steps_per_epoch = max(1, math.ceil(len(train_paths) / float(args.batch_size)))
    val_steps = max(1, math.ceil(len(val_paths) / float(args.batch_size)))

    # --- QAT output path -----------------------------------------------------
    qat_path = args.checkpoint_path.replace(".keras", "_qat.keras")

    # --- Fine-tune with QAT --------------------------------------------------
    qat_cb = QATCallback(num_bits=8, per_channel=True)
    print(f"[QAT] Fine-tuning for {args.epochs} epochs at LR={args.learning_rate}")
    train_model(
        model,
        train_dataset,
        val_dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        checkpoint_path=qat_path,
        steps_per_epoch=steps_per_epoch,
        val_steps=val_steps,
        is_multilabel=is_multilabel,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.grad_clip,
        extra_callbacks=[qat_cb],
    )

    print(f"[QAT] Fine-tuned model saved to {qat_path}")
    print(f"[QAT] Convert with: python -m birdnet_stm32 convert "
          f"--checkpoint_path {qat_path} --model_config {cfg_path}")
