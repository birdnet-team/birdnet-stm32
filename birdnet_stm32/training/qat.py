"""Keras 3 quantization-aware fine-tuning for full-INT8 TFLite deployment.

The training graph simulates both quantized kernels and per-tensor activation
requantization.  It shares variables with a clean deployment graph, so the
saved checkpoint contains no FakeQuant operators or training-only wrappers.
"""

import argparse
import json
import math
import os
from collections.abc import Iterable

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

QUANTIZABLE_TYPES = (layers.Conv2D, layers.DepthwiseConv2D, layers.Dense)
ACTIVATION_BOUNDARY_TYPES = (
    layers.BatchNormalization,
    layers.ReLU,
    layers.Add,
    layers.Multiply,
    layers.Dense,
    layers.GlobalAveragePooling2D,
)


def fake_quantize_weights(
    w: np.ndarray,
    num_bits: int = 8,
    per_channel: bool = True,
    channel_axis: int = -1,
) -> np.ndarray:
    """Quantize and dequantize a kernel on TFLite's symmetric INT8 grid."""
    qmax = (1 << (num_bits - 1)) - 1
    if per_channel and w.ndim > 1:
        axis = channel_axis % w.ndim
        reduce_axes = tuple(i for i in range(w.ndim) if i != axis)
        amax = np.max(np.abs(w), axis=reduce_axes, keepdims=True)
    else:
        amax = np.max(np.abs(w))
    scale = np.maximum(amax / qmax, 1e-12)
    return (np.clip(np.round(w / scale), -qmax, qmax) * scale).astype(np.float32)


def _all_layers(model: tf.keras.Model) -> list[tf.keras.layers.Layer]:
    """Return every nested layer exactly once."""
    flattened = model._flatten_layers(include_self=False, recursive=True)  # noqa: SLF001
    return list(dict.fromkeys(flattened))


def _channel_axis(layer: tf.keras.layers.Layer) -> int:
    """Return TFLite's output-channel axis for a kernel."""
    return -2 if isinstance(layer, layers.DepthwiseConv2D) else -1


def _is_quantizable(layer: tf.keras.layers.Layer) -> bool:
    """Return whether a layer owns a kernel quantized by full-INT8 TFLite."""
    return isinstance(layer, QUANTIZABLE_TYPES) and bool(layer.trainable_weights)


def _is_activation_boundary(layer: tf.keras.layers.Layer) -> bool:
    """Return whether an outer-graph tensor is requantized during inference."""
    if isinstance(layer, layers.BatchNormalization):
        # Conv + BN + ReLU is folded into one quantized TFLite operator. Keep a
        # boundary only for linear project BNs whose output feeds an Add.
        consumers = [node.operation for node in layer._outbound_nodes]  # noqa: SLF001
        return not consumers or not all(isinstance(consumer, layers.ReLU) for consumer in consumers)
    return isinstance(layer, ACTIVATION_BOUNDARY_TYPES[1:]) or layer.__class__.__name__ == "AudioFrontendLayer"


@tf.keras.utils.register_keras_serializable(package="birdnet_stm32")
class FakeQuantActivation(layers.Layer):
    """Static per-tensor INT8 fake quantizer with an asymmetric zero point."""

    def __init__(self, minimum: float, maximum: float, num_bits: int = 8, **kwargs):
        super().__init__(trainable=False, **kwargs)
        minimum = min(float(minimum), 0.0)
        maximum = max(float(maximum), 0.0)
        if maximum - minimum < 1e-6:
            maximum = minimum + 1e-6
        self.minimum = minimum
        self.maximum = maximum
        self.num_bits = int(num_bits)

    def call(self, inputs):
        """Apply the same scalar affine grid used for TFLite activations."""
        qmin = -(1 << (self.num_bits - 1))
        qmax = (1 << (self.num_bits - 1)) - 1
        scale = tf.cast((self.maximum - self.minimum) / (qmax - qmin), inputs.dtype)
        zero_point = tf.clip_by_value(
            tf.round(tf.cast(qmin, inputs.dtype) - tf.cast(self.minimum, inputs.dtype) / scale),
            tf.cast(qmin, inputs.dtype),
            tf.cast(qmax, inputs.dtype),
        )
        quantized = tf.clip_by_value(tf.round(inputs / scale) + zero_point, qmin, qmax)
        dequantized = (quantized - zero_point) * scale
        # Explicit straight-through estimator. Unlike TensorFlow's FakeQuant
        # gradient kernel, this is supported by deterministic GPU execution.
        return inputs + tf.stop_gradient(dequantized - inputs)

    def get_config(self):
        """Return serializable quantizer settings."""
        config = super().get_config()
        config.update({"minimum": self.minimum, "maximum": self.maximum, "num_bits": self.num_bits})
        return config


def _fake_quantize_kernel_tensor(kernel: tf.Tensor, channel_axis: int) -> tf.Tensor:
    """Differentiably fake-quantize a kernel per output channel."""
    rank = len(kernel.shape)
    axis = channel_axis % rank
    transposed = axis != rank - 1
    if transposed:
        permutation = [index for index in range(rank) if index != axis] + [axis]
        inverse = np.argsort(permutation).tolist()
        kernel = tf.transpose(kernel, permutation)
    reduce_axes = tuple(range(rank - 1))
    maximum = tf.stop_gradient(tf.reduce_max(tf.abs(kernel), axis=reduce_axes, keepdims=True))
    scale = tf.maximum(maximum / tf.cast(127.0, kernel.dtype), tf.cast(1e-12, kernel.dtype))
    dequantized = tf.clip_by_value(tf.round(kernel / scale), -127.0, 127.0) * scale
    quantized = kernel + tf.stop_gradient(dequantized - kernel)
    return tf.transpose(quantized, inverse) if transposed else quantized


class _QuantizedKernelCall(layers.Layer):
    """Call a built Conv/DWConv/Dense layer with an STE-quantized kernel."""

    def __init__(self, target: tf.keras.layers.Layer, **kwargs):
        super().__init__(trainable=True, **kwargs)
        self.target = target

    def call(self, inputs):
        """Run the target math without replacing its full-precision variable."""
        kernel = _fake_quantize_kernel_tensor(self.target.kernel, _channel_axis(self.target))
        if isinstance(self.target, layers.Conv2D):
            output = self.target.convolution_op(inputs, kernel)
        elif isinstance(self.target, layers.DepthwiseConv2D):
            if self.target.data_format != "channels_last":
                raise ValueError("QAT supports channels_last DepthwiseConv2D only")
            output = tf.nn.depthwise_conv2d(
                inputs,
                kernel,
                strides=(1, *self.target.strides, 1),
                padding=self.target.padding.upper(),
                data_format="NHWC",
                dilations=self.target.dilation_rate,
            )
        elif isinstance(self.target, layers.Dense):
            output = tf.linalg.matmul(inputs, kernel)
        else:  # pragma: no cover - constructor is internal and type-guarded
            raise TypeError(f"Unsupported quantized kernel layer: {type(self.target).__name__}")
        if self.target.bias is not None:
            output = output + self.target.bias
        return self.target.activation(output)


class _ActivationRangeCollector:
    """Observe internal custom-layer tensors without changing their values."""

    def __init__(self):
        self.ranges: dict[str, list[float]] = {}

    def activation(self, name: str, inputs):
        """Record one tensor's scalar range and return it unchanged."""
        array = np.asarray(inputs)
        values = self.ranges.setdefault(name, [float("inf"), -float("inf")])
        values[0] = min(values[0], float(np.min(array)), 0.0)
        values[1] = max(values[1], float(np.max(array)), 0.0)
        return inputs

    def kernel(self, layer, inputs):
        """Call an ordinary full-precision kernel without adding a boundary."""
        return layer(inputs)


class _FrontendQuantizationHook:
    """Apply static activation and kernel fake quantization inside a frontend."""

    def __init__(self, activation_ranges: dict[str, tuple[float, float]]):
        self.activation_ranges = activation_ranges
        self._activations: dict[str, FakeQuantActivation] = {}
        self._kernels: dict[str, _QuantizedKernelCall] = {}

    def activation(self, name: str, inputs):
        """Fake-quantize an internal activation on its calibrated grid."""
        if name not in self.activation_ranges:
            raise KeyError(f"Missing calibrated QAT activation range: {name}")
        if name not in self._activations:
            minimum, maximum = self.activation_ranges[name]
            self._activations[name] = FakeQuantActivation(
                minimum,
                maximum,
                name=f"{name}_fake_quant",
            )
        return self._activations[name](inputs)

    def kernel(self, layer, inputs):
        """Run an internal kernel with per-channel INT8 fake quantization."""
        if layer.name not in self._kernels:
            self._kernels[layer.name] = _QuantizedKernelCall(
                layer,
                name=f"{layer.name}_quantized_kernel",
            )
        return self._kernels[layer.name](inputs)


class _DistilledQATModel(tf.keras.Model):
    """Train a quantized student against labels and a frozen float teacher."""

    def __init__(
        self,
        student: tf.keras.Model,
        teacher: tf.keras.Model,
        distillation_weight: float = 1.0,
    ):
        super().__init__(inputs=student.inputs, outputs=student.outputs, name=student.name)
        teacher.trainable = False
        self.teacher = teacher
        self.distillation_weight = float(distillation_weight)
        self.distillation_metric = tf.keras.metrics.Mean(name="distillation_kl")

    def compute_loss(self, x, y, y_pred, sample_weight=None, training=True):
        """Add multi-label Bernoulli KL divergence to supervised BCE."""
        supervised = super().compute_loss(
            x=x,
            y=y,
            y_pred=y_pred,
            sample_weight=sample_weight,
            training=training,
        )
        teacher_pred = tf.stop_gradient(self.teacher(x, training=False))
        epsilon = tf.cast(tf.keras.backend.epsilon(), y_pred.dtype)
        teacher_pred = tf.clip_by_value(teacher_pred, epsilon, 1.0 - epsilon)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = tf.keras.losses.binary_crossentropy(teacher_pred, y_pred)
        teacher_entropy = tf.keras.losses.binary_crossentropy(teacher_pred, teacher_pred)
        divergence = tf.reduce_mean(cross_entropy - teacher_entropy)
        self.distillation_metric.update_state(divergence)
        return supervised + self.distillation_weight * divergence


def calibrate_activation_ranges(
    model: tf.keras.Model,
    dataset: Iterable,
    max_samples: int = 64,
) -> dict[str, tuple[float, float]]:
    """Measure scalar activation ranges on real inputs for QAT initialization."""
    boundaries = [layer for layer in model.layers if _is_activation_boundary(layer)]
    if not boundaries:
        raise ValueError("Model has no supported activation quantization boundaries")
    probe = tf.keras.Model(model.inputs, [layer.output for layer in boundaries])
    ranges = {layer.name: [float("inf"), -float("inf")] for layer in boundaries}
    ranges["__input__"] = [float("inf"), -float("inf")]
    frontends = [layer for layer in _all_layers(model) if layer.__class__.__name__ == "AudioFrontendLayer"]
    collector = _ActivationRangeCollector()
    for frontend in frontends:
        frontend.set_quantization_hook(collector)

    seen = 0
    try:
        for batch in dataset:
            inputs = batch[0] if isinstance(batch, (tuple, list)) and len(batch) == 2 else batch
            inputs = np.asarray(inputs)
            for sample in inputs:
                ranges["__input__"][0] = min(ranges["__input__"][0], float(np.min(sample)), 0.0)
                ranges["__input__"][1] = max(ranges["__input__"][1], float(np.max(sample)), 0.0)
                outputs = probe([sample[None]], training=False)
                if not isinstance(outputs, (tuple, list)):
                    outputs = [outputs]
                for layer, output in zip(boundaries, outputs, strict=True):
                    array = np.asarray(output)
                    ranges[layer.name][0] = min(ranges[layer.name][0], float(np.min(array)), 0.0)
                    ranges[layer.name][1] = max(ranges[layer.name][1], float(np.max(array)), 0.0)
                seen += 1
                if seen >= max_samples:
                    break
            if seen >= max_samples:
                break
    finally:
        for frontend in frontends:
            frontend.set_quantization_hook(None)
    if seen == 0:
        raise ValueError("Activation calibration dataset yielded no samples")

    ranges.update(collector.ranges)
    result = {name: (values[0], values[1]) for name, values in ranges.items()}
    print(f"[QAT] Calibrated {len(result)} activation tensors on {seen} samples")
    return result


def build_qat_model(
    deployment_model: tf.keras.Model,
    activation_ranges: dict[str, tuple[float, float]],
) -> tf.keras.Model:
    """Build an activation-fake-quant graph sharing deployment model weights."""

    def clone_function(layer):
        if layer.__class__.__name__ != "AudioFrontendLayer":
            return layer
        clone = layer.__class__.from_config(layer.get_config())
        clone.build(tuple(layer.input.shape))
        clone.set_weights(layer.get_weights())
        clone.set_quantization_hook(_FrontendQuantizationHook(activation_ranges))
        return clone

    def call_function(layer, *args, **kwargs):
        if _is_quantizable(layer):
            output = _QuantizedKernelCall(layer, name=f"{layer.name}_quantized_kernel")(*args, **kwargs)
        else:
            output = layer(*args, **kwargs)
        if layer.name in activation_ranges:
            minimum, maximum = activation_ranges[layer.name]
            output = FakeQuantActivation(
                minimum,
                maximum,
                name=f"{layer.name}_fake_quant",
            )(output)
        return output

    inner_model = tf.keras.models.clone_model(
        deployment_model,
        clone_function=clone_function,
        call_function=call_function,
    )
    raw_inputs = tf.keras.Input(
        shape=deployment_model.input_shape[1:],
        dtype=deployment_model.input_dtype,
        name="qat_input",
    )
    minimum, maximum = activation_ranges["__input__"]
    quantized_inputs = FakeQuantActivation(minimum, maximum, name="input_fake_quant")(raw_inputs)
    return tf.keras.Model(raw_inputs, inner_model(quantized_inputs), name=f"{deployment_model.name}_qat")


def sync_frontend_weights(qat_model: tf.keras.Model, deployment_model: tf.keras.Model) -> None:
    """Copy separately cloned custom-frontend weights into the clean model."""
    qat_frontends = {
        layer.name: layer
        for layer in _all_layers(qat_model)
        if layer.__class__.__name__ == "AudioFrontendLayer" and getattr(layer, "_quantization_hook", None) is not None
    }
    deployment_frontends = {
        layer.name: layer for layer in _all_layers(deployment_model) if layer.__class__.__name__ == "AudioFrontendLayer"
    }
    if qat_frontends.keys() != deployment_frontends.keys():
        raise ValueError("QAT and deployment frontend layers do not match")
    for name, frontend in qat_frontends.items():
        deployment_frontends[name].set_weights(frontend.get_weights())


def freeze_batch_norm(model: tf.keras.Model) -> int:
    """Freeze all BatchNormalization layers, including nested frontend BN."""
    batch_norms = [layer for layer in _all_layers(model) if isinstance(layer, layers.BatchNormalization)]
    for layer in batch_norms:
        layer.trainable = False
    return len(batch_norms)


def _detect_loss(model: tf.keras.Model) -> str:
    """Return the multi-label classifier loss."""
    del model
    return "binary_crossentropy"


def run_qat(args: argparse.Namespace) -> None:
    """Fine-tune a pretrained model against weight and activation INT8 noise."""
    from birdnet_stm32.data.dataset import (
        load_classes_file,
        load_file_paths_from_directory,
        upsample_minority_classes,
    )
    from birdnet_stm32.data.generator import load_dataset
    from birdnet_stm32.models.frontend import AudioFrontendLayer
    from birdnet_stm32.models.magnitude import MagnitudeScalingLayer
    from birdnet_stm32.training.config import ModelConfig
    from birdnet_stm32.training.trainer import train_model

    if not os.path.isfile(args.checkpoint_path):
        raise FileNotFoundError(f"QAT requires a pretrained model: {args.checkpoint_path}")
    print(f"[QAT] Loading pretrained model from {args.checkpoint_path}")
    deployment_model = tf.keras.models.load_model(
        args.checkpoint_path,
        compile=False,
        custom_objects={
            "AudioFrontendLayer": AudioFrontendLayer,
            "MagnitudeScalingLayer": MagnitudeScalingLayer,
        },
    )
    teacher_model = tf.keras.models.load_model(
        args.checkpoint_path,
        compile=False,
        custom_objects={
            "AudioFrontendLayer": AudioFrontendLayer,
            "MagnitudeScalingLayer": MagnitudeScalingLayer,
        },
    )
    teacher_model.trainable = False

    cfg_path = getattr(args, "model_config", "") or os.path.splitext(args.checkpoint_path)[0] + "_model_config.json"
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Model config not found: {cfg_path}")
    cfg = ModelConfig.load(cfg_path)

    classes = load_classes_file(args.classes_file) if args.classes_file else list(cfg.class_names)
    if not classes:
        raise ValueError("QAT requires class_names in the model config or --classes_file")
    if classes != cfg.class_names:
        raise ValueError("QAT class order must exactly match the pretrained model config")
    if len(classes) != deployment_model.output_shape[-1]:
        raise ValueError("QAT dataset class count does not match the pretrained model output")

    train_paths, _ = load_file_paths_from_directory(args.data_path_train, classes=classes)
    if args.data_path_val:
        val_paths, _ = load_file_paths_from_directory(args.data_path_val, classes=classes)
    else:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(train_paths)
        split_idx = int(len(train_paths) * (1 - args.val_split))
        train_paths, val_paths = train_paths[:split_idx], train_paths[split_idx:]
    if not train_paths or not val_paths:
        raise ValueError("QAT requires non-empty training and validation datasets")

    if args.upsample_ratio and 0 < args.upsample_ratio <= 1.0:
        train_paths = upsample_minority_classes(train_paths, classes, args.upsample_ratio)

    common_kwargs = dict(
        sample_rate=cfg.sample_rate,
        max_duration=args.max_duration,
        chunk_duration=cfg.chunk_duration,
        spec_width=cfg.spec_width,
        mel_bins=cfg.num_mels,
        fft_length=cfg.fft_length,
        mag_scale=cfg.mag_scale,
        num_workers=args.num_workers,
        max_chunks_per_file=args.max_chunks_per_file,
        prefetch_batches=args.prefetch_batches,
    )
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
        snr_threshold=0.0,
        spec_augment=False,
        **common_kwargs,
    )

    n_frozen = freeze_batch_norm(deployment_model)
    print(f"[QAT] Frozen {n_frozen} BatchNorm layers")
    activation_ranges = calibrate_activation_ranges(deployment_model, val_dataset, max_samples=256)
    qat_student = build_qat_model(deployment_model, activation_ranges)
    qat_model = _DistilledQATModel(qat_student, teacher_model)
    print("[QAT] Enabled frozen-teacher Bernoulli KL consistency loss (weight=1.0)")

    qat_path = args.checkpoint_path.replace(".keras", "_qat.keras")
    ranges_path = qat_path.replace(".keras", "_activation_ranges.json")
    with open(ranges_path, "w", encoding="utf-8") as handle:
        json.dump({name: {"min": lo, "max": hi} for name, (lo, hi) in activation_ranges.items()}, handle, indent=2)

    steps_per_epoch = max(1, math.ceil(len(train_paths) / float(args.batch_size)))
    val_steps = max(1, math.ceil(len(val_paths) / float(args.batch_size)))
    print(f"[QAT] Training on {len(train_paths)} files, validating on {len(val_paths)} files")
    print(f"[QAT] Fine-tuning for {args.epochs} epochs at LR={args.learning_rate}")
    train_model(
        qat_model,
        train_dataset,
        val_dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        checkpoint_path=qat_path,
        steps_per_epoch=steps_per_epoch,
        val_steps=val_steps,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        loss_fn=_detect_loss(deployment_model),
        gradient_clip_norm=args.grad_clip,
        checkpoint_model=deployment_model,
        checkpoint_sync=lambda: sync_frontend_weights(qat_model, deployment_model),
    )
    print(f"[QAT] Clean quantization-ready checkpoint saved to {qat_path}")
    print(f"[QAT] Activation calibration ranges saved to {ranges_path}")
