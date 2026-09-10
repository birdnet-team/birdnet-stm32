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

from birdnet_stm32.training.distillation import DistilledModel, all_layers, validate_loss_weights

QUANTIZABLE_TYPES = (layers.Conv2D, layers.DepthwiseConv2D, layers.Dense)
ACTIVATION_BOUNDARY_TYPES = (
    layers.BatchNormalization,
    layers.ReLU,
    layers.Add,
    layers.Multiply,
    layers.Dense,
    layers.GlobalAveragePooling2D,
)
INFERENCE_PASSTHROUGH_TYPES = (
    layers.Dropout,
    layers.SpatialDropout1D,
    layers.SpatialDropout2D,
    layers.SpatialDropout3D,
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
    return np.asarray(np.clip(np.round(w / scale), -qmax, qmax) * scale, dtype=np.float32)


def _channel_axis(layer: tf.keras.layers.Layer) -> int:
    """Return TFLite's output-channel axis for a kernel."""
    return -2 if isinstance(layer, layers.DepthwiseConv2D) else -1


def _is_quantizable(layer: tf.keras.layers.Layer) -> bool:
    """Return whether a layer owns a kernel quantized by full-INT8 TFLite."""
    return isinstance(layer, QUANTIZABLE_TYPES) and hasattr(layer, "kernel")


def _is_activation_boundary(layer: tf.keras.layers.Layer) -> bool:
    """Return whether an outer-graph tensor is requantized during inference."""
    if isinstance(layer, layers.BatchNormalization):
        # Conv + BN + ReLU is folded into one quantized TFLite operator. Keep a
        # boundary only for linear project BNs whose output feeds an Add.
        # Dropout layers disappear during inference, so follow through them
        # before deciding whether the effective consumer is a fused ReLU.
        consumers = [node.operation for node in layer._outbound_nodes]  # noqa: SLF001
        while consumers and all(isinstance(consumer, INFERENCE_PASSTHROUGH_TYPES) for consumer in consumers):
            consumers = [
                node.operation
                for consumer in consumers
                for node in consumer._outbound_nodes  # noqa: SLF001
            ]
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

    def call(self, inputs, linear=False):
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
        return output if linear else self.target.activation(output)


# Sample values for percentile bounds. Bounds must also exist in the saved
# deployment frontend; fake quantization alone never survives clean export.
_RESERVOIR_PER_SAMPLE = 4096


def _reservoir_add(store: dict[str, list], name: str, array: np.ndarray, rng: np.random.Generator) -> None:
    """Keep a bounded random subsample of one tensor's values."""
    flat = np.asarray(array).reshape(-1)
    if flat.size > _RESERVOIR_PER_SAMPLE:
        flat = flat[rng.integers(0, flat.size, _RESERVOIR_PER_SAMPLE)]
    store.setdefault(name, []).append(flat.astype(np.float32, copy=False))


def _reservoir_range(store: dict[str, list], name: str, percentile: float) -> tuple[float, float] | None:
    """Return the percentile range for one tensor, or None if unseen."""
    chunks = store.get(name)
    if not chunks:
        return None
    pooled = np.concatenate(chunks)
    lo = float(np.percentile(pooled, 100.0 - percentile))
    hi = float(np.percentile(pooled, percentile))
    return (min(lo, 0.0), max(hi, 0.0))


class _ActivationRangeCollector:
    """Observe internal custom-layer tensors without changing their values."""

    def __init__(self, reservoir: dict[str, list] | None = None, rng: np.random.Generator | None = None):
        self.ranges: dict[str, list[float]] = {}
        self.reservoir = reservoir
        self.rng = rng

    def activation(self, name: str, inputs):
        """Record one tensor's scalar range and return it unchanged."""
        array = np.asarray(inputs)
        values = self.ranges.setdefault(name, [float("inf"), -float("inf")])
        values[0] = min(values[0], float(np.min(array)), 0.0)
        values[1] = max(values[1], float(np.max(array)), 0.0)
        if self.reservoir is not None and self.rng is not None:
            _reservoir_add(self.reservoir, name, array, self.rng)
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


# The alias keeps the QAT-specific name used across this module and its tests.
_DistilledQATModel = DistilledModel


def calibrate_activation_ranges(
    model: tf.keras.Model,
    dataset: Iterable,
    max_samples: int = 64,
    percentile: float = 100.0,
) -> dict[str, tuple[float, float]]:
    """Measure scalar activation ranges on real inputs for QAT initialization.

    Args:
        model: Deployment model to probe.
        dataset: Calibration data.
        max_samples: Number of samples to observe.
        percentile: Upper percentile defining each range; 100 is absolute
            min/max, the previous behaviour. Below 100 the range is taken from a
            bounded reservoir of observed values, clipping outliers that would
            otherwise stretch the INT8 grid.
    """
    if not 50.0 < percentile <= 100.0:
        raise ValueError("Calibration percentile must be in (50, 100]")
    if max_samples <= 0:
        raise ValueError("Calibration sample count must be positive")
    boundaries = [layer for layer in model.layers if _is_activation_boundary(layer)]
    if not boundaries:
        raise ValueError("Model has no supported activation quantization boundaries")
    sigmoid_heads = [layer for layer in boundaries if _is_sigmoid_dense(layer)]
    probe = tf.keras.Model(
        model.inputs, [layer.output for layer in boundaries] + [layer.input for layer in sigmoid_heads]
    )
    ranges = {layer.name: [float("inf"), -float("inf")] for layer in boundaries}
    ranges["__input__"] = [float("inf"), -float("inf")]
    frontends = [layer for layer in all_layers(model) if layer.__class__.__name__ == "AudioFrontendLayer"]
    use_percentile = percentile < 100.0
    reservoir: dict[str, list] = {}
    rng = np.random.default_rng(0)
    collector = _ActivationRangeCollector(reservoir if use_percentile else None, rng)
    for frontend in frontends:
        frontend.set_quantization_hook(collector)

    seen = 0
    try:
        for batch in dataset:
            inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
            inputs = np.asarray(inputs)
            for sample in inputs:
                ranges["__input__"][0] = min(ranges["__input__"][0], float(np.min(sample)), 0.0)
                ranges["__input__"][1] = max(ranges["__input__"][1], float(np.max(sample)), 0.0)
                outputs = probe([sample[None]], training=False)
                if not isinstance(outputs, (tuple, list)):
                    outputs = [outputs]
                for layer, output in zip(boundaries, outputs[: len(boundaries)], strict=True):
                    array = np.asarray(output)
                    ranges[layer.name][0] = min(ranges[layer.name][0], float(np.min(array)), 0.0)
                    ranges[layer.name][1] = max(ranges[layer.name][1], float(np.max(array)), 0.0)
                for layer, features in zip(sigmoid_heads, outputs[len(boundaries) :], strict=True):
                    logits = np.asarray(features) @ np.asarray(layer.kernel)
                    if layer.bias is not None:
                        logits += np.asarray(layer.bias)
                    collector.activation(f"{layer.name}__logits", logits)
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
    if use_percentile:
        clipped = 0
        internal_names = set(collector.ranges) - {f"{layer.name}__logits" for layer in sigmoid_heads}
        for name in internal_names:
            narrowed = _reservoir_range(reservoir, name, percentile)
            if narrowed is None:
                continue
            lo, hi = narrowed
            wide_lo, wide_hi = result[name]
            # Never widen a range: the percentile is a clip, not a re-estimate.
            lo, hi = max(lo, wide_lo), min(hi, wide_hi)
            if hi <= lo:
                continue
            if (hi - lo) < (wide_hi - wide_lo):
                clipped += 1
            result[name] = (lo, hi)
        print(
            f"[QAT] Calibrated {len(result)} activation tensors on {seen} samples "
            f"at the {percentile:g}th percentile ({clipped} ranges narrowed)"
        )
    else:
        print(f"[QAT] Calibrated {len(result)} activation tensors on {seen} samples")
    for layer in sigmoid_heads:
        # TFLite LOGISTIC has fixed scale 1/256 and zero point -128.
        result[layer.name] = (0.0, 255.0 / 256.0)
    return result


def _is_sigmoid_dense(layer):
    return isinstance(layer, layers.Dense) and layer.activation == tf.keras.activations.sigmoid


def bound_frontend(model, activation_ranges):
    """Clone custom frontends with persistent internal activation bounds."""

    def clone(layer):
        if layer.__class__.__name__ != "AudioFrontendLayer":
            return layer.__class__.from_config(layer.get_config())
        config = layer.get_config()
        config["activation_bounds"] = {
            name: values
            for name, values in activation_ranges.items()
            if name.startswith(f"{layer.name}_") and values[1] > values[0]
        }
        bounded = layer.__class__.from_config(config)
        bounded.build(tuple(layer.input.shape))
        bounded.set_weights(layer.get_weights())
        freeze_batch_norm(bounded)
        return bounded

    bounded_model = tf.keras.models.clone_model(model, clone_function=clone)
    bounded_model.set_weights(model.get_weights())
    return bounded_model


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
        freeze_batch_norm(clone)
        clone.set_quantization_hook(_FrontendQuantizationHook(activation_ranges))
        return clone

    def call_function(layer, *args, **kwargs):
        if _is_quantizable(layer):
            kernel_call = _QuantizedKernelCall(layer, name=f"{layer.name}_quantized_kernel")
            if _is_sigmoid_dense(layer):
                output = kernel_call(*args, linear=True, **kwargs)
                lo, hi = activation_ranges[f"{layer.name}__logits"]
                output = FakeQuantActivation(lo, hi, name=f"{layer.name}_logits_fake_quant")(output)
                output = layers.Activation("sigmoid")(output)
            else:
                output = kernel_call(*args, **kwargs)
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
        for layer in all_layers(qat_model)
        if layer.__class__.__name__ == "AudioFrontendLayer" and getattr(layer, "_quantization_hook", None) is not None
    }
    deployment_frontends = {
        layer.name: layer for layer in all_layers(deployment_model) if layer.__class__.__name__ == "AudioFrontendLayer"
    }
    if qat_frontends.keys() != deployment_frontends.keys():
        raise ValueError("QAT and deployment frontend layers do not match")
    for name, frontend in qat_frontends.items():
        deployment_frontends[name].set_weights(frontend.get_weights())


def freeze_batch_norm(model: tf.keras.Model) -> int:
    """Freeze all BatchNormalization layers, including nested frontend BN."""
    batch_norms = [layer for layer in all_layers(model) if isinstance(layer, layers.BatchNormalization)]
    for layer in batch_norms:
        layer.trainable = False
    return len(batch_norms)


def _detect_loss(model: tf.keras.Model) -> str:
    """Return the multi-label classifier loss."""
    del model
    return "binary_crossentropy"


def run_qat(args: argparse.Namespace) -> None:
    """Fine-tune a pretrained model against weight and activation INT8 noise."""
    from birdnet_stm32.conversion.quantize import representative_data_gen, stratified_sample_paths
    from birdnet_stm32.data.dataset import (
        load_classes_file,
        load_file_paths_from_directory,
        upsample_minority_classes,
    )
    from birdnet_stm32.data.generator import load_dataset
    from birdnet_stm32.models.runners import load_keras_model
    from birdnet_stm32.training.config import ModelConfig
    from birdnet_stm32.training.trainer import train_model
    from birdnet_stm32.training.validation import VALIDATION_SUBSET_SEED, Int8Selection, stratified_validation_subset

    if not args.checkpoint_path.endswith(".keras"):
        raise ValueError("QAT checkpoint must end in .keras")
    qat_path = args.checkpoint_path.replace(".keras", "_qat.keras")
    if os.path.exists(qat_path) or os.path.exists(qat_path.replace(".keras", "_selection.json")):
        raise FileExistsError("QAT outputs already exist; use a new run directory")
    if args.resume:
        raise ValueError("QAT resume is unsupported: start from an explicit checkpoint in a new output directory")
    if args.mixed_precision:
        raise ValueError("QAT requires float32 training for INT8 grid simulation")
    if not args.data_path_val:
        raise ValueError("QAT requires an explicit, disjoint --data_path_val")
    if not os.path.isfile(args.checkpoint_path):
        raise FileNotFoundError(f"QAT requires a pretrained model: {args.checkpoint_path}")
    print(f"[QAT] Loading pretrained model from {args.checkpoint_path}")
    deployment_model = load_keras_model(args.checkpoint_path)
    teacher_model = load_keras_model(args.checkpoint_path)
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

    train_paths, train_classes = load_file_paths_from_directory(args.data_path_train, classes=classes)
    if args.data_path_val:
        val_paths, val_classes = load_file_paths_from_directory(args.data_path_val, classes=classes)
    else:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(train_paths)
        split_idx = int(len(train_paths) * (1 - args.val_split))
        train_paths, val_paths = train_paths[:split_idx], train_paths[split_idx:]
    if not train_paths or not val_paths:
        raise ValueError("QAT requires non-empty training and validation datasets")
    if train_classes != classes:
        raise ValueError(
            f"QAT training data is missing configured classes: {[name for name in classes if name not in train_classes]}"
        )
    if args.data_path_val and val_classes != classes:
        raise ValueError(
            f"QAT validation data is missing configured classes: {[name for name in classes if name not in val_classes]}"
        )

    if set(map(os.path.realpath, train_paths)) & set(map(os.path.realpath, val_paths)):
        raise ValueError("QAT training and validation manifests overlap")

    # Conversion samples the physical training manifest, never the optionally
    # duplicated epoch-balancing list used by the trainer.
    calibration_source_paths = list(train_paths)
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

    calibration_count = int(args.qat_calibration_samples)
    calibration_paths = stratified_sample_paths(calibration_source_paths, calibration_count, seed=42)
    if len(calibration_paths) != calibration_count:
        raise ValueError(
            f"QAT requested {calibration_count} calibration paths but only {len(calibration_paths)} are available"
        )
    calibration_data = list(
        representative_data_gen(
            calibration_paths,
            cfg.to_dict(),
            num_samples=calibration_count,
        )
    )
    if len(calibration_data) != calibration_count:
        raise ValueError("Calibration skipped files; refusing to train on an incomplete manifest")
    activation_ranges = calibrate_activation_ranges(
        deployment_model,
        calibration_data,
        max_samples=calibration_count,
        percentile=float(getattr(args, "qat_calibration_percentile", 100.0)),
    )
    if args.qat_calibration_percentile < 100.0:
        deployment_model = bound_frontend(deployment_model, activation_ranges)
        # Re-observe the bounded graph: these are the values conversion sees.
        activation_ranges = calibrate_activation_ranges(
            deployment_model,
            calibration_data,
            max_samples=calibration_count,
        )
    print(f"[QAT] Activation ranges use the converter's exact stratified {calibration_count}-sample manifest (seed=42)")
    extra_callbacks: list[tf.keras.callbacks.Callback] = []

    loss_weights = {
        "distillation_weight": float(args.qat_distillation_weight),
        "cosine_weight": float(args.qat_cosine_weight),
        "cosine_tail_weight": float(args.qat_cosine_tail_weight),
        "cosine_tail_fraction": float(args.qat_cosine_tail_fraction),
    }
    validate_loss_weights(loss_weights)
    qat_student = build_qat_model(deployment_model, activation_ranges)
    qat_model = _DistilledQATModel(qat_student, teacher_model, **loss_weights)
    print(
        "[QAT] Enabled frozen-teacher consistency losses "
        f"(Bernoulli KL weight={loss_weights['distillation_weight']:.3g}, "
        f"cosine mean weight={loss_weights['cosine_weight']:.3g}, "
        f"worst {loss_weights['cosine_tail_fraction']:.1%} cosine "
        f"weight={loss_weights['cosine_tail_weight']:.3g})"
    )

    qat_path = args.checkpoint_path.replace(".keras", "_qat.keras")
    ranges_path = qat_path.replace(".keras", "_activation_ranges.json")
    with open(ranges_path, "w", encoding="utf-8") as handle:
        json.dump({name: {"min": lo, "max": hi} for name, (lo, hi) in activation_ranges.items()}, handle, indent=2)

    cfg.save(qat_path.replace(".keras", "_model_config.json"))
    with open(qat_path.replace(".keras", "_labels.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(classes) + "\n")
    # Selection converts and scores an INT8 model every epoch, twice over the
    # manifest. A fixed stratified subset keeps that affordable; it must be the
    # same draw for every arm and epoch or the comparison means nothing, so it
    # is seeded and its hash goes into the selection report.
    subset = int(getattr(args, "validation_subset", 0) or 0)
    selection_paths = stratified_validation_subset(val_paths, subset)
    if len(selection_paths) < len(val_paths):
        print(
            f"[QAT] Selecting on a fixed stratified subset of {len(selection_paths)} "
            f"of {len(val_paths)} validation files (seed {VALIDATION_SUBSET_SEED})"
        )
    elif subset:
        print(f"[QAT] Requested subset {subset} >= {len(val_paths)} validation files; using all")
    selector = Int8Selection(
        deployment_model,
        teacher_model,
        calibration_data,
        qat_path,
        sync=lambda: sync_frontend_weights(qat_model, deployment_model),
        files=selection_paths,
        classes=classes,
        cfg=cfg.to_dict(),
        overlap=args.validation_overlap,
        pooling=args.validation_pooling,
        batch_size=args.batch_size,
    )
    extra_callbacks.append(selector)
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
        checkpoint_monitor="val_int8_cmap",
        checkpoint_mode="max",
        checkpoint_managed=True,
        extra_callbacks=extra_callbacks,
    )
    print(f"[QAT] Best converted INT8 cMAP: {selector.best:.6f}; checkpoint: {qat_path}")
    print(f"[QAT] Selected development TFLite: {selector.int8_path}")
    print(f"[QAT] Activation calibration ranges saved to {ranges_path}")
