"""Gradual magnitude pruning as a post-training compression step.

The step mirrors quantization-aware training: it starts from a converged
checkpoint, fine-tunes it under a growing perturbation, and writes a clean
deployment checkpoint that contains no training-only wrappers.  Here the
perturbation is a binary mask over the kernels of the pointwise convolutions,
grown along the cubic sparsity ramp of Zhu & Gupta (2017).

Design notes that matter for keeping accuracy:

* Masks are applied in the forward pass only.  The dense variable keeps its
  full-precision value, so a weight that was masked early can re-enter the
  network at the next mask update if its magnitude recovered.
* Only kernels large enough to be redundant are pruned.  Depthwise kernels,
  squeeze-and-excite and classifier ``Dense`` layers, the audio frontend, and
  any kernel below ``min_layer_params`` are exempt; they hold a negligible
  share of the parameters and a large share of the sensitivity.
* The frozen pre-pruning checkpoint supervises the student through the same
  teacher-consistency losses used by QAT.
* Checkpoint selection and early stopping only start once the ramp has
  finished, so the saved model is always at the requested sparsity.
* The step ends with a held-out macro ROC-AUC comparison against the teacher
  and fails if the drop exceeds ``--prune_max_auc_drop``.
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

# Kernels below this size are the stem, the squeeze-and-excite projections and
# similar bottlenecks: a rounding error in the parameter budget, but the first
# place accuracy breaks when they lose weights.
DEFAULT_MIN_LAYER_PARAMS = 1024
# A single layer stripped past this point stops passing a usable signal, so the
# global allocator never takes more from it even if its weights rank smallest.
MAX_LAYER_SPARSITY = 0.95
# Sparsity below this is indistinguishable from an unpruned kernel that happens
# to contain a few exact zeros, and must not be mistaken for a pruning mask.
SPARSITY_DETECTION_THRESHOLD = 0.01

# Dense is prunable only for the classifier head, never for the squeeze-and-
# excite gates: the head is the artifact shipped over a narrowband link, so its
# weights are the ones worth compressing.
PRUNABLE_TYPES = (layers.Conv2D, layers.Dense)
PRUNE_SCOPES = ("layerwise", "global")


def _kernel_index(layer: tf.keras.layers.Layer) -> int:
    """Return the position of ``layer.kernel`` within ``layer.get_weights()``."""
    for index, variable in enumerate(layer.weights):
        if variable is layer.kernel:
            return index
    raise ValueError(f"Layer {layer.name} has no kernel among its weights")


def _frontend_layers(model: tf.keras.Model) -> set[int]:
    """Return ``id()`` of every layer owned by an audio frontend.

    The frontend holds mel mixers and Gabor quadrature filterbanks whose
    kernels are structured signal-processing filters, not redundant capacity.
    """
    owned: set[int] = set()
    for layer in all_layers(model):
        if layer.__class__.__name__ != "AudioFrontendLayer":
            continue
        owned.add(id(layer))
        owned.update(id(nested) for nested in all_layers(layer))
    return owned


def classifier_head_layer(model: tf.keras.Model) -> tf.keras.layers.Layer | None:
    """Return the ``Dense`` layer that produces the model's output, if any.

    This is the layer that becomes the separately shipped classifier head at
    conversion time, which is why it is the one ``Dense`` worth pruning.
    """
    for tensor in model.outputs:
        history = getattr(tensor, "_keras_history", None)
        operation = getattr(history, "operation", None)
        if operation is None and history is not None:
            operation = history[0]
        if isinstance(operation, layers.Dense):
            return operation
    return None


def select_prunable_layers(
    model: tf.keras.Model,
    min_layer_params: int = DEFAULT_MIN_LAYER_PARAMS,
    include_head: bool = True,
) -> list[tf.keras.layers.Layer]:
    """Return the layers whose kernels are pruned, in graph order.

    Selects dense (non-depthwise) convolutions outside the audio frontend whose
    kernel holds at least ``min_layer_params`` weights.  In a DS-CNN these are
    the expand, project and embedding pointwise convolutions, which carry the
    overwhelming majority of the parameters.  The classifier head is included
    by default because conversion ships it as its own artifact, so its size is
    what an over-the-air model update actually costs.

    Args:
        model: Deployment model to inspect.
        min_layer_params: Smallest kernel size that is eligible for pruning.
        include_head: Whether the output ``Dense`` layer is prunable.

    Returns:
        List of prunable layers.
    """
    frontend = _frontend_layers(model)
    head = classifier_head_layer(model) if include_head else None
    selected = []
    for layer in all_layers(model):
        if id(layer) in frontend or isinstance(layer, layers.DepthwiseConv2D):
            continue
        # Squeeze-and-excite gates are Dense too, and are tiny gain modulators
        # rather than spare capacity; only the output head qualifies.
        if isinstance(layer, layers.Dense) and layer is not head:
            continue
        if not isinstance(layer, PRUNABLE_TYPES) or getattr(layer, "kernel", None) is None:
            continue
        if int(np.prod(layer.kernel.shape)) < int(min_layer_params):
            continue
        selected.append(layer)
    return selected


def polynomial_sparsity(
    step: int,
    begin_step: int,
    end_step: int,
    final_sparsity: float,
    initial_sparsity: float = 0.0,
    power: float = 3.0,
) -> float:
    """Return the cubic-ramp target sparsity for a training step.

    Follows Zhu & Gupta (2017): most of the sparsity is added early, while the
    remaining weights still have many steps to compensate, and the ramp
    flattens out as it approaches the target.

    Args:
        step: Current global training step.
        begin_step: Step at which pruning starts.
        end_step: Step at which ``final_sparsity`` is reached.
        final_sparsity: Sparsity held after ``end_step``.
        initial_sparsity: Sparsity applied at ``begin_step``.
        power: Ramp exponent; 3 is the published default.

    Returns:
        Target fraction of pruned weights in [0, 1].
    """
    if step <= begin_step:
        return float(initial_sparsity)
    if step >= end_step or end_step <= begin_step:
        return float(final_sparsity)
    progress = (step - begin_step) / float(end_step - begin_step)
    return float(final_sparsity + (initial_sparsity - final_sparsity) * (1.0 - progress) ** power)


def _mask_smallest(magnitudes: np.ndarray, num_pruned: int) -> np.ndarray:
    """Return a 0/1 mask that zeroes exactly ``num_pruned`` smallest entries."""
    mask = np.ones(magnitudes.shape, dtype=np.float32)
    num_pruned = int(np.clip(num_pruned, 0, magnitudes.size))
    if num_pruned == 0:
        return mask
    if num_pruned >= magnitudes.size:
        return np.zeros(magnitudes.shape, dtype=np.float32)
    flat = magnitudes.reshape(-1)
    pruned = np.argpartition(flat, num_pruned - 1)[:num_pruned]
    mask.reshape(-1)[pruned] = 0.0
    return mask


def compute_masks(
    kernels: dict[str, np.ndarray],
    sparsity: float,
    scope: str = "layerwise",
    max_layer_sparsity: float = MAX_LAYER_SPARSITY,
    overrides: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Compute magnitude masks reaching ``sparsity`` over the prunable kernels.

    Args:
        kernels: Mapping of layer name to its current kernel values.
        sparsity: Target fraction of pruned weights.
        scope: ``'layerwise'`` gives every layer the same sparsity;
            ``'global'`` ranks all prunable weights together so that redundant
            layers absorb more of the budget, capped per layer by
            ``max_layer_sparsity``.
        max_layer_sparsity: Per-layer ceiling used by the global scope.
        overrides: Optional per-layer sparsity targets. These layers are always
            pruned layerwise at their own rate and are excluded from a global
            ranking, so the classifier head can be compressed harder than the
            backbone it ships alongside.

    Returns:
        Mapping of layer name to a float32 0/1 mask of the kernel's shape.

    Raises:
        ValueError: If ``scope`` is unknown or a sparsity is outside [0, 1).
    """
    if scope not in PRUNE_SCOPES:
        raise ValueError(f"Invalid prune scope: '{scope}'. Valid options: {PRUNE_SCOPES}")
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("Target sparsity must be in [0, 1)")
    overrides = {name: value for name, value in (overrides or {}).items() if name in kernels}
    if any(not 0.0 <= value < 1.0 for value in overrides.values()):
        raise ValueError("Override sparsity must be in [0, 1)")
    if overrides:
        masks = {
            name: _mask_smallest(np.abs(kernels[name]), int(round(value * kernels[name].size)))
            for name, value in overrides.items()
        }
        remaining = {name: kernel for name, kernel in kernels.items() if name not in overrides}
        if remaining:
            masks.update(compute_masks(remaining, sparsity, scope, max_layer_sparsity))
        return masks
    if sparsity == 0.0:
        return {name: np.ones(kernel.shape, dtype=np.float32) for name, kernel in kernels.items()}

    if scope == "layerwise":
        return {
            name: _mask_smallest(np.abs(kernel), int(round(sparsity * kernel.size))) for name, kernel in kernels.items()
        }

    if not 0.0 <= max_layer_sparsity < 1.0:
        raise ValueError("Maximum layer sparsity must be in [0, 1)")

    # Global scope with a per-layer capacity constraint. Only the smallest
    # ``max_layer_sparsity`` weights from each layer are eligible for the
    # shared ranking. Ranking that eligible pool redistributes any budget a
    # capped layer cannot absorb while still pruning the globally smallest
    # feasible set of weights.
    total_params = sum(kernel.size for kernel in kernels.values())
    target_count = int(round(sparsity * total_params))
    masks = {name: np.ones(kernel.shape, dtype=np.float32) for name, kernel in kernels.items()}
    if target_count == 0:
        return masks

    eligible_magnitudes = []
    eligible_locations: list[tuple[str, np.ndarray]] = []
    for name, kernel in kernels.items():
        flat_magnitudes = np.abs(kernel).reshape(-1)
        capacity = min(kernel.size, int(math.floor(max_layer_sparsity * kernel.size)))
        if capacity == 0:
            continue
        if capacity == kernel.size:
            indices = np.arange(kernel.size)
        else:
            indices = np.argpartition(flat_magnitudes, capacity - 1)[:capacity]
        eligible_magnitudes.append(flat_magnitudes[indices])
        eligible_locations.append((name, indices))

    available = sum(values.size for values in eligible_magnitudes)
    if target_count > available:
        raise ValueError(
            f"Target global sparsity {sparsity:.1%} exceeds the per-layer ceiling "
            f"({available}/{total_params} weights can be pruned)"
        )

    pool = np.concatenate(eligible_magnitudes)
    selected = _mask_smallest(pool, target_count) == 0.0
    offset = 0
    for (name, indices), magnitudes in zip(eligible_locations, eligible_magnitudes, strict=True):
        local_selected = selected[offset : offset + magnitudes.size]
        masks[name].reshape(-1)[indices[local_selected]] = 0.0
        offset += magnitudes.size
    return masks


def mask_sparsity(masks: dict[str, np.ndarray]) -> float:
    """Return the fraction of zeros across all masks."""
    total = sum(mask.size for mask in masks.values())
    if total == 0:
        return 0.0
    zeros = sum(int(mask.size - np.count_nonzero(mask)) for mask in masks.values())
    return zeros / float(total)


class _MaskedKernelCall(layers.Layer):
    """Call a built Conv2D or Dense with a binary mask applied to its kernel.

    The mask is a non-trainable variable of this wrapper, so it never reaches
    the deployment checkpoint, and the target's dense kernel keeps its
    full-precision value for possible revival at the next mask update.
    """

    def __init__(self, target: tf.keras.layers.Layer, **kwargs):
        super().__init__(trainable=True, **kwargs)
        self.target = target
        self.mask = self.add_weight(
            shape=tuple(target.kernel.shape),
            initializer="ones",
            trainable=False,
            name="pruning_mask",
        )

    def call(self, inputs):
        """Run the target's kernel math with the masked kernel."""
        dense_kernel = self.target.kernel
        mask = tf.cast(self.mask, dense_kernel.dtype)
        # Forward with the mask, but use a straight-through gradient so a
        # temporarily masked weight can recover and re-enter at a later mask
        # update. The deployment export still receives only the masked values.
        kernel = dense_kernel + tf.stop_gradient(dense_kernel * mask - dense_kernel)
        if isinstance(self.target, layers.Dense):
            output = tf.linalg.matmul(inputs, kernel)
        else:
            output = self.target.convolution_op(inputs, kernel)
        if self.target.bias is not None:
            output = output + self.target.bias
        return self.target.activation(output)


def build_pruning_model(
    deployment_model: tf.keras.Model,
    prunable_layers: list[tf.keras.layers.Layer],
) -> tuple[tf.keras.Model, dict[str, _MaskedKernelCall]]:
    """Build a mask-applying graph that shares every variable with the model.

    Args:
        deployment_model: Converged model to prune.
        prunable_layers: Layers selected by :func:`select_prunable_layers`.

    Returns:
        Tuple of the masked training graph and a mapping from layer name to
        the wrapper owning that layer's mask variable.
    """
    targets = {id(layer) for layer in prunable_layers}
    wrappers: dict[str, _MaskedKernelCall] = {}

    def call_function(layer, *args, **kwargs):
        if id(layer) not in targets:
            return layer(*args, **kwargs)
        if layer.name not in wrappers:
            wrappers[layer.name] = _MaskedKernelCall(layer, name=f"{layer.name}_masked_kernel")
        return wrappers[layer.name](*args, **kwargs)

    pruned = tf.keras.models.clone_model(
        deployment_model,
        clone_function=lambda layer: layer,
        call_function=call_function,
    )
    missing = {layer.name for layer in prunable_layers} - wrappers.keys()
    if missing:
        raise ValueError(f"Prunable layers never reached during graph cloning: {sorted(missing)}")
    return pruned, wrappers


def apply_masks_to_export(
    source_model: tf.keras.Model,
    export_model: tf.keras.Model,
    masks: dict[str, np.ndarray],
) -> None:
    """Copy *source_model* into *export_model* with the masks baked into it.

    ``export_model`` is an independent load of the same checkpoint, so writing
    it leaves the training graph's dense weights untouched.

    Args:
        source_model: Model whose (dense) weights are being trained.
        export_model: Structurally identical model that is checkpointed.
        masks: Mapping of layer name to a 0/1 kernel mask.

    Raises:
        ValueError: If the two models do not enumerate identical layers.
    """
    source_layers = all_layers(source_model)
    export_layers = all_layers(export_model)
    if [layer.name for layer in source_layers] != [layer.name for layer in export_layers]:
        raise ValueError("Pruning export model does not match the trained model layer for layer")
    for source_layer, export_layer in zip(source_layers, export_layers, strict=True):
        weights = source_layer.get_weights()
        if not weights:
            continue
        mask = masks.get(source_layer.name)
        if mask is not None:
            index = _kernel_index(source_layer)
            weights[index] = weights[index] * mask
        export_layer.set_weights(weights)


class GradualPruningScheduler(tf.keras.callbacks.Callback):
    """Grow the pruning masks along a cubic ramp, then hold them fixed.

    Args:
        wrappers: Mapping of layer name to its mask-owning wrapper.
        begin_step: Global step at which pruning starts.
        end_step: Global step at which the final sparsity is reached.
        final_sparsity: Target fraction of pruned weights.
        frequency: Steps between mask recomputations during the ramp.
        scope: ``'layerwise'`` or ``'global'`` sparsity allocation.
        initial_sparsity: Sparsity applied at ``begin_step``.
        final_overrides: Optional per-layer final sparsity targets, ramped on
            the same schedule as the rest of the model.
    """

    def __init__(
        self,
        wrappers: dict[str, _MaskedKernelCall],
        begin_step: int,
        end_step: int,
        final_sparsity: float,
        frequency: int = 100,
        scope: str = "layerwise",
        initial_sparsity: float = 0.0,
        final_overrides: dict[str, float] | None = None,
    ):
        super().__init__()
        self.wrappers = wrappers
        self.final_overrides = dict(final_overrides or {})
        self.begin_step = max(0, int(begin_step))
        self.end_step = max(self.begin_step, int(end_step))
        self.final_sparsity = float(final_sparsity)
        self.frequency = max(1, int(frequency))
        self.scope = scope
        self.initial_sparsity = float(initial_sparsity)
        self.step = 0
        self.current_sparsity = 0.0
        self._frozen = False

    def masks(self) -> dict[str, np.ndarray]:
        """Return the current masks as numpy arrays."""
        return {name: wrapper.mask.numpy() for name, wrapper in self.wrappers.items()}

    def _scaled_overrides(self, sparsity: float) -> dict[str, float]:
        """Scale each override by how far the shared ramp has progressed."""
        if not self.final_overrides or self.final_sparsity <= 0:
            return dict(self.final_overrides)
        progress = sparsity / self.final_sparsity
        return {name: value * progress for name, value in self.final_overrides.items()}

    def update_masks(self, sparsity: float) -> None:
        """Recompute every mask from the current kernel magnitudes."""
        kernels = {name: wrapper.target.kernel.numpy() for name, wrapper in self.wrappers.items()}
        masks = compute_masks(kernels, sparsity, scope=self.scope, overrides=self._scaled_overrides(sparsity))
        for name, wrapper in self.wrappers.items():
            wrapper.mask.assign(masks[name])
        self.current_sparsity = mask_sparsity(masks)

    def on_train_batch_end(self, batch, logs=None):
        """Advance the ramp and refresh the masks on the update schedule."""
        self.step += 1
        if self._frozen:
            return
        reached_end = self.step >= self.end_step
        if not reached_end and (self.step < self.begin_step or self.step % self.frequency != 0):
            return
        self.update_masks(
            polynomial_sparsity(
                self.step,
                self.begin_step,
                self.end_step,
                self.final_sparsity,
                self.initial_sparsity,
            )
        )
        if reached_end:
            # Freeze the mask for the remaining epochs so the surviving weights
            # fine-tune against a stationary architecture.
            self._frozen = True
            print(f"\n[prune] Ramp complete at step {self.step}: masks frozen at {self.current_sparsity:.1%} sparsity")

    def on_epoch_end(self, epoch, logs=None):
        """Report the sparsity actually in force for this epoch."""
        state = "frozen" if self._frozen else "ramping"
        print(f"[prune] epoch={epoch + 1} sparsity={self.current_sparsity:.1%} ({state})")


def collect_sparsity_masks(
    model: tf.keras.Model,
    min_layer_params: int = 1,
    min_sparsity: float = SPARSITY_DETECTION_THRESHOLD,
) -> dict[str, np.ndarray]:
    """Recover pruning masks from the exact zeros already in a checkpoint.

    A later fine-tuning step such as QAT would otherwise refill the pruned
    weights with gradient updates and silently undo the pruning.

    Args:
        model: Loaded checkpoint to inspect.
        min_layer_params: Smallest kernel inspected. The default includes tiny
            classifier heads because pruning may have used a lowered size floor
            and downstream QAT does not otherwise know that original setting.
        min_sparsity: Smallest zero fraction treated as a deliberate mask.

    Returns:
        Mapping of layer name to a 0/1 mask, empty when the model is dense.
    """
    masks = {}
    for layer in select_prunable_layers(model, min_layer_params=min_layer_params):
        kernel = layer.kernel.numpy()
        mask = (kernel != 0.0).astype(np.float32)
        if 1.0 - float(mask.mean()) >= min_sparsity:
            masks[layer.name] = mask
    return masks


class SparsityMaskEnforcer(tf.keras.callbacks.Callback):
    """Hold a previously pruned model's zeros at zero during further training.

    Args:
        model: Model owning the kernels (shared with the training graph).
        masks: Mapping of layer name to a 0/1 kernel mask.
    """

    def __init__(self, model: tf.keras.Model, masks: dict[str, np.ndarray]):
        super().__init__()
        by_name = {layer.name: layer for layer in all_layers(model)}
        self._targets = [(by_name[name], tf.constant(mask)) for name, mask in masks.items()]

    def enforce(self) -> None:
        """Re-apply every mask to its kernel."""
        for layer, mask in self._targets:
            layer.kernel.assign(layer.kernel * tf.cast(mask, layer.kernel.dtype))

    def on_train_batch_end(self, batch, logs=None):
        self.enforce()

    def on_train_end(self, logs=None):
        # EarlyStopping restores the best weights after the last batch, which
        # reinstates whatever the optimizer had written into the pruned slots.
        self.enforce()


def sparsity_report(
    model: tf.keras.Model,
    masks: dict[str, np.ndarray],
) -> dict:
    """Summarize per-layer and whole-model sparsity of a pruned model.

    Args:
        model: Pruned deployment model.
        masks: Masks that were applied to it.

    Returns:
        Dict with total parameter counts and a per-layer breakdown.
    """
    layer_rows = []
    pruned_total = 0
    pruned_zeros = 0
    for layer in all_layers(model):
        if layer.name not in masks:
            continue
        kernel = layer.kernel.numpy()
        zeros = int(kernel.size - np.count_nonzero(kernel))
        pruned_total += kernel.size
        pruned_zeros += zeros
        layer_rows.append(
            {
                "layer": layer.name,
                "params": int(kernel.size),
                "zeros": zeros,
                "sparsity": zeros / float(kernel.size),
            }
        )
    model_params = int(sum(int(np.prod(weight.shape)) for weight in model.weights))
    model_zeros = int(sum(int(np.prod(w.shape)) - np.count_nonzero(w) for w in model.get_weights()))
    return {
        "prunable_params": int(pruned_total),
        "prunable_zeros": int(pruned_zeros),
        "prunable_sparsity": pruned_zeros / float(pruned_total) if pruned_total else 0.0,
        "model_params": model_params,
        "model_zeros": model_zeros,
        "model_sparsity": model_zeros / float(model_params) if model_params else 0.0,
        "layers": layer_rows,
    }


def _collect_eval_batches(dataset: Iterable, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Materialize up to ``max_samples`` (inputs, labels) pairs from a dataset."""
    inputs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    seen = 0
    for batch in dataset:
        x, y = batch[0], batch[1]
        x = np.asarray(x)
        y = np.asarray(y)
        inputs.append(x)
        labels.append(y)
        seen += len(x)
        if seen >= max_samples:
            break
    if not inputs:
        raise ValueError("Pruning accuracy gate found no validation samples")
    return np.concatenate(inputs)[:max_samples], np.concatenate(labels)[:max_samples]


def macro_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Return macro-averaged ROC-AUC over classes that have both label values.

    Args:
        labels: Multi-hot label matrix [N, C].
        scores: Predicted probabilities [N, C].

    Returns:
        Mean per-class ROC-AUC, or ``nan`` when no class is scorable.
    """
    from sklearn.metrics import roc_auc_score

    per_class = []
    for index in range(labels.shape[1]):
        column = labels[:, index]
        positives = int(column.sum())
        if positives == 0 or positives == column.size:
            continue
        per_class.append(float(roc_auc_score(column, scores[:, index])))
    return float(np.mean(per_class)) if per_class else float("nan")


def evaluate_accuracy_gate(
    teacher_model: tf.keras.Model,
    pruned_model: tf.keras.Model,
    dataset: Iterable,
    max_samples: int,
    max_auc_drop: float,
    batch_size: int,
) -> dict:
    """Compare pruned and unpruned macro ROC-AUC on identical validation data.

    Args:
        teacher_model: Frozen pre-pruning checkpoint.
        pruned_model: Masked deployment model.
        dataset: Validation dataset to sample from.
        max_samples: Number of validation samples to score.
        max_auc_drop: Largest tolerated ROC-AUC regression.
        batch_size: Prediction batch size.

    Returns:
        Dict with both AUCs, their difference, the tolerance, and a pass flag.
    """
    inputs, labels = _collect_eval_batches(dataset, max_samples)
    baseline_scores = teacher_model.predict(inputs, batch_size=batch_size, verbose=0)
    pruned_scores = pruned_model.predict(inputs, batch_size=batch_size, verbose=0)
    baseline_auc = macro_roc_auc(labels, baseline_scores)
    pruned_auc = macro_roc_auc(labels, pruned_scores)
    drop = baseline_auc - pruned_auc
    return {
        "samples": int(len(inputs)),
        "baseline_macro_roc_auc": baseline_auc,
        "pruned_macro_roc_auc": pruned_auc,
        "roc_auc_drop": float(drop),
        "max_roc_auc_drop": float(max_auc_drop),
        "passed": bool(np.isfinite(drop) and drop <= max_auc_drop),
    }


def run_pruning(args: argparse.Namespace) -> None:
    """Prune a pretrained checkpoint with a gradual magnitude schedule."""
    from birdnet_stm32.data.dataset import (
        load_classes_file,
        load_file_paths_from_directory,
        upsample_minority_classes,
    )
    from birdnet_stm32.data.generator import load_dataset
    from birdnet_stm32.models.frontend import AudioFrontendLayer
    from birdnet_stm32.models.magnitude import MagnitudeScalingLayer
    from birdnet_stm32.training.config import ModelConfig
    from birdnet_stm32.training.qat import freeze_batch_norm
    from birdnet_stm32.training.trainer import train_model

    if not os.path.isfile(args.checkpoint_path):
        raise FileNotFoundError(f"Pruning requires a pretrained model: {args.checkpoint_path}")
    if not 0.0 < args.prune_final_sparsity < 1.0:
        raise ValueError("--prune_final_sparsity must be in (0, 1)")
    if not 0.0 < args.prune_ramp_fraction <= 1.0:
        raise ValueError("--prune_ramp_fraction must be in (0, 1]")
    if args.prune_scope not in PRUNE_SCOPES:
        raise ValueError(f"Invalid --prune_scope: '{args.prune_scope}'. Valid options: {PRUNE_SCOPES}")
    if args.prune_scope == "global" and args.prune_final_sparsity > MAX_LAYER_SPARSITY:
        raise ValueError(
            f"--prune_final_sparsity cannot exceed {MAX_LAYER_SPARSITY:.0%} in global mode "
            "because that is the per-layer sparsity ceiling"
        )
    if args.prune_frequency <= 0:
        raise ValueError("--prune_frequency must be positive")
    if args.prune_min_layer_params <= 0:
        raise ValueError("--prune_min_layer_params must be positive")
    if args.prune_eval_samples <= 0:
        raise ValueError("--prune_eval_samples must be positive")
    if args.prune_max_auc_drop < 0:
        raise ValueError("--prune_max_auc_drop must be non-negative")
    if args.prune_head_sparsity != -1.0 and not 0.0 <= args.prune_head_sparsity < 1.0:
        raise ValueError("--prune_head_sparsity must be -1 or in [0, 1)")

    custom_objects = {
        "AudioFrontendLayer": AudioFrontendLayer,
        "MagnitudeScalingLayer": MagnitudeScalingLayer,
    }
    print(f"[prune] Loading pretrained model from {args.checkpoint_path}")

    def _load() -> tf.keras.Model:
        return tf.keras.models.load_model(args.checkpoint_path, compile=False, custom_objects=custom_objects)

    deployment_model = _load()
    # A frozen teacher supervises the student, and a third independent copy
    # receives the masked weights so checkpointing never destroys the dense
    # values the next mask update needs.
    teacher_model = _load()
    teacher_model.trainable = False
    export_model = _load()

    cfg_path = getattr(args, "model_config", "") or os.path.splitext(args.checkpoint_path)[0] + "_model_config.json"
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Model config not found: {cfg_path}")
    cfg = ModelConfig.load(cfg_path)

    classes = load_classes_file(args.classes_file) if args.classes_file else list(cfg.class_names)
    if not classes:
        raise ValueError("Pruning requires class_names in the model config or --classes_file")
    if classes != cfg.class_names:
        raise ValueError("Pruning class order must exactly match the pretrained model config")
    if len(classes) != deployment_model.output_shape[-1]:
        raise ValueError("Pruning dataset class count does not match the pretrained model output")

    train_paths, _ = load_file_paths_from_directory(args.data_path_train, classes=classes)
    if args.data_path_val:
        val_paths, _ = load_file_paths_from_directory(args.data_path_val, classes=classes)
    else:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(train_paths)
        split_idx = int(len(train_paths) * (1 - args.val_split))
        train_paths, val_paths = train_paths[:split_idx], train_paths[split_idx:]
    if not train_paths or not val_paths:
        raise ValueError("Pruning requires non-empty training and validation datasets")
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
    # Pruning re-optimizes an already-converged model, so keep the light
    # augmentation that regularizes it and drop the label-mixing ones that
    # would blur the teacher/student comparison.
    train_dataset = load_dataset(
        train_paths,
        classes,
        audio_frontend=cfg.audio_frontend,
        batch_size=args.batch_size,
        mixup_alpha=0.0,
        mixup_probability=0.0,
        random_offset=True,
        snr_threshold=0.1,
        spec_augment=args.spec_augment,
        freq_mask_max=args.freq_mask_max,
        time_mask_max=args.time_mask_max,
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
    print(f"[prune] Frozen {n_frozen} BatchNorm layers")

    prunable_layers = select_prunable_layers(
        deployment_model,
        min_layer_params=args.prune_min_layer_params,
        include_head=args.prune_head,
    )
    if not prunable_layers:
        raise ValueError(
            "No prunable layers found. Lower --prune_min_layer_params "
            f"(currently {args.prune_min_layer_params}) or check the architecture."
        )
    prunable_params = int(sum(np.prod(layer.kernel.shape) for layer in prunable_layers))
    model_params = int(sum(int(np.prod(weight.shape)) for weight in deployment_model.weights))
    print(
        f"[prune] {len(prunable_layers)} prunable layers hold {prunable_params:,} of "
        f"{model_params:,} parameters ({prunable_params / max(model_params, 1):.1%})"
    )

    head_layer = classifier_head_layer(deployment_model) if args.prune_head else None
    head_overrides: dict[str, float] = {}
    if head_layer is not None and head_layer in prunable_layers and args.prune_head_sparsity >= 0:
        head_overrides[head_layer.name] = float(args.prune_head_sparsity)
    if head_layer is not None and head_layer in prunable_layers:
        target = head_overrides.get(head_layer.name, args.prune_final_sparsity)
        print(
            f"[prune] Classifier head '{head_layer.name}' "
            f"({int(np.prod(head_layer.kernel.shape)):,} weights) targets {target:.0%} sparsity"
        )
    elif args.prune_head:
        print("[prune] No eligible classifier head found; the head stays dense")

    loss_weights = {
        "distillation_weight": float(args.prune_distillation_weight),
        "cosine_weight": float(args.prune_cosine_weight),
        "cosine_tail_weight": float(args.prune_cosine_tail_weight),
        "cosine_tail_fraction": float(args.prune_cosine_tail_fraction),
    }
    validate_loss_weights(loss_weights)

    pruning_student, wrappers = build_pruning_model(deployment_model, prunable_layers)
    pruning_model = DistilledModel(pruning_student, teacher_model, **loss_weights)

    steps_per_epoch = max(1, math.ceil(len(train_paths) / float(args.batch_size)))
    val_steps = max(1, math.ceil(len(val_paths) / float(args.batch_size)))
    total_steps = steps_per_epoch * args.epochs
    end_step = max(1, int(round(total_steps * args.prune_ramp_fraction)))
    ramp_epochs = math.ceil(end_step / float(steps_per_epoch))
    scheduler = GradualPruningScheduler(
        wrappers,
        begin_step=0,
        end_step=end_step,
        final_sparsity=args.prune_final_sparsity,
        frequency=args.prune_frequency,
        scope=args.prune_scope,
        final_overrides=head_overrides,
    )
    print(f"[prune] Training on {len(train_paths)} files, validating on {len(val_paths)} files")
    print(
        f"[prune] Cubic ramp to {args.prune_final_sparsity:.0%} {args.prune_scope} sparsity over "
        f"{end_step} steps (~{ramp_epochs} of {args.epochs} epochs), then {args.epochs - ramp_epochs} "
        "epoch(s) of fixed-mask fine-tuning"
    )
    if ramp_epochs >= args.epochs:
        print(
            "[prune] WARNING: the ramp consumes the whole run, so no epoch is eligible for "
            "best-checkpoint selection and the final-epoch weights are saved. Raise --epochs "
            "or lower --prune_ramp_fraction to fine-tune the surviving weights."
        )
    print(
        "[prune] Frozen-teacher consistency losses "
        f"(Bernoulli KL weight={loss_weights['distillation_weight']:.3g}, "
        f"cosine mean weight={loss_weights['cosine_weight']:.3g}, "
        f"worst {loss_weights['cosine_tail_fraction']:.1%} cosine "
        f"weight={loss_weights['cosine_tail_weight']:.3g})"
    )

    pruned_path = args.checkpoint_path.replace(".keras", "_pruned.keras")
    train_model(
        pruning_model,
        train_dataset,
        val_dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        checkpoint_path=pruned_path,
        steps_per_epoch=steps_per_epoch,
        val_steps=val_steps,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.grad_clip,
        checkpoint_model=export_model,
        checkpoint_sync=lambda: apply_masks_to_export(deployment_model, export_model, scheduler.masks()),
        # Only models that already carry the requested sparsity may be selected;
        # a mid-ramp epoch always scores better and would win otherwise.
        checkpoint_start_epoch=ramp_epochs,
        extra_callbacks=[scheduler],
    )

    # train_model's on_train_end sync already wrote the final masked weights,
    # but reload so the report describes exactly the bytes that were saved.
    pruned_model = tf.keras.models.load_model(pruned_path, compile=False, custom_objects=custom_objects)
    report = sparsity_report(pruned_model, scheduler.masks())
    print(
        f"[prune] Pruned kernels: {report['prunable_zeros']:,} of {report['prunable_params']:,} weights zeroed "
        f"({report['prunable_sparsity']:.1%}); whole model {report['model_sparsity']:.1%}"
    )

    gate = evaluate_accuracy_gate(
        teacher_model,
        pruned_model,
        val_dataset,
        max_samples=args.prune_eval_samples,
        max_auc_drop=args.prune_max_auc_drop,
        batch_size=args.batch_size,
    )
    report.update(
        {
            "checkpoint": os.path.basename(pruned_path),
            "source_checkpoint": os.path.basename(args.checkpoint_path),
            "final_sparsity": float(args.prune_final_sparsity),
            "scope": args.prune_scope,
            "ramp_steps": int(end_step),
            "min_layer_params": int(args.prune_min_layer_params),
            "head_layer": head_layer.name if head_layer is not None and head_layer in prunable_layers else None,
            "head_sparsity_target": (
                head_overrides.get(head_layer.name, float(args.prune_final_sparsity))
                if head_layer is not None and head_layer in prunable_layers
                else None
            ),
            "accuracy_gate": gate,
        }
    )
    report_path = pruned_path.replace(".keras", "_pruning_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    out_cfg_path = os.path.splitext(pruned_path)[0] + "_model_config.json"
    cfg.save(out_cfg_path)

    print(
        f"[prune] Macro ROC-AUC on {gate['samples']} held-out samples: "
        f"baseline={gate['baseline_macro_roc_auc']:.4f} pruned={gate['pruned_macro_roc_auc']:.4f} "
        f"drop={gate['roc_auc_drop']:+.4f} (tolerance {gate['max_roc_auc_drop']:.4f})"
    )
    print(f"[prune] Pruned checkpoint saved to {pruned_path}")
    print(f"[prune] Sparsity report saved to {report_path}")
    print(f"[prune] Model config saved to {out_cfg_path}")

    if not gate["passed"]:
        raise RuntimeError(
            f"Pruning accuracy gate failed: macro ROC-AUC dropped by {gate['roc_auc_drop']:.4f}, "
            f"above the {gate['max_roc_auc_drop']:.4f} tolerance. The checkpoint was kept for inspection "
            f"at {pruned_path}. Lower --prune_final_sparsity, raise --epochs, or use "
            "--prune_scope global to let redundant layers absorb more of the budget."
        )
