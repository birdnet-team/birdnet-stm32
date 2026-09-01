"""Frozen-teacher consistency losses shared by the compression fine-tuning steps.

Both quantization-aware training and gradual magnitude pruning perturb a model
that already converged.  Hard labels alone do not tell the perturbed student
where the original decision surface was, so both steps optimize the supervised
loss together with a Bernoulli KL divergence and a cosine agreement term
against a frozen copy of the pre-compression checkpoint.  The tail term targets
the worst samples in each batch, which is where a compressed detector actually
loses ranking quality.
"""

import tensorflow as tf

# Shared defaults for the teacher-consistency objective.  QAT and pruning both
# perturb an already-converged model, so the same weighting applies to both.
DEFAULT_DISTILLATION_WEIGHT = 1.0
DEFAULT_COSINE_WEIGHT = 0.10
DEFAULT_COSINE_TAIL_WEIGHT = 0.75
DEFAULT_COSINE_TAIL_FRACTION = 0.10


def all_layers(model: tf.keras.Model) -> list[tf.keras.layers.Layer]:
    """Return every nested layer of *model* exactly once, in graph order."""
    flattened = model._flatten_layers(include_self=False, recursive=True)  # noqa: SLF001
    return list(dict.fromkeys(flattened))


class DistilledModel(tf.keras.Model):
    """Train a perturbed student against labels and a frozen float teacher.

    Args:
        student: Functional model holding the perturbed (quantized or masked)
            forward graph.  Its variables are shared with the deployment model.
        teacher: Frozen copy of the pre-compression checkpoint.
        distillation_weight: Weight of the per-output Bernoulli KL divergence.
        cosine_weight: Weight of the mean per-sample cosine distance.
        cosine_tail_weight: Weight of the worst-sample cosine distance.
        cosine_tail_fraction: Fraction of each batch entering the tail term.
    """

    def __init__(
        self,
        student: tf.keras.Model,
        teacher: tf.keras.Model,
        distillation_weight: float = DEFAULT_DISTILLATION_WEIGHT,
        cosine_weight: float = DEFAULT_COSINE_WEIGHT,
        cosine_tail_weight: float = DEFAULT_COSINE_TAIL_WEIGHT,
        cosine_tail_fraction: float = DEFAULT_COSINE_TAIL_FRACTION,
    ):
        super().__init__(inputs=student.inputs, outputs=student.outputs, name=student.name)
        teacher.trainable = False
        self.teacher = teacher
        self.distillation_weight = float(distillation_weight)
        self.cosine_weight = float(cosine_weight)
        self.cosine_tail_weight = float(cosine_tail_weight)
        self.cosine_tail_fraction = float(cosine_tail_fraction)
        self.distillation_metric = tf.keras.metrics.Mean(name="distillation_kl")
        self.cosine_metric = tf.keras.metrics.Mean(name="distillation_cosine_loss")
        self.cosine_tail_metric = tf.keras.metrics.Mean(name="distillation_cosine_tail_loss")

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
        teacher_direction = tf.math.l2_normalize(teacher_pred, axis=-1)
        student_direction = tf.math.l2_normalize(y_pred, axis=-1)
        per_sample_cosine_loss = 1.0 - tf.reduce_sum(teacher_direction * student_direction, axis=-1)
        cosine_loss = tf.reduce_mean(per_sample_cosine_loss)
        self.cosine_metric.update_state(cosine_loss)
        tail_count = tf.maximum(
            1,
            tf.cast(
                tf.math.ceil(tf.cast(tf.size(per_sample_cosine_loss), tf.float32) * self.cosine_tail_fraction),
                tf.int32,
            ),
        )
        cosine_tail_loss = tf.reduce_mean(tf.math.top_k(per_sample_cosine_loss, k=tail_count).values)
        self.cosine_tail_metric.update_state(cosine_tail_loss)
        return (
            supervised
            + self.distillation_weight * divergence
            + self.cosine_weight * cosine_loss
            + self.cosine_tail_weight * cosine_tail_loss
        )


def validate_loss_weights(weights: dict[str, float]) -> None:
    """Raise if a teacher-consistency weighting is outside its valid range.

    Args:
        weights: Mapping with the four ``DistilledModel`` loss keys.

    Raises:
        ValueError: If any weight is negative or the tail fraction is not in (0, 1].
    """
    if any(value < 0 for name, value in weights.items() if name != "cosine_tail_fraction"):
        raise ValueError("Teacher-consistency loss weights must be non-negative")
    if not 0 < weights["cosine_tail_fraction"] <= 1:
        raise ValueError("Cosine tail fraction must be in (0, 1]")
