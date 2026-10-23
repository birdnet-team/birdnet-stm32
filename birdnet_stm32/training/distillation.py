"""Frozen-teacher losses: QAT consistency, and online distillation from a larger model.

QAT perturbs a model that already converged.  Hard labels alone do not tell the
perturbed student where the original decision surface was, so it optimizes the
supervised loss together with a Bernoulli KL divergence and a cosine agreement
term against a frozen copy of the pre-compression checkpoint.  The tail term
targets the worst samples in each batch, which is where a quantized detector
actually loses ranking quality.

Online distillation (``OnlineDistilledModel``) trains a smaller student from
scratch on its labels plus the same Bernoulli KL against a larger trained model,
run frozen on the very batch the student sees, augmentation and all, so no
offline pass over the training set is needed.
"""

import tensorflow as tf

# Defaults for the teacher-consistency objective.
DEFAULT_DISTILLATION_WEIGHT = 1.0
DEFAULT_COSINE_WEIGHT = 0.10
DEFAULT_COSINE_TAIL_WEIGHT = 0.75
DEFAULT_COSINE_TAIL_FRACTION = 0.10


def bernoulli_kl(teacher_pred: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Mean Bernoulli KL divergence of the student's per-class probabilities from the teacher's.

    Each output is its own Bernoulli variable (multi-label sigmoid outputs), so the
    divergence is the binary cross-entropy against the teacher minus the teacher's
    own entropy, averaged over classes and batch. Computed in float32 whatever the
    models' precision.

    Args:
        teacher_pred: ``[B, C]`` teacher probabilities.
        y_pred: ``[B, C]`` student probabilities.

    Returns:
        Scalar divergence (0 when the two agree).
    """
    epsilon = tf.keras.backend.epsilon()
    teacher_pred = tf.clip_by_value(tf.cast(teacher_pred, tf.float32), epsilon, 1.0 - epsilon)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), epsilon, 1.0 - epsilon)
    cross_entropy = tf.keras.losses.binary_crossentropy(teacher_pred, y_pred)
    teacher_entropy = tf.keras.losses.binary_crossentropy(teacher_pred, teacher_pred)
    return tf.reduce_mean(cross_entropy - teacher_entropy)


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
        divergence = bernoulli_kl(teacher_pred, y_pred)
        self.distillation_metric.update_state(divergence)
        epsilon = tf.cast(tf.keras.backend.epsilon(), y_pred.dtype)
        teacher_pred = tf.clip_by_value(teacher_pred, epsilon, 1.0 - epsilon)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
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


DEFAULT_ONLINE_DISTILLATION_WEIGHT = 1.0


class OnlineDistilledModel(tf.keras.Model):
    """Train a student on its labels plus the Bernoulli KL to a frozen larger model's outputs on the same batch.

    Called, it is the student: same inputs and outputs, so validation and
    checkpoint selection see the deployed graph. Checkpoint the ``student``
    (``train_model(training_wrapper=...)`` does), never this wrapper: the teacher
    exists for training only.

    Args:
        student: The model being trained. Its variables are shared.
        teacher: The trained model to distil from; frozen here. It must take the
            student's input and score the student's classes in the same order.
        weight: Weight of the KL divergence against the supervised loss.
    """

    def __init__(
        self, student: tf.keras.Model, teacher: tf.keras.Model, weight: float = DEFAULT_ONLINE_DISTILLATION_WEIGHT
    ):
        super().__init__(inputs=student.inputs, outputs=student.outputs, name=student.name)
        check_distillation_teacher(student, teacher)
        if weight <= 0:
            raise ValueError(f"online distillation weight must be positive, got {weight}")
        teacher.trainable = False
        self.teacher = teacher
        self.distill_weight = float(weight)
        self.distill_metric = tf.keras.metrics.Mean(name="distill_kl")

    def distillation_loss(self, x: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """The weighted KL term for one batch; also updates ``distill_kl``."""
        divergence = bernoulli_kl(tf.stop_gradient(self.teacher(x, training=False)), y_pred)
        self.distill_metric.update_state(divergence)
        return self.distill_weight * divergence

    def train_step(self, data):
        """The default Keras step, plus the distillation term."""
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x=x, y=y, y_pred=y_pred, training=True) + self.distillation_loss(x, y_pred)
            self._loss_tracker.update_state(loss, sample_weight=tf.shape(y)[0])
            scaled = self.optimizer.scale_loss(loss)
        variables = self.trainable_weights
        self.optimizer.apply_gradients(zip(tape.gradient(scaled, variables), variables, strict=True))
        return self.compute_metrics(x, y, y_pred)

    def test_step(self, data):
        """The default step; validation does not run the teacher."""
        logs = super().test_step(data)
        logs.pop(self.distill_metric.name, None)
        return logs


def check_distillation_teacher(student: tf.keras.Model, teacher: tf.keras.Model) -> None:
    """Raise unless *teacher* takes the student's input and has as many outputs.

    Raises:
        ValueError: On a different input shape or class count.
    """
    if tuple(teacher.input_shape[1:]) != tuple(student.input_shape[1:]):
        raise ValueError(f"teacher input {teacher.input_shape[1:]} != student input {student.input_shape[1:]}")
    if teacher.output_shape[-1] != student.output_shape[-1]:
        raise ValueError(f"teacher scores {teacher.output_shape[-1]} classes, the student {student.output_shape[-1]}")
