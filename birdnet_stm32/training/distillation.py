"""Knowledge distillation loss for training with soft teacher labels.

Implements a combined loss that blends the standard hard-label loss with
a KL-divergence distillation loss from a teacher model's soft predictions.
"""

import tensorflow as tf


class DistillationLoss(tf.keras.losses.Loss):
    """Combined hard-label + soft-label distillation loss.

    Loss = (1 - alpha) * student_loss(y_true, y_pred) +
           alpha * T^2 * KL(softmax(teacher_logits/T) || softmax(student_logits/T))

    For simplicity, this implementation accepts pre-computed soft labels
    (teacher probabilities) rather than teacher logits, and uses
    categorical crossentropy as the distillation term.

    Args:
        alpha: Weight for the distillation loss (0 = pure hard labels, 1 = pure distillation).
        temperature: Softmax temperature for smoothing teacher predictions.
        student_loss: Base loss function for hard labels.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        temperature: float = 3.0,
        student_loss: tf.keras.losses.Loss | None = None,
        name: str = "distillation_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.temperature = temperature
        self.student_loss = student_loss or tf.keras.losses.CategoricalCrossentropy(
            from_logits=False
        )

    def call(self, y_true, y_pred):
        """Compute combined distillation loss.

        Args:
            y_true: Ground truth labels. Should be a concatenation of
                [hard_labels, soft_labels] along the last axis, where
                hard_labels has shape [B, C] and soft_labels has shape [B, C].
                Total shape: [B, 2*C].
            y_pred: Student model predictions [B, C].

        Returns:
            Scalar loss value.
        """
        num_classes = tf.shape(y_pred)[-1]
        hard_labels = y_true[:, :num_classes]
        soft_labels = y_true[:, num_classes:]

        # Hard label loss
        hard_loss = self.student_loss(hard_labels, y_pred)

        # Distillation loss: KL between temperature-smoothed distributions
        T = self.temperature
        soft_targets = tf.nn.softmax(tf.math.log(soft_labels + 1e-7) / T, axis=-1)
        soft_pred = tf.nn.softmax(tf.math.log(y_pred + 1e-7) / T, axis=-1)
        distill_loss = tf.keras.losses.KLDivergence()(soft_targets, soft_pred) * (T * T)

        return (1.0 - self.alpha) * hard_loss + self.alpha * distill_loss

    def get_config(self):
        """Return serializable config."""
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "temperature": self.temperature,
        })
        return config
