"""Loss functions for audio classification training.

Provides focal loss as an alternative to standard crossentropy for
imbalanced class distributions.
"""

import tensorflow as tf


class BinaryFocalLoss(tf.keras.losses.Loss):
    """Binary focal loss for multi-label classification.

    Focal loss down-weights well-classified examples, focusing training on
    hard negatives. Equivalent to binary crossentropy when gamma=0.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", 2017.

    Args:
        gamma: Focusing parameter (>= 0). Higher values focus more on hard examples.
        from_logits: Whether predictions are raw logits (True) or probabilities (False).
    """

    def __init__(self, gamma: float = 2.0, from_logits: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        """Compute focal loss.

        Args:
            y_true: Ground-truth labels [B, C].
            y_pred: Predicted probabilities or logits [B, C].

        Returns:
            Scalar loss.
        """
        y_true = tf.cast(y_true, y_pred.dtype)
        if self.from_logits:
            bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
            p_t = tf.sigmoid(y_pred)
        else:
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)

        focal_weight = (1.0 - p_t) ** self.gamma
        return tf.reduce_mean(focal_weight * bce)

    def get_config(self):
        """Return serializable config."""
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "from_logits": self.from_logits})
        return cfg
