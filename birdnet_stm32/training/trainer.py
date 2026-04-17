"""Training loop with cosine LR schedule, early stopping, and checkpointing."""

import os

import tensorflow as tf

VALID_OPTIMIZERS = ("adam", "sgd", "adamw")


def _build_optimizer(
    name: str,
    learning_rate: tf.keras.optimizers.schedules.LearningRateSchedule,
    weight_decay: float = 0.0,
) -> tf.keras.optimizers.Optimizer:
    """Build a Keras optimizer by name.

    Args:
        name: Optimizer name ('adam', 'sgd', or 'adamw').
        learning_rate: Learning rate schedule.
        weight_decay: Weight decay factor (only used by adamw).

    Returns:
        Configured Keras optimizer.

    Raises:
        ValueError: If name is not a valid optimizer.
    """
    name = name.lower()
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    if name == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Invalid optimizer: '{name}'. Valid options: {VALID_OPTIMIZERS}")


def train_model(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    epochs: int = 50,
    learning_rate: float = 0.001,
    batch_size: int = 64,
    patience: int = 10,
    checkpoint_path: str = "checkpoints/best_model.keras",
    steps_per_epoch: int | None = None,
    val_steps: int | None = None,
    is_multilabel: bool = False,
    optimizer: str = "adam",
    weight_decay: float = 0.0,
    loss_fn: str | tf.keras.losses.Loss | None = None,
) -> tf.keras.callbacks.History:
    """Train a model with cosine LR schedule, early stopping, and checkpointing.

    Monitors val_loss (min). Best model is saved as a full .keras file.

    Args:
        model: Model to train.
        train_dataset: Training dataset (infinite).
        val_dataset: Validation dataset (infinite).
        epochs: Number of epochs.
        learning_rate: Initial learning rate for cosine schedule.
        batch_size: Unused; kept for API symmetry with data loader.
        patience: Early stopping patience (epochs).
        checkpoint_path: Path to save the best .keras model.
        steps_per_epoch: Training steps per epoch (> 0 required).
        val_steps: Validation steps per epoch (defaults to 1 if <= 0).
        is_multilabel: If True, uses binary_crossentropy; else categorical_crossentropy.
        optimizer: Optimizer name ('adam', 'sgd', or 'adamw').
        weight_decay: Weight decay factor (only used by 'adamw').
        loss_fn: Optional custom loss function. Overrides is_multilabel default.

    Returns:
        Keras training history.
    """
    if steps_per_epoch is None or steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be > 0")
    if val_steps is None or val_steps <= 0:
        val_steps = 1

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=epochs * steps_per_epoch,
        alpha=0.0,
    )

    opt = _build_optimizer(optimizer, lr_schedule, weight_decay)

    if loss_fn is None:
        loss_fn = "binary_crossentropy" if is_multilabel else "categorical_crossentropy"

    model.compile(
        optimizer=opt,
        loss=loss_fn,
        metrics=[tf.keras.metrics.AUC(curve="ROC", multi_label=True, name="roc_auc")],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, mode="min"),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_loss", save_best_only=True, mode="min", save_weights_only=False
        ),
    ]
    return model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks,
    )


def compute_hop_length(sample_rate: int, chunk_duration: int, spec_width: int) -> int:
    """Compute hop length to produce spec_width frames from an input chunk.

    Args:
        sample_rate: Sampling rate (Hz).
        chunk_duration: Chunk duration (seconds).
        spec_width: Desired number of frames.

    Returns:
        Hop length in samples (floor(T / spec_width), at least 1).
    """
    T = int(sample_rate * chunk_duration)
    return max(1, T // int(spec_width))
