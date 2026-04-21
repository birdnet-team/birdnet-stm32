"""Training loop with cosine LR schedule, early stopping, and checkpointing."""

import json
import os

import tensorflow as tf

VALID_OPTIMIZERS = ("adam", "sgd", "adamw")


def _build_optimizer(
    name: str,
    learning_rate: tf.keras.optimizers.schedules.LearningRateSchedule,
    weight_decay: float = 0.0,
    gradient_clip_norm: float = 0.0,
) -> tf.keras.optimizers.Optimizer:
    """Build a Keras optimizer by name.

    Args:
        name: Optimizer name ('adam', 'sgd', or 'adamw').
        learning_rate: Learning rate schedule.
        weight_decay: Weight decay factor (only used by adamw).
        gradient_clip_norm: Max gradient norm for clipping (0 = disabled).

    Returns:
        Configured Keras optimizer.

    Raises:
        ValueError: If name is not a valid optimizer.
    """
    name = name.lower()
    clip_kw = {"clipnorm": gradient_clip_norm} if gradient_clip_norm > 0 else {}
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate, **clip_kw)
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, **clip_kw)
    if name == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay, **clip_kw)
    raise ValueError(f"Invalid optimizer: '{name}'. Valid options: {VALID_OPTIMIZERS}")


def train_model(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    epochs: int = 50,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    patience: int = 10,
    checkpoint_path: str = "checkpoints/best_model.keras",
    steps_per_epoch: int | None = None,
    val_steps: int | None = None,
    is_multilabel: bool = False,
    optimizer: str = "adam",
    weight_decay: float = 0.0,
    loss_fn: str | tf.keras.losses.Loss | None = None,
    gradient_clip_norm: float = 1.0,
    class_weights: dict[int, float] | None = None,
    resume: bool = False,
    extra_callbacks: list[tf.keras.callbacks.Callback] | None = None,
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
        gradient_clip_norm: Max gradient norm for clipping (0 = disabled).
        class_weights: Optional class index → weight mapping for imbalanced data.
        resume: If True, load optimizer state from a previous run and continue.
        extra_callbacks: Additional Keras callbacks (e.g. QAT callback).

    Returns:
        Keras training history.
    """
    if steps_per_epoch is None or steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be > 0")
    if val_steps is None or val_steps <= 0:
        val_steps = 1

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    # Resume: reload model from checkpoint if it exists
    initial_epoch = 0
    state_path = checkpoint_path.replace(".keras", "_train_state.json")
    if resume and os.path.isfile(checkpoint_path):
        print(f"[resume] Loading model from {checkpoint_path}")
        from birdnet_stm32.models.frontend import AudioFrontendLayer
        from birdnet_stm32.models.magnitude import MagnitudeScalingLayer

        model = tf.keras.models.load_model(
            checkpoint_path,
            compile=False,
            custom_objects={
                "AudioFrontendLayer": AudioFrontendLayer,
                "MagnitudeScalingLayer": MagnitudeScalingLayer,
            },
        )
        if os.path.isfile(state_path):
            with open(state_path) as f:
                state = json.load(f)
            initial_epoch = state.get("epoch", 0)
            print(f"[resume] Resuming from epoch {initial_epoch}")

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=epochs * steps_per_epoch,
        alpha=0.0,
    )

    opt = _build_optimizer(optimizer, lr_schedule, weight_decay, gradient_clip_norm)

    if loss_fn is None:
        loss_fn = "binary_crossentropy" if is_multilabel else "categorical_crossentropy"

    model.compile(
        optimizer=opt,
        loss=loss_fn,
        metrics=[tf.keras.metrics.AUC(curve="ROC", multi_label=True, name="roc_auc")],
    )

    class _SaveTrainState(tf.keras.callbacks.Callback):
        """Save epoch counter alongside checkpoint for resume support."""

        def on_epoch_end(self, epoch, logs=None):
            with open(state_path, "w") as f:
                json.dump({"epoch": epoch + 1}, f)

    class _CSVHistoryLogger(tf.keras.callbacks.Callback):
        """Append per-epoch metrics to a CSV file alongside the checkpoint."""

        def __init__(self, csv_path):
            super().__init__()
            self.csv_path = csv_path
            self._header_written = os.path.isfile(csv_path) and resume

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            import csv

            write_header = not self._header_written
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["epoch"] + sorted(logs.keys()))
                if write_header:
                    writer.writeheader()
                    self._header_written = True
                row = {"epoch": epoch + 1}
                row.update({k: f"{v:.6f}" for k, v in logs.items()})
                writer.writerow(row)

    csv_path = checkpoint_path.replace(".keras", "_history.csv")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, mode="min"),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_loss", save_best_only=True, mode="min", save_weights_only=False
        ),
        _SaveTrainState(),
        _CSVHistoryLogger(csv_path),
    ]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    # Save training curves as PNG
    _save_training_curves(history, checkpoint_path.replace(".keras", "_curves.png"))

    return history


def _save_training_curves(history: tf.keras.callbacks.History, path: str) -> None:
    """Save loss and ROC-AUC training curves as a PNG image.

    Args:
        history: Keras training history.
        path: Output PNG path.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[info] matplotlib not installed; skipping training curves plot.")
        return

    hist = history.history
    epochs_range = range(1, len(hist.get("loss", [])) + 1)
    if not epochs_range:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax = axes[0]
    ax.plot(epochs_range, hist["loss"], label="train")
    if "val_loss" in hist:
        ax.plot(epochs_range, hist["val_loss"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ROC-AUC
    ax = axes[1]
    if "roc_auc" in hist:
        ax.plot(epochs_range, hist["roc_auc"], label="train")
    if "val_roc_auc" in hist:
        ax.plot(epochs_range, hist["val_roc_auc"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("ROC-AUC")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"Training curves saved to {path}")


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
