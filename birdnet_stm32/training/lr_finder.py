"""Learning rate finder — exponential sweep to find optimal LR range.

Implements the LR range test: increases LR exponentially from a small value
to a large value over a fixed number of steps, recording loss at each step.
The optimal LR is typically in the steepest descent region of the loss curve.
"""

import math

import numpy as np
import tensorflow as tf


class LRFinder(tf.keras.callbacks.Callback):
    """Exponential LR sweep callback.

    Increases the learning rate from ``min_lr`` to ``max_lr`` over
    ``num_steps`` mini-batches and records the loss at each step.

    Args:
        min_lr: Starting learning rate.
        max_lr: Ending learning rate.
        num_steps: Number of training steps in the sweep.
    """

    def __init__(self, min_lr: float = 1e-7, max_lr: float = 1.0, num_steps: int = 200):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_steps = num_steps
        self.lrs: list[float] = []
        self.losses: list[float] = []
        self._step = 0
        self._best_loss = float("inf")

    def on_train_batch_begin(self, batch, logs=None):
        """Set LR for the current step."""
        frac = self._step / max(self.num_steps - 1, 1)
        lr = self.min_lr * (self.max_lr / self.min_lr) ** frac
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)

    def on_train_batch_end(self, batch, logs=None):
        """Record LR and loss."""
        logs = logs or {}
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        loss = logs.get("loss", 0.0)
        self.lrs.append(lr)
        self.losses.append(float(loss))

        # Early stop if loss explodes (> 4× best)
        if loss < self._best_loss:
            self._best_loss = loss
        if loss > 4 * self._best_loss and self._step > 10:
            self.model.stop_training = True

        self._step += 1
        if self._step >= self.num_steps:
            self.model.stop_training = True

    def suggest_lr(self, smoothing: int = 5) -> float:
        """Find the LR with steepest loss descent (smoothed gradient).

        Args:
            smoothing: Window size for moving-average smoothing.

        Returns:
            Suggested learning rate.
        """
        if len(self.losses) < smoothing + 2:
            return self.lrs[len(self.lrs) // 2] if self.lrs else 1e-3

        # Smooth losses
        kernel = np.ones(smoothing) / smoothing
        smoothed = np.convolve(self.losses, kernel, mode="valid")
        log_lrs = np.log10(self.lrs[: len(smoothed)])

        # Steepest descent = most negative gradient
        grads = np.gradient(smoothed, log_lrs)
        idx = int(np.argmin(grads))
        return float(10 ** log_lrs[idx])

    def plot(self, path: str | None = None, suggested_lr: float | None = None) -> None:
        """Plot LR vs loss curve.

        Args:
            path: If given, saves to this file (PNG). Otherwise calls plt.show().
            suggested_lr: If given, draws a vertical line at the suggested LR.
        """
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[info] matplotlib not installed; skipping LR finder plot.")
            return

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.lrs, self.losses)
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Loss")
        ax.set_title("LR Range Test")
        ax.grid(True, alpha=0.3)

        if suggested_lr is not None:
            ax.axvline(x=suggested_lr, color="r", linestyle="--", label=f"suggested: {suggested_lr:.2e}")
            ax.legend()

        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=100)
            plt.close(fig)
            print(f"LR finder plot saved to {path}")
        else:
            plt.show()


def run_lr_finder(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    min_lr: float = 1e-7,
    max_lr: float = 1.0,
    num_steps: int = 200,
    loss_fn: str | tf.keras.losses.Loss = "categorical_crossentropy",
    plot_path: str | None = None,
) -> float:
    """Run an LR range test and return the suggested learning rate.

    The model weights are restored after the sweep — no permanent changes.

    Args:
        model: Compiled or uncompiled Keras model.
        dataset: Training dataset (infinite, batched).
        min_lr: Starting learning rate.
        max_lr: Ending learning rate.
        num_steps: Number of sweep steps.
        loss_fn: Loss function.
        plot_path: Optional path to save the LR-vs-loss plot.

    Returns:
        Suggested optimal learning rate.
    """
    # Save weights to restore after sweep
    original_weights = model.get_weights()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=min_lr),
        loss=loss_fn,
    )

    finder = LRFinder(min_lr=min_lr, max_lr=max_lr, num_steps=num_steps)
    steps_per_epoch = num_steps
    model.fit(dataset, epochs=1, steps_per_epoch=steps_per_epoch, callbacks=[finder], verbose=1)

    suggested = finder.suggest_lr()
    print(f"Suggested LR: {suggested:.2e}")

    if plot_path:
        finder.plot(path=plot_path, suggested_lr=suggested)

    # Restore original weights
    model.set_weights(original_weights)

    return suggested
