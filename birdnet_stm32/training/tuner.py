"""Optuna hyperparameter tuning for DS-CNN training."""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import optuna
import tensorflow as tf

from birdnet_stm32.data.dataset import load_file_paths_from_directory, upsample_minority_classes
from birdnet_stm32.data.generator import load_dataset
from birdnet_stm32.models.dscnn import build_dscnn_model
from birdnet_stm32.training.config import ModelConfig
from birdnet_stm32.training.trainer import compute_hop_length, train_model


def _build_search_space(trial: optuna.Trial, args: argparse.Namespace) -> dict:
    """Sample hyperparameters from the Optuna search space.

    Fixed architecture parameters (sample_rate, num_mels, spec_width, fft_length,
    chunk_duration, audio_frontend, mag_scale) are taken from the CLI args as-is.
    The search space covers training and model scaling knobs.

    Args:
        trial: Optuna trial object.
        args: CLI arguments (used as defaults and bounds).

    Returns:
        Dictionary of sampled hyperparameters.
    """
    return {
        "alpha": trial.suggest_float("alpha", 0.25, 1.5, step=0.25),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.2, 0.7, step=0.1),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "mixup_alpha": trial.suggest_float("mixup_alpha", 0.0, 0.4, step=0.1),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2, step=0.05),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw"]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "use_se": trial.suggest_categorical("use_se", [True, False]),
        "use_inverted_residual": trial.suggest_categorical("use_inverted_residual", [True, False]),
        "grad_clip": trial.suggest_float("grad_clip", 0.0, 5.0, step=1.0),
    }


def _objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """Optuna objective: train one configuration and return best val ROC-AUC.

    Args:
        trial: Optuna trial object.
        args: Base CLI arguments.

    Returns:
        Best validation ROC-AUC achieved during training.
    """
    hp = _build_search_space(trial, args)

    hop_length = compute_hop_length(args.sample_rate, args.chunk_duration, args.spec_width)

    # Load file paths
    file_paths, classes = load_file_paths_from_directory(args.data_path_train)
    split_idx = int(len(file_paths) * (1 - args.val_split))
    train_paths = file_paths[:split_idx]
    val_paths = file_paths[split_idx:]

    if args.upsample_ratio and 0 < args.upsample_ratio < 1.0:
        train_paths = upsample_minority_classes(train_paths, classes, args.upsample_ratio)

    # Datasets
    common_kwargs = dict(
        sample_rate=args.sample_rate,
        max_duration=args.max_duration,
        chunk_duration=args.chunk_duration,
        spec_width=args.spec_width,
        mel_bins=args.num_mels,
        fft_length=args.fft_length,
        mag_scale=args.mag_scale,
    )
    is_multilabel = hp["mixup_alpha"] > 0
    train_dataset = load_dataset(
        train_paths,
        classes,
        audio_frontend=args.audio_frontend,
        batch_size=hp["batch_size"],
        mixup_alpha=hp["mixup_alpha"],
        mixup_probability=args.mixup_probability if hp["mixup_alpha"] > 0 else 0.0,
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
        audio_frontend=args.audio_frontend,
        batch_size=hp["batch_size"],
        mixup_alpha=0.0,
        mixup_probability=0.0,
        random_offset=False,
        snr_threshold=0.5,
        spec_augment=False,
        **common_kwargs,
    )

    steps_per_epoch = max(1, math.ceil(len(train_paths) / float(hp["batch_size"])))
    val_steps = max(1, math.ceil(len(val_paths) / float(hp["batch_size"])))

    # Build model
    model = build_dscnn_model(
        num_mels=args.num_mels,
        spec_width=args.spec_width,
        sample_rate=args.sample_rate,
        chunk_duration=args.chunk_duration,
        audio_frontend=args.audio_frontend,
        num_classes=len(classes),
        alpha=hp["alpha"],
        depth_multiplier=args.depth_multiplier,
        embeddings_size=args.embeddings_size,
        fft_length=args.fft_length,
        mag_scale=args.mag_scale,
        frontend_trainable=args.frontend_trainable,
        class_activation="sigmoid" if is_multilabel else "softmax",
        dropout_rate=hp["dropout"],
        use_se=hp["use_se"],
        se_reduction=args.se_reduction,
        use_inverted_residual=hp["use_inverted_residual"],
        expansion_factor=args.expansion_factor,
        use_attention_pooling=args.use_attention_pooling,
    )

    # Loss
    loss_fn = None
    if hp["label_smoothing"] > 0:
        if is_multilabel:
            loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=hp["label_smoothing"])
        else:
            loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=hp["label_smoothing"])

    # Per-trial checkpoint in a subdirectory
    trial_dir = os.path.join(os.path.dirname(args.checkpoint_path), "optuna_trials")
    os.makedirs(trial_dir, exist_ok=True)
    trial_ckpt = os.path.join(trial_dir, f"trial_{trial.number}.keras")

    history = train_model(
        model,
        train_dataset,
        val_dataset,
        epochs=args.epochs,
        learning_rate=hp["learning_rate"],
        batch_size=hp["batch_size"],
        checkpoint_path=trial_ckpt,
        steps_per_epoch=steps_per_epoch,
        val_steps=val_steps,
        is_multilabel=is_multilabel,
        optimizer=hp["optimizer"],
        weight_decay=hp["weight_decay"] if hp["optimizer"] == "adamw" else 0.0,
        loss_fn=loss_fn,
        gradient_clip_norm=hp["grad_clip"],
    )

    best_roc_auc = max(history.history.get("val_roc_auc", [0.0]))

    # Prune unpromising trials early (report after each epoch via history)
    for epoch_idx, val_auc in enumerate(history.history.get("val_roc_auc", [])):
        trial.report(val_auc, epoch_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Clear Keras session to free GPU memory between trials
    tf.keras.backend.clear_session()

    return best_roc_auc


def run_tuning(args: argparse.Namespace) -> None:
    """Run Optuna hyperparameter search.

    Creates an Optuna study that maximizes val_roc_auc. After all trials,
    prints the best hyperparameters and saves them as a JSON file alongside
    the checkpoint directory.

    Args:
        args: CLI arguments including n_trials.
    """
    study = optuna.create_study(
        direction="maximize",
        study_name="birdnet_stm32_tune",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5),
    )

    study.optimize(lambda trial: _objective(trial, args), n_trials=args.n_trials)

    # Report results
    print("\n" + "=" * 60)
    print("Optuna tuning complete")
    print(f"  Best val_roc_auc: {study.best_value:.4f}")
    print(f"  Best trial: #{study.best_trial.number}")
    print("  Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    print("=" * 60)

    # Save best params as JSON
    import json

    results_path = os.path.join(
        os.path.dirname(args.checkpoint_path), "optuna_best_params.json"
    )
    best_data = {
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "best_params": study.best_trial.params,
        "n_trials": len(study.trials),
    }
    with open(results_path, "w") as f:
        json.dump(best_data, f, indent=2)
    print(f"Best parameters saved to '{results_path}'")

    # Copy best trial checkpoint to main checkpoint path
    best_trial_ckpt = os.path.join(
        os.path.dirname(args.checkpoint_path),
        "optuna_trials",
        f"trial_{study.best_trial.number}.keras",
    )
    if os.path.isfile(best_trial_ckpt):
        import shutil

        shutil.copy2(best_trial_ckpt, args.checkpoint_path)
        print(f"Best model copied to '{args.checkpoint_path}'")
