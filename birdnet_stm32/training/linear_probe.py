"""Linear probing: freeze a pretrained backbone and train only the classifier head.

This enables transfer learning where users provide their own dataset and
the backbone's learned representations are re-used without modification.
Only the final dropout + dense layer are retrained, making training fast
and requiring very little data.
"""

import math
import os

import tensorflow as tf

from birdnet_stm32.data.dataset import load_file_paths_from_directory, upsample_minority_classes
from birdnet_stm32.data.generator import load_dataset
from birdnet_stm32.models.frontend import AudioFrontendLayer, normalize_frontend_name
from birdnet_stm32.models.magnitude import MagnitudeScalingLayer
from birdnet_stm32.training.config import ModelConfig
from birdnet_stm32.training.trainer import compute_hop_length, train_model


def run_linear_probe(args) -> None:
    """Run linear probing on a pretrained model with a new dataset.

    Loads the model from ``args.checkpoint_path``, freezes all layers except
    the classifier head, rebuilds the head to match the new class count from
    ``args.data_path_train``, and fine-tunes.

    Args:
        args: Parsed CLI arguments (same namespace as ``cli/train.py``).
    """
    if not os.path.isfile(args.checkpoint_path):
        raise FileNotFoundError(f"Pretrained model not found: {args.checkpoint_path}")

    # Load pretrained model config
    cfg_path = os.path.splitext(args.checkpoint_path)[0] + "_model_config.json"
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Model config not found: {cfg_path}. Train a full model first.")
    old_cfg = ModelConfig.load(cfg_path)

    # Load pretrained model
    print(f"[linear-probe] Loading pretrained model from {args.checkpoint_path}")
    base_model = tf.keras.models.load_model(
        args.checkpoint_path,
        compile=False,
        custom_objects={
            "AudioFrontendLayer": AudioFrontendLayer,
            "MagnitudeScalingLayer": MagnitudeScalingLayer,
        },
    )

    # Discover new classes from the user's dataset
    file_paths, classes = load_file_paths_from_directory(args.data_path_train)
    if not classes:
        raise ValueError("No classes found in the training data.")
    print(f"[linear-probe] {len(classes)} target classes, {len(file_paths)} files")

    # Find the embeddings layer (just before dropout/dense head)
    # Strategy: find the last GlobalAveragePooling2D or the attention pool output
    embedding_layer = None
    for layer in reversed(base_model.layers):
        if isinstance(layer, (tf.keras.layers.GlobalAveragePooling2D, tf.keras.layers.Flatten)):
            embedding_layer = layer
            break
        if "attn_pool" in layer.name or "gap" in layer.name:
            embedding_layer = layer
            break

    if embedding_layer is None:
        raise RuntimeError("Could not find embedding layer (GAP/attention pool) in pretrained model.")

    # Build new model: backbone (frozen) → dropout → new dense head
    backbone_output = embedding_layer.output
    x = tf.keras.layers.Dropout(args.dropout, name="probe_dropout")(backbone_output)
    activation = "sigmoid" if args.mixup_probability > 0 else "softmax"
    outputs = tf.keras.layers.Dense(len(classes), activation=activation, name="probe_pred")(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs, name="linear_probe")

    # Freeze all layers except the new head
    for layer in model.layers:
        if layer.name in ("probe_dropout", "probe_pred"):
            layer.trainable = True
        else:
            layer.trainable = False

    trainable_count = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    total_count = sum(tf.keras.backend.count_params(w) for w in model.weights)
    print(f"[linear-probe] Trainable params: {trainable_count:,} / {total_count:,} total")
    model.summary()

    # Prepare data
    audio_frontend = normalize_frontend_name(old_cfg.audio_frontend)
    hop_length = compute_hop_length(old_cfg.sample_rate, old_cfg.chunk_duration, old_cfg.spec_width)

    split_idx = int(len(file_paths) * (1 - args.val_split))
    train_paths = file_paths[:split_idx]
    val_paths = file_paths[split_idx:]
    print(f"[linear-probe] Training on {len(train_paths)} files, validating on {len(val_paths)} files.")

    if args.upsample_ratio and 0 < args.upsample_ratio < 1.0:
        train_paths = upsample_minority_classes(train_paths, classes, args.upsample_ratio)

    common_kwargs = dict(
        sample_rate=old_cfg.sample_rate,
        max_duration=args.max_duration,
        chunk_duration=old_cfg.chunk_duration,
        spec_width=old_cfg.spec_width,
        mel_bins=old_cfg.num_mels,
        fft_length=old_cfg.fft_length,
        mag_scale=old_cfg.mag_scale,
    )
    train_dataset = load_dataset(
        train_paths,
        classes,
        audio_frontend=audio_frontend,
        batch_size=args.batch_size,
        mixup_alpha=args.mixup_alpha,
        mixup_probability=args.mixup_probability,
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
        audio_frontend=audio_frontend,
        batch_size=args.batch_size,
        mixup_alpha=0.0,
        mixup_probability=0.0,
        random_offset=False,
        snr_threshold=0.5,
        spec_augment=False,
        **common_kwargs,
    )

    steps_per_epoch = max(1, math.ceil(len(train_paths) / float(args.batch_size)))
    val_steps = max(1, math.ceil(len(val_paths) / float(args.batch_size)))

    is_multilabel = args.mixup_probability > 0

    # Save updated model config for the new classes
    new_cfg = ModelConfig(
        sample_rate=old_cfg.sample_rate,
        num_mels=old_cfg.num_mels,
        spec_width=old_cfg.spec_width,
        fft_length=old_cfg.fft_length,
        chunk_duration=old_cfg.chunk_duration,
        hop_length=hop_length,
        audio_frontend=audio_frontend,
        mag_scale=old_cfg.mag_scale,
        embeddings_size=old_cfg.embeddings_size,
        alpha=old_cfg.alpha,
        depth_multiplier=old_cfg.depth_multiplier,
        num_classes=len(classes),
        class_names=classes,
        frontend_trainable=old_cfg.frontend_trainable,
        n_mfcc=old_cfg.n_mfcc,
        use_se=old_cfg.use_se,
        se_reduction=old_cfg.se_reduction,
        use_inverted_residual=old_cfg.use_inverted_residual,
        expansion_factor=old_cfg.expansion_factor,
        use_attention_pooling=old_cfg.use_attention_pooling,
        dropout_rate=args.dropout,
    )
    out_cfg_path = os.path.splitext(args.checkpoint_path)[0] + "_probe_model_config.json"
    new_cfg.save(out_cfg_path)
    print(f"[linear-probe] Saved probe config to '{out_cfg_path}'")

    out_checkpoint = os.path.splitext(args.checkpoint_path)[0] + "_probe.keras"

    # Train (head only)
    print("[linear-probe] Starting head-only training...")
    train_model(
        model,
        train_dataset,
        val_dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        checkpoint_path=out_checkpoint,
        steps_per_epoch=steps_per_epoch,
        val_steps=val_steps,
        is_multilabel=is_multilabel,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.grad_clip,
    )
    print(f"[linear-probe] Done. Model saved to '{out_checkpoint}'.")

    # Save labels
    labels_file = out_checkpoint.replace(".keras", "_labels.txt")
    with open(labels_file, "w") as f:
        for cls in classes:
            f.write(f"{cls}\n")
    print(f"[linear-probe] Labels saved to '{labels_file}'.")

    # Also save a numpy embedding extractor for convenience
    print(f"[linear-probe] Backbone embeddings dim: {embedding_layer.output.shape[-1]}")
