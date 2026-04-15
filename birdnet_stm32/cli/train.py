"""CLI entry point for training."""

import argparse
import json
import math
import os

import numpy as np
import tensorflow as tf

from birdnet_stm32.data.dataset import load_file_paths_from_directory, upsample_minority_classes
from birdnet_stm32.data.generator import load_dataset
from birdnet_stm32.models.dscnn import build_dscnn_model
from birdnet_stm32.training.trainer import compute_hop_length, train_model


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train STM32N6 audio classifier")
    parser.add_argument("--data_path_train", type=str, required=True, help="Path to train dataset")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per class")
    parser.add_argument("--upsample_ratio", type=float, default=0.5, help="Upsample ratio for minority classes")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Audio sample rate (Hz)")
    parser.add_argument("--num_mels", type=int, default=64, help="Number of mel bins")
    parser.add_argument("--spec_width", type=int, default=256, help="Spectrogram width (frames)")
    parser.add_argument("--fft_length", type=int, default=512, help="FFT length")
    parser.add_argument("--chunk_duration", type=float, default=3, help="Audio chunk duration (seconds)")
    parser.add_argument("--max_duration", type=int, default=30, help="Max audio duration (seconds)")
    parser.add_argument(
        "--audio_frontend",
        type=str,
        default="hybrid",
        choices=["precomputed", "hybrid", "raw", "librosa", "tf"],
    )
    parser.add_argument("--mag_scale", type=str, default="pwl", choices=["pcen", "pwl", "db", "none"])
    parser.add_argument("--embeddings_size", type=int, default=256, help="Embeddings layer size")
    parser.add_argument("--alpha", type=float, default=1.0, help="Width multiplier")
    parser.add_argument("--depth_multiplier", type=int, default=1, help="Depth multiplier")
    parser.add_argument("--frontend_trainable", action="store_true", default=False)
    parser.add_argument("--mixup_alpha", type=float, default=0.2, help="Mixup alpha")
    parser.add_argument("--mixup_probability", type=float, default=0.25, help="Mixup batch fraction")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/best_model.keras")
    return parser.parse_args()


def main():
    """Train a DS-CNN model on a class-structured audio dataset."""
    args = get_args()

    # Early warning for STM32N6 raw frontend constraint
    if args.audio_frontend in ("tf", "raw"):
        T = int(args.sample_rate * args.chunk_duration)
        if T >= (1 << 16):
            print(f"[WARN] STM32N6 constraint: raw input length {T} >= 65536.")
            print("       Use --sample_rate 16000 or --chunk_duration 2, or --audio_frontend hybrid.")

    hop_length = compute_hop_length(args.sample_rate, args.chunk_duration, args.spec_width)

    # Load file paths
    file_paths, classes = load_file_paths_from_directory(args.data_path_train)

    # Train/val split
    split_idx = int(len(file_paths) * (1 - args.val_split))
    train_paths = file_paths[:split_idx]
    val_paths = file_paths[split_idx:]
    print(f"Training on {len(train_paths)} files, validating on {len(val_paths)} files.")

    # Upsample
    if args.upsample_ratio and 0 < args.upsample_ratio < 1.0:
        train_paths = upsample_minority_classes(train_paths, classes, args.upsample_ratio)
        print(f"After upsampling: {len(train_paths)} training files.")

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
    train_dataset = load_dataset(
        train_paths,
        classes,
        audio_frontend=args.audio_frontend,
        batch_size=args.batch_size,
        mixup_alpha=args.mixup_alpha,
        mixup_probability=args.mixup_probability,
        random_offset=True,
        snr_threshold=0.1,
        **common_kwargs,
    )
    val_dataset = load_dataset(
        val_paths,
        classes,
        audio_frontend=args.audio_frontend,
        batch_size=args.batch_size,
        mixup_alpha=0.0,
        mixup_probability=0.0,
        random_offset=False,
        snr_threshold=0.5,
        **common_kwargs,
    )

    steps_per_epoch = max(1, math.ceil(len(train_paths) / float(args.batch_size)))
    val_steps = max(1, math.ceil(len(val_paths) / float(args.batch_size)))

    # Build model
    print("Building model...")
    model = build_dscnn_model(
        num_mels=args.num_mels,
        spec_width=args.spec_width,
        sample_rate=args.sample_rate,
        chunk_duration=args.chunk_duration,
        audio_frontend=args.audio_frontend,
        num_classes=len(classes),
        alpha=args.alpha,
        depth_multiplier=args.depth_multiplier,
        embeddings_size=args.embeddings_size,
        fft_length=args.fft_length,
        mag_scale=args.mag_scale,
        frontend_trainable=args.frontend_trainable,
        class_activation="sigmoid" if args.mixup_probability > 0 else "softmax",
    )
    model.summary()

    # Save model config
    cfg = {
        "sample_rate": args.sample_rate,
        "num_mels": args.num_mels,
        "spec_width": args.spec_width,
        "fft_length": args.fft_length,
        "chunk_duration": args.chunk_duration,
        "hop_length": hop_length,
        "audio_frontend": args.audio_frontend,
        "mag_scale": args.mag_scale,
        "embeddings_size": args.embeddings_size,
        "alpha": args.alpha,
        "depth_multiplier": args.depth_multiplier,
        "num_classes": len(classes),
        "class_names": classes,
        "frontend_trainable": args.frontend_trainable,
    }
    cfg_path = os.path.splitext(args.checkpoint_path)[0] + "_model_config.json"
    os.makedirs(os.path.dirname(cfg_path) or ".", exist_ok=True)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved model config to '{cfg_path}'")

    # Train
    print("Starting training...")
    train_model(
        model,
        train_dataset,
        val_dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint_path,
        steps_per_epoch=steps_per_epoch,
        val_steps=val_steps,
        is_multilabel=(args.mixup_probability > 0),
    )
    print(f"Training complete. Best model saved to '{args.checkpoint_path}'.")

    # Save labels
    labels_file = args.checkpoint_path.replace(".keras", "_labels.txt")
    with open(labels_file, "w") as f:
        for cls in classes:
            f.write(f"{cls}\n")


if __name__ == "__main__":
    main()
