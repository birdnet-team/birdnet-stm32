"""CLI entry point for training."""

import argparse
import math
import os
import random
import time

import numpy as np
import tensorflow as tf

from birdnet_stm32.data.dataset import (
    get_classes_with_most_samples,
    load_file_paths_from_directory,
    upsample_minority_classes,
)
from birdnet_stm32.data.generator import estimate_samples_per_epoch, load_dataset
from birdnet_stm32.models.dscnn import build_dscnn_model
from birdnet_stm32.models.frontend import normalize_frontend_name
from birdnet_stm32.training.config import ModelConfig
from birdnet_stm32.training.losses import BinaryFocalLoss
from birdnet_stm32.training.trainer import compute_hop_length, train_model


def _read_meminfo_gb() -> tuple[float, float]:
    """Return (total_gb, available_gb) from /proc/meminfo."""
    mem_total_kb = 0
    mem_avail_kb = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    mem_avail_kb = int(line.split()[1])
    except OSError:
        return 0.0, 0.0
    return mem_total_kb / (1024 * 1024), mem_avail_kb / (1024 * 1024)


class AdaptiveLoaderTuner(tf.keras.callbacks.Callback):
    """Tune loader in-flight queue online using throughput and free RAM."""

    def __init__(
        self,
        control: dict,
        batch_size: int,
        adjust_every: int = 200,
        min_inflight: int = 128,
        max_inflight: int = 4096,
        target_free_gb: float = 8.0,
    ):
        super().__init__()
        self.control = control
        self.batch_size = batch_size
        self.adjust_every = adjust_every
        self.min_inflight = min_inflight
        self.max_inflight = max_inflight
        self.target_free_gb = target_free_gb
        self._t0 = 0.0
        self._steps = 0
        self._durations: list[float] = []
        self._prev_throughput: float | None = None
        self._direction = 1
        self._step_size = max(32, min(512, min_inflight // 2))

    def on_train_batch_begin(self, batch, logs=None):
        self._t0 = time.perf_counter()

    def on_train_batch_end(self, batch, logs=None):
        dt = time.perf_counter() - self._t0
        self._durations.append(dt)
        self._steps += 1
        if self._steps % self.adjust_every != 0:
            return

        step_sec = float(np.mean(self._durations)) if self._durations else 0.0
        self._durations.clear()
        throughput = self.batch_size / max(step_sec, 1e-6)
        _total_gb, avail_gb = _read_meminfo_gb()

        cur = int(self.control.get("max_inflight_files", self.min_inflight))
        new = cur
        reason = "hold"

        if avail_gb > 0 and avail_gb < self.target_free_gb:
            new = max(self.min_inflight, int(cur * 0.8))
            self._direction = -1
            self._step_size = max(16, self._step_size // 2)
            reason = "memory-pressure"
        elif self._prev_throughput is None:
            new = min(self.max_inflight, cur + self._step_size)
            self._direction = 1
            reason = "initial-probe"
        elif throughput >= self._prev_throughput * 1.01:
            if self._direction >= 0:
                new = min(self.max_inflight, cur + self._step_size)
                reason = "throughput-up"
            else:
                new = max(self.min_inflight, cur - self._step_size)
                reason = "throughput-up-down-dir"
        else:
            self._step_size = max(16, self._step_size // 2)
            if self._direction >= 0:
                new = max(self.min_inflight, cur - self._step_size)
                self._direction = -1
                reason = "reverse-down"
            else:
                new = min(self.max_inflight, cur + self._step_size)
                self._direction = 1
                reason = "reverse-up"

        self._prev_throughput = throughput
        if new != cur:
            self.control["max_inflight_files"] = int(new)
        print(
            f"[loader-tune] step={self._steps} throughput={throughput:.1f} samples/s "
            f"avail_mem={avail_gb:.1f}GB inflight {cur}->{int(self.control.get('max_inflight_files', cur))} "
            f"({reason})"
        )


# Internal defaults: keep CLI simple while auto-balancing loader throughput
# and memory usage at runtime.
_LOADER_TUNE_ADJUST_EVERY = 200
_LOADER_TARGET_FREE_GB = 8.0


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for training.

    Sensible defaults are chosen so that most users only need to specify
    ``--data_path_train``.  Architecture features that improve accuracy
    (SE attention, inverted residuals, SpecAugment, balanced class weights,
    deterministic seeding, gradient clipping, label smoothing) are **on by
    default** and can be disabled with ``--no_*`` flags when experimenting.
    """
    parser = argparse.ArgumentParser(description="Train STM32N6 audio classifier")

    # -- Data -----------------------------------------------------------------
    parser.add_argument("--data_path_train", type=str, required=True, help="Path to train dataset")
    parser.add_argument("--max_classes", type=int, default=None, help="Use top N classes by sample count")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per class")
    parser.add_argument("--upsample_ratio", type=float, default=0.5, help="Upsample ratio for minority classes")

    # -- Audio ----------------------------------------------------------------
    parser.add_argument("--sample_rate", type=int, default=24000, help="Audio sample rate (Hz)")
    parser.add_argument("--num_mels", type=int, default=64, help="Number of mel bins")
    parser.add_argument("--spec_width", type=int, default=256, help="Spectrogram width (frames)")
    parser.add_argument("--fft_length", type=int, default=512, help="FFT length")
    parser.add_argument("--chunk_duration", type=float, default=3, help="Audio chunk duration (seconds)")
    parser.add_argument(
        "--max_duration",
        type=int,
        default=60,
        help=(
            "Maximum seconds to read per file. The loader still reads only the bytes it needs "
            "for the candidate chunks (smart-crop bounded by --max_chunks_per_file)."
        ),
    )
    parser.add_argument(
        "--audio_frontend",
        type=str,
        default="hybrid",
        choices=["hybrid", "raw", "librosa", "mfcc", "log_mel"],
        help="Audio frontend mode",
    )
    parser.add_argument("--mag_scale", type=str, default="pwl", choices=["pcen", "pwl", "db", "none"])
    parser.add_argument("--n_mfcc", type=int, default=20, help="Number of MFCC coefficients (mfcc frontend only)")

    # -- Model architecture ---------------------------------------------------
    parser.add_argument("--embeddings_size", type=int, default=256, help="Embeddings layer size")
    parser.add_argument("--alpha", type=float, default=1.0, help="Width multiplier")
    parser.add_argument("--depth_multiplier", type=int, default=1, help="Depth multiplier")
    parser.add_argument("--frontend_trainable", action="store_true", default=False)
    parser.add_argument("--no_se", action="store_true", default=False, help="Disable SE channel attention")
    parser.add_argument("--se_reduction", type=int, default=8, help="SE channel reduction factor")
    parser.add_argument("--no_inverted_residual", action="store_true", default=False, help="Use plain DS blocks")
    parser.add_argument("--expansion_factor", type=int, default=2, help="Expansion factor for inverted residuals")
    parser.add_argument(
        "--use_attention_pooling", action="store_true", default=False, help="Use attention pooling instead of GAP"
    )

    # -- Augmentation ---------------------------------------------------------
    parser.add_argument("--no_spec_augment", action="store_true", default=False, help="Disable SpecAugment")
    parser.add_argument("--freq_mask_max", type=int, default=8, help="Max frequency mask width (bins)")
    parser.add_argument("--time_mask_max", type=int, default=25, help="Max time mask width (frames)")
    parser.add_argument("--mixup_alpha", type=float, default=0.2, help="Mixup alpha")
    parser.add_argument("--mixup_probability", type=float, default=0.25, help="Mixup batch fraction")

    # -- Training -------------------------------------------------------------
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Parallel data loading workers (0 = sequential)")
    parser.add_argument(
        "--max_chunks_per_file",
        type=int,
        default=3,
        help="Max salient chunks to extract per file open (reduces redundant I/O for long recordings)",
    )
    parser.add_argument(
        "--prefetch_batches",
        type=int,
        default=2,
        help="Loader prefetch queue depth in batches (higher = faster, but more RAM)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate before classifier head")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd", "adamw"], help="Optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay (adamw only)")
    parser.add_argument(
        "--loss",
        type=str,
        default="auto",
        choices=["auto", "focal"],
        help="Loss function. 'auto' selects based on mixup; 'focal' uses focal loss.",
    )
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma (focusing parameter)")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument(
        "--checkpoint_path", type=str, default="checkpoints/best_model.keras", help="Output checkpoint path (.keras)"
    )
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor (0 = off)")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Max gradient norm for clipping (0 = disabled)")
    parser.add_argument(
        "--no_class_weights",
        action="store_true",
        default=False,
        help="Disable balanced inverse-frequency class weighting",
    )
    parser.add_argument(
        "--mixed_precision", action="store_true", default=False, help="Enable FP16 mixed precision training"
    )
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic training")

    # -- Tuning & QAT --------------------------------------------------------
    parser.add_argument(
        "--tune", action="store_true", default=False, help="Run Optuna hyperparameter search instead of single training"
    )
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials (used with --tune)")
    parser.add_argument(
        "--qat",
        action="store_true",
        default=False,
        help="Quantization-aware fine-tuning (requires pretrained --checkpoint_path)",
    )

    # -- Linear probing -------------------------------------------------------
    parser.add_argument(
        "--linear_probe",
        action="store_true",
        default=False,
        help="Freeze backbone and train only the classifier head (requires pretrained --checkpoint_path)",
    )

    args = parser.parse_args()

    # Derive positive flags from --no_* flags
    args.use_se = not args.no_se
    args.use_inverted_residual = not args.no_inverted_residual
    args.spec_augment = not args.no_spec_augment
    args.class_weights = "none" if args.no_class_weights else "balanced"
    args.deterministic = True  # always deterministic

    return args


def main():
    """Train a DS-CNN model on a class-structured audio dataset."""
    args = get_args()
    args.audio_frontend = normalize_frontend_name(args.audio_frontend)

    # Enable dynamic GPU memory growth so we don't reserve all VRAM
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Deterministic mode: seed all RNGs and enable TF deterministic ops
    if args.deterministic:
        seed = args.seed
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"Deterministic mode enabled (seed={seed}).")

    # Early warning for STM32N6 raw frontend constraint
    if args.audio_frontend == "raw":
        T = int(args.sample_rate * args.chunk_duration)
        if T >= (1 << 16):
            print(f"[WARN] STM32N6 constraint: raw input length {T} >= 65536.")
            print("       Use --sample_rate 16000 or --chunk_duration 2, or --audio_frontend hybrid.")

    # Mixed precision
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled (float16 compute, float32 accumulation).")

    # Optuna hyperparameter tuning
    if args.tune:
        from birdnet_stm32.training.tuner import run_tuning

        run_tuning(args)
        return

    # Quantization-aware fine-tuning
    if args.qat:
        from birdnet_stm32.training.qat import run_qat

        run_qat(args)
        return

    # Linear probing: freeze backbone, retrain head only
    if args.linear_probe:
        from birdnet_stm32.training.linear_probe import run_linear_probe

        run_linear_probe(args)
        return

    hop_length = compute_hop_length(args.sample_rate, args.chunk_duration, args.spec_width)

    # Load file paths
    top_classes = None
    if args.max_classes is not None:
        top_classes = get_classes_with_most_samples(args.data_path_train, n_classes=args.max_classes)
        print(f"Selected top {len(top_classes)} classes by sample count.")
    file_paths, classes = load_file_paths_from_directory(
        args.data_path_train, classes=top_classes, max_samples=args.max_samples
    )

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
        prefetch_batches=args.prefetch_batches,
    )

    initial_inflight = max(256, args.num_workers * 64)
    common_kwargs["max_inflight_files"] = initial_inflight

    train_loader_control: dict | None = {"max_inflight_files": int(initial_inflight)} if args.num_workers > 0 else None
    extra_callbacks: list[tf.keras.callbacks.Callback] = []
    if train_loader_control is not None:
        extra_callbacks.append(
            AdaptiveLoaderTuner(
                control=train_loader_control,
                batch_size=args.batch_size,
                adjust_every=_LOADER_TUNE_ADJUST_EVERY,
                min_inflight=max(64, args.num_workers * 8),
                max_inflight=max(512, args.num_workers * 256),
                target_free_gb=_LOADER_TARGET_FREE_GB,
            )
        )
        print(f"Loader auto-tuning enabled (initial max_inflight_files={initial_inflight}).")

    train_kwargs = dict(common_kwargs)
    if train_loader_control is not None:
        train_kwargs["loader_control"] = train_loader_control

    val_kwargs = dict(common_kwargs)
    train_dataset = load_dataset(
        train_paths,
        classes,
        audio_frontend=args.audio_frontend,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_chunks_per_file=args.max_chunks_per_file,
        mixup_alpha=args.mixup_alpha,
        mixup_probability=args.mixup_probability,
        random_offset=True,
        snr_threshold=0.1,
        spec_augment=args.spec_augment,
        freq_mask_max=args.freq_mask_max,
        time_mask_max=args.time_mask_max,
        **train_kwargs,
    )
    val_dataset = load_dataset(
        val_paths,
        classes,
        audio_frontend=args.audio_frontend,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_chunks_per_file=1,
        mixup_alpha=0.0,
        mixup_probability=0.0,
        random_offset=False,
        snr_threshold=0.5,
        spec_augment=False,
        **val_kwargs,
    )

    steps_per_epoch = max(
        1, math.ceil(estimate_samples_per_epoch(len(train_paths), args.max_chunks_per_file) / float(args.batch_size))
    )
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
        dropout_rate=args.dropout,
        use_se=args.use_se,
        se_reduction=args.se_reduction,
        use_inverted_residual=args.use_inverted_residual,
        expansion_factor=args.expansion_factor,
        use_attention_pooling=args.use_attention_pooling,
    )
    model.summary()

    # Save model config
    cfg = ModelConfig(
        sample_rate=args.sample_rate,
        num_mels=args.num_mels,
        spec_width=args.spec_width,
        fft_length=args.fft_length,
        chunk_duration=args.chunk_duration,
        hop_length=hop_length,
        audio_frontend=args.audio_frontend,
        mag_scale=args.mag_scale,
        embeddings_size=args.embeddings_size,
        alpha=args.alpha,
        depth_multiplier=args.depth_multiplier,
        num_classes=len(classes),
        class_names=classes,
        frontend_trainable=args.frontend_trainable,
        n_mfcc=args.n_mfcc,
        use_se=args.use_se,
        se_reduction=args.se_reduction,
        use_inverted_residual=args.use_inverted_residual,
        expansion_factor=args.expansion_factor,
        use_attention_pooling=args.use_attention_pooling,
        dropout_rate=args.dropout,
    )
    cfg_path = os.path.splitext(args.checkpoint_path)[0] + "_model_config.json"
    cfg.save(cfg_path)
    print(f"Saved model config to '{cfg_path}'")

    # Resolve loss function
    is_multilabel = args.mixup_probability > 0
    loss_fn = None
    if args.loss == "focal":
        loss_fn = BinaryFocalLoss(gamma=args.focal_gamma)
    elif args.label_smoothing > 0:
        if is_multilabel:
            loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing)
        else:
            loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)

    # Class weights
    class_weights = None
    if args.class_weights == "balanced":
        from collections import Counter

        label_counts = Counter()
        for p in train_paths:
            label_str = p.split("/")[-2]
            if label_str in classes:
                label_counts[classes.index(label_str)] += 1
        if label_counts:
            total = sum(label_counts.values())
            n_classes = len(classes)
            class_weights = {i: total / (n_classes * label_counts.get(i, 1)) for i in range(n_classes)}
            print(
                f"Balanced class weights: min={min(class_weights.values()):.2f}, max={max(class_weights.values()):.2f}"
            )

    # Train
    print("Starting training...")
    try:
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
            is_multilabel=is_multilabel,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            loss_fn=loss_fn,
            gradient_clip_norm=args.grad_clip,
            class_weights=class_weights,
            resume=args.resume,
            extra_callbacks=extra_callbacks,
        )
        print(f"Training complete. Best model saved to '{args.checkpoint_path}'.")
    except KeyboardInterrupt:
        print(f"\nTraining interrupted. Best checkpoint so far: '{args.checkpoint_path}'")

    # Save labels
    labels_file = args.checkpoint_path.replace(".keras", "_labels.txt")
    with open(labels_file, "w") as f:
        for cls in classes:
            f.write(f"{cls}\n")


if __name__ == "__main__":
    main()
