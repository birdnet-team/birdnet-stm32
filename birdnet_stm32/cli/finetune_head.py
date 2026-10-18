"""CLI entry point for the post-training head fine-tune (``birdnet_stm32.training.head_finetune``).

Run it on a trained float checkpoint, before ``equalize`` and ``convert``. Only the classifier ``pred`` changes.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from birdnet_stm32.cli.train import CLASS_CAP_PER_EPOCH, TEACHER_WEIGHT

HEAD_LEARNING_RATE = 3e-4
HEAD_EPOCHS = 30
HEAD_PATIENCE = 5
FAR_WEIGHT = 0.3
VALIDATION_SUBSET = 40_000


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Refit the classifier head on the frozen backbone over every training file (one cached chunk "
        "each), from its trained weights, class-capped per epoch as in training. Run after train, before equalize."
    )
    p.add_argument("--checkpoint_path", required=True, help="Trained float .keras model")
    p.add_argument("--model_config", default="", help="Its config JSON (default: <checkpoint>_model_config.json)")
    p.add_argument("--data_path_train", required=True, help="Training data directory (class folders + noise folders)")
    p.add_argument("--data_path_val", required=True, help="Validation directory (class folders), for epoch selection")
    p.add_argument("--output_path", required=True, help="Output .keras (the config is written beside it)")
    p.add_argument(
        "--cache_dir",
        default="",
        help="Embedding cache (default: <output dir>/head_cache; ~2 kB per file + 2 bytes per output)",
    )
    p.add_argument("--num_workers", type=int, default=8, help="Loader workers for the embedding pass")
    p.add_argument("--teacher_cache", default="", help="Teacher target cache, as in train")
    p.add_argument(
        "--teacher_weight",
        type=float,
        default=None,
        help=f"Teacher share of the targets (default {TEACHER_WEIGHT:g} with --teacher_cache, else 0)",
    )
    p.add_argument(
        "--teacher_background",
        action="store_true",
        help="Background birds as in train: the teacher's own score from 0.3, 1 from 0.8 (experimental)",
    )
    p.add_argument(
        "--crop_policy",
        choices=["energy", "teacher"],
        default=None,
        help="Which chunk of a file (default: teacher with --teacher_cache, else energy)",
    )
    p.add_argument("--label_sidecar", default="", help="Per-file label additions, as in train")
    p.add_argument("--field_table", default="", help="Field clips table, as in train (the class cap's field share)")
    p.add_argument("--field_share", type=float, default=0.5)
    p.add_argument("--field_per_recording", type=int, default=2)
    p.add_argument("--class_cap_per_epoch", type=int, default=CLASS_CAP_PER_EPOCH, help="0 = every file every epoch")
    p.add_argument("--epochs", type=int, default=HEAD_EPOCHS)
    p.add_argument("--patience", type=int, default=HEAD_PATIENCE, help="Epochs without validation gain before stopping")
    p.add_argument("--learning_rate", type=float, default=HEAD_LEARNING_RATE)
    p.add_argument(
        "--cooccurrence_csv",
        default="",
        help="label,group CSV (e.g. geographic regions): negatives from labels sharing no group with an "
        "output count --far_weight; labels not listed co-occur with everything",
    )
    p.add_argument("--far_weight", type=float, default=FAR_WEIGHT)
    p.add_argument("--validation_subset", type=int, default=VALIDATION_SUBSET, help="Stratified validation files")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = get_args()
    from birdnet_stm32.data.dataset import load_file_paths_from_directory
    from birdnet_stm32.models.runners import load_keras_model
    from birdnet_stm32.training import head_finetune as hf
    from birdnet_stm32.training.validation import stratified_validation_subset

    if not args.model_config:
        args.model_config = os.path.splitext(args.checkpoint_path)[0] + "_model_config.json"
    if not os.path.isfile(args.model_config):
        raise FileNotFoundError(f"Model config JSON not found: {args.model_config}")
    cfg = json.loads(Path(args.model_config).read_text())  # read as is: the config is copied verbatim
    classes = cfg["class_names"]
    if args.teacher_weight is None:
        args.teacher_weight = TEACHER_WEIGHT if args.teacher_cache else 0.0
    if args.crop_policy is None:
        args.crop_policy = "teacher" if args.teacher_cache else "energy"
    if args.crop_policy == "teacher" and not args.teacher_cache:
        raise SystemExit("--crop_policy teacher needs --teacher_cache")
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out.parent / "head_cache"

    model = load_keras_model(args.checkpoint_path)
    common = dict(
        audio_frontend=cfg["audio_frontend"],
        spec_width=cfg["spec_width"],
        mel_bins=cfg["num_mels"],
        sample_rate=cfg["sample_rate"],
        chunk_duration=cfg["chunk_duration"],
        fft_length=cfg.get("fft_length", 512),
        mag_scale=cfg.get("mag_scale", "pwl"),
        input_compression=cfg.get("input_compression", "none"),
    )
    train_kwargs = dict(common, crop_policy=args.crop_policy)
    if args.teacher_cache:
        train_kwargs.update(
            teacher_cache=args.teacher_cache,
            teacher_weight=args.teacher_weight,
            teacher_background=args.teacher_background,
        )
    if args.label_sidecar:
        from birdnet_stm32.data.sidecar import load_label_sidecar

        train_kwargs["label_sidecar"] = load_label_sidecar(args.label_sidecar, classes)

    train_paths, _ = load_file_paths_from_directory(args.data_path_train, classes=classes)
    train_paths = sorted(train_paths)
    print(f"Embedding {len(train_paths)} training files once (cache {cache_dir})")
    hf.embed_files(model, train_paths, classes, cache_dir / "train", train_kwargs, num_workers=args.num_workers)
    val_paths, _ = load_file_paths_from_directory(args.data_path_val, classes=classes)
    val_paths = sorted(
        p
        for p in stratified_validation_subset(val_paths, args.validation_subset)
        if p.replace("\\", "/").split("/")[-2] in set(classes)
    )
    print(f"Embedding {len(val_paths)} validation files (hard labels)")
    hf.embed_files(model, val_paths, classes, cache_dir / "val", dict(common), num_workers=args.num_workers)

    import numpy as np

    t = cache_dir / "train"
    emb, ys = np.load(t / "emb.npy", mmap_mode="r"), np.load(t / "y.npy", mmap_mode="r")
    ok, cls = np.load(t / "ok.npy"), np.load(t / "cls.npy")
    v = cache_dir / "val"
    vok = np.load(v / "ok.npy") == 1
    val_emb, val_y = np.load(v / "emb.npy")[vok].astype(np.float32), np.load(v / "y.npy")[vok].astype(np.float32)

    field_kwargs = {}
    if args.field_table:
        from birdnet_stm32.data.sidecar import load_field_table

        field_kwargs = dict(
            field=load_field_table(args.field_table),
            field_share=args.field_share,
            per_recording=args.field_per_recording,
        )
    epoch_rows = hf.capped_epochs(
        (t / "files.txt").read_text().splitlines(), ok, args.class_cap_per_epoch, field_kwargs, seed=args.seed
    )
    weights = None
    if args.cooccurrence_csv:
        co = hf.load_cooccurrence(args.cooccurrence_csv, classes)
        weights = hf.negative_weights(co, args.far_weight)
        print(
            f"Co-occurrence weighting: {co.mean():.0%} of output pairs co-occur; other negatives count {args.far_weight:g}"
        )
    w0, b0 = model.get_layer("pred").get_weights()
    z0 = val_emb @ w0 + b0
    before = {"val_class_ap": round(hf.class_mean_ap(val_y, z0), 4), "val_micro_ap": round(hf.micro_ap(val_y, z0), 4)}
    print(f"Trained head: val class AP {before['val_class_ap']}, micro AP {before['val_micro_ap']}")
    w, b, history = hf.fit_head(
        emb,
        ys,
        cls,
        epoch_rows,
        val_emb,
        val_y,
        w0,
        b0,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        patience=args.patience,
        weights=weights,
        seed=args.seed,
    )
    hf.replace_head(model, w, b)
    model.save(out)
    out_cfg = out.with_name(out.stem + "_model_config.json")
    shutil.copyfile(args.model_config, out_cfg)  # only pred's weights changed
    report = {
        "checkpoint": str(args.checkpoint_path),
        "training_files": int((ok == 1).sum()),
        "unreadable_files": int((ok == 2).sum()),
        "validation_files": int(vok.sum()),
        "settings": {k: v for k, v in vars(args).items() if k not in ("checkpoint_path", "output_path")},
        "before": before,
        "history": history,
        "selected_epoch": next(h["epoch"] for h in history if h["selected"]),
    }
    hf.save_report(out.with_name(out.stem + "_head_finetune.json"), report)
    sel = next(h for h in history if h["selected"])
    print(
        f"Saved {out} (epoch {sel['epoch']}: val class AP {sel['val_class_ap']}, micro AP {sel['val_micro_ap']}; "
        f"before {before['val_class_ap']} / {before['val_micro_ap']})"
    )
    print(json.dumps({"config": str(out_cfg)}))


if __name__ == "__main__":
    main()
