"""CLI entry point for raw-frontend per-band equalization (before QAT or conversion)."""

import argparse
import json
import os
import shutil

from birdnet_stm32.conversion.equalize import STAGES, equalize_raw_frontend
from birdnet_stm32.conversion.quantize import calibration_source, representative_data_gen, stratified_sample_paths
from birdnet_stm32.models.runners import load_keras_model
from birdnet_stm32.training.config import ModelConfig

# Held-out calibration inputs used only to prove the float output is unchanged.
CHECK_SAMPLES = 64


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for equalization."""
    parser = argparse.ArgumentParser(
        description="Rescale a raw-frontend model's bands so each gets the same share of its INT8 grids. "
        "Exact in float; run it on a trained float checkpoint before QAT or conversion."
    )
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Trained float .keras model (raw frontend)")
    parser.add_argument("--model_config", type=str, default="", help="Path to model config JSON")
    parser.add_argument("--data_path_train", type=str, required=True, help="Training data directory")
    parser.add_argument(
        "--calibration_dir", type=str, default="", help="Draw the calibration audio from here instead (see convert)"
    )
    parser.add_argument("--output_path", type=str, required=True, help="Output .keras path")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1024,
        help="Stratified calibration files, drawn as by convert (seed 42)",
    )
    parser.add_argument(
        "--gain_samples",
        type=int,
        default=512,
        help=f"Calibration inputs that set the gains; the next {CHECK_SAMPLES} check float identity",
    )
    parser.add_argument("--stages", default=",".join(STAGES), help=f"Comma-separated subset of {','.join(STAGES)}")
    return parser.parse_args()


def main():
    """Equalize a raw-frontend checkpoint and save it with its config and a report."""
    args = get_args()
    if not args.model_config:
        args.model_config = os.path.splitext(args.checkpoint_path)[0] + "_model_config.json"
    if not os.path.isfile(args.model_config):
        raise FileNotFoundError(f"Model config JSON not found: {args.model_config}")
    if args.gain_samples + CHECK_SAMPLES > args.num_samples:
        raise SystemExit(f"--num_samples must be at least --gain_samples + {CHECK_SAMPLES}")
    cfg = ModelConfig.load(args.model_config).to_dict()

    file_paths = calibration_source(args.data_path_train, args.calibration_dir, cfg.get("class_names") or None)
    paths = stratified_sample_paths(file_paths, args.num_samples, seed=42)
    if len(paths) != args.num_samples:
        raise ValueError(f"Requested {args.num_samples} calibration paths but only {len(paths)} are available.")
    tensors = [item[0] for item in representative_data_gen(paths, cfg, num_samples=args.num_samples)]
    if len(tensors) != args.num_samples:
        raise RuntimeError(f"Calibration produced {len(tensors)} of {args.num_samples} tensors")

    model = load_keras_model(args.checkpoint_path)
    report = equalize_raw_frontend(
        model,
        tensors[: args.gain_samples],
        tensors[args.gain_samples : args.gain_samples + CHECK_SAMPLES],
        stages=[s for s in args.stages.split(",") if s],
    )
    report.update(checkpoint=os.path.abspath(args.checkpoint_path), num_samples=args.num_samples)
    for name in sorted(report["before"]):
        short = name.replace("audio_frontend_", "")
        print(f"{short:28s} spread {report['before'][name]:7.2f} -> {report['after'][name]:6.2f}")
    print(f"max |float output difference| over {CHECK_SAMPLES} held-out inputs: {report['max_abs_output_diff']:.2e}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    model.save(args.output_path)
    stem = os.path.splitext(args.output_path)[0]
    shutil.copy2(args.model_config, stem + "_model_config.json")
    labels = args.model_config.replace("_model_config.json", "_labels.txt")
    if labels != args.model_config and os.path.isfile(labels):
        shutil.copy2(labels, stem + "_labels.txt")
    with open(stem + "_equalization.json", "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Saved {args.output_path}")


if __name__ == "__main__":
    main()
