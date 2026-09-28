"""CLI entry point: add a compressing second exposure to a trained raw model (then fine-tune it)."""

import argparse
import json
import os
import shutil

from birdnet_stm32.conversion.exposure import add_exposure
from birdnet_stm32.conversion.quantize import calibration_source, representative_data_gen, stratified_sample_paths
from birdnet_stm32.models.runners import load_keras_model
from birdnet_stm32.training.config import ModelConfig


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Switch a trained raw model to a two-exposure, compressing filterbank. The result must be "
        "fine-tuned (train --init_checkpoint) before equalize, QAT and conversion."
    )
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Trained float .keras model (raw frontend)")
    parser.add_argument("--model_config", type=str, default="", help="Path to model config JSON")
    parser.add_argument("--data_path_train", type=str, required=True, help="Training data directory")
    parser.add_argument("--output_path", type=str, required=True, help="Output .keras path")
    parser.add_argument(
        "--gain", type=float, default=16.0, help="Gain of the second exposure (the knee sits at 1/gain)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=512, help="Stratified training windows that normalize the bank (seed 42)"
    )
    return parser.parse_args()


def main():
    """Add the exposure and save the model with its config, labels and a report."""
    args = get_args()
    if not args.model_config:
        args.model_config = os.path.splitext(args.checkpoint_path)[0] + "_model_config.json"
    cfg = ModelConfig.load(args.model_config).to_dict()
    if cfg["audio_frontend"] != "raw":
        raise SystemExit("add-exposure applies to raw models only")

    file_paths = calibration_source(args.data_path_train, "", cfg.get("class_names") or None)
    paths = stratified_sample_paths(file_paths, args.num_samples, seed=42)
    tensors = [item[0] for item in representative_data_gen(paths, cfg, num_samples=len(paths))]
    if not tensors:
        raise RuntimeError("No calibration windows")

    model, report = add_exposure(load_keras_model(args.checkpoint_path), cfg, tensors, gain=args.gain)
    report.update(checkpoint=os.path.abspath(args.checkpoint_path), num_samples=len(tensors))
    print(json.dumps(report, indent=2))

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    model.save(args.output_path)
    stem = os.path.splitext(args.output_path)[0]
    cfg.update(raw_exposure_gain=float(args.gain), raw_exposure_mode="compress")
    ModelConfig.from_dict(cfg).save(stem + "_model_config.json")
    labels = args.model_config.replace("_model_config.json", "_labels.txt")
    if labels != args.model_config and os.path.isfile(labels):
        shutil.copy2(labels, stem + "_labels.txt")
    with open(stem + "_exposure.json", "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Saved {args.output_path}; fine-tune it with train --init_checkpoint")


if __name__ == "__main__":
    main()
