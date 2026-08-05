"""CLI entry point for TFLite conversion."""

import argparse
import json
import os
import random
import tempfile
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from birdnet_stm32.audio.activity import pick_random_samples
from birdnet_stm32.conversion.quantize import convert_to_tflite, representative_data_gen
from birdnet_stm32.conversion.validate import validate_models
from birdnet_stm32.data.dataset import load_file_paths_from_directory
from birdnet_stm32.models.frontend import AudioFrontendLayer, hybrid_fft_bins, normalize_frontend_name
from birdnet_stm32.models.magnitude import MagnitudeScalingLayer
from birdnet_stm32.training.config import ModelConfig

random.seed(42)
np.random.seed(42)


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for conversion."""
    parser = argparse.ArgumentParser(
        description="Convert Keras model to quantized TFLite (float32 I/O, INT8 internal)."
    )
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to trained .keras model")
    parser.add_argument("--model_config", type=str, default="", help="Path to model config JSON")
    parser.add_argument("--output_path", type=str, default="", help="Output .tflite path")
    parser.add_argument("--data_path_train", type=str, default="", help="Training data directory for rep. dataset")
    parser.add_argument("--num_samples", type=int, default=1024, help="Representative dataset samples")
    parser.add_argument("--validate_samples", type=int, default=256, help="Validation samples")
    parser.add_argument(
        "--min_cosine_sim",
        type=float,
        default=0.95,
        help="Minimum mean cosine similarity threshold. Conversion fails if below (0 to disable).",
    )
    parser.add_argument(
        "--min_cosine_p05",
        type=float,
        default=0.90,
        help="Minimum fifth-percentile cosine similarity (0 to disable).",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="ptq",
        choices=["ptq", "dynamic"],
        help="Quantization mode: 'ptq' (full INT8 with calibration, default) or 'dynamic' (dynamic range, no calibration data needed).",
    )
    parser.add_argument(
        "--per_tensor",
        action="store_true",
        default=False,
        help="Use per-tensor quantization instead of per-channel (default). Per-channel is more accurate.",
    )
    parser.add_argument(
        "--batch_validate",
        type=int,
        default=0,
        help="Run validation N times with different random seeds and report worst-case metrics (0 = off).",
    )
    parser.add_argument(
        "--export_onnx",
        action="store_true",
        default=False,
        help="Also export an ONNX model (requires tf2onnx).",
    )
    parser.add_argument(
        "--report_json",
        type=str,
        default="",
        help="Path to save a structured JSON conversion report.",
    )
    return parser.parse_args()


def _stratified_sample_paths(
    file_paths: list[str],
    num_samples: int,
    *,
    seed: int,
    exclude: set[str] | None = None,
) -> list[str]:
    """Select exactly ``num_samples`` paths with balanced class coverage."""
    excluded = exclude or set()
    grouped: dict[str, list[str]] = defaultdict(list)
    for path in file_paths:
        if path not in excluded:
            grouped[os.path.basename(os.path.dirname(path))].append(path)

    rng = random.Random(seed)
    class_names = sorted(grouped)
    rng.shuffle(class_names)
    for paths in grouped.values():
        rng.shuffle(paths)

    selected: list[str] = []
    offsets = {name: 0 for name in class_names}
    while len(selected) < num_samples:
        added = False
        for name in class_names:
            offset = offsets[name]
            paths = grouped[name]
            if offset < len(paths):
                selected.append(paths[offset])
                offsets[name] = offset + 1
                added = True
                if len(selected) == num_samples:
                    break
        if not added:
            break
    return selected


def _write_report(path: str, report: dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"Conversion report saved to {path}")


def main():
    """Convert a trained Keras model to quantized TFLite and validate."""
    args = get_args()

    # Resolve config path
    if not args.model_config:
        args.model_config = os.path.splitext(args.checkpoint_path)[0] + "_model_config.json"
    if not os.path.isfile(args.model_config):
        raise FileNotFoundError(f"Model config JSON not found: {args.model_config}")
    cfg = ModelConfig.load(args.model_config).to_dict()

    # Load model
    model = tf.keras.models.load_model(
        args.checkpoint_path,
        compile=False,
        custom_objects={"AudioFrontendLayer": AudioFrontendLayer, "MagnitudeScalingLayer": MagnitudeScalingLayer},
    )
    print(f"Loaded model from {args.checkpoint_path}")

    # Build representative dataset generator
    if os.path.isdir(args.data_path_train):
        configured_classes = cfg.get("class_names") or None
        file_paths, classes = load_file_paths_from_directory(args.data_path_train, classes=configured_classes)
        if not file_paths:
            raise ValueError("No training audio found for the classes in the model config.")

        class_count = len({os.path.basename(os.path.dirname(path)) for path in file_paths})
        stratified_paths = _stratified_sample_paths(file_paths, args.num_samples, seed=42)
        print(f"Representative dataset: {len(stratified_paths)} stratified samples from {class_count} folders.")

        def rep_data_gen():
            return representative_data_gen(stratified_paths, cfg, num_samples=len(stratified_paths))

        # Calibration and validation must be disjoint; overlap makes parity
        # reports optimistic and invalidates a release gate.
        val_paths_subset = _stratified_sample_paths(
            file_paths,
            args.validate_samples,
            seed=43,
            exclude=set(stratified_paths),
        )
        if not val_paths_subset:
            raise ValueError("No files remain for disjoint quantization validation.")
        print(f"Validation dataset: {len(val_paths_subset)} disjoint stratified samples.")

        def rep_data_gen_val():
            return representative_data_gen(val_paths_subset, cfg, num_samples=len(val_paths_subset))
    else:
        print("No training data directory provided; generating random representative dataset.")

        def rep_data_gen(num_samples=args.num_samples):
            sr = int(cfg["sample_rate"])
            cd = cfg["chunk_duration"]
            T = int(sr * cd)
            spec_width = int(cfg["spec_width"])
            n_fft = int(cfg["fft_length"])
            frontend = normalize_frontend_name(cfg["audio_frontend"])
            num_mels = int(cfg["num_mels"])
            fft_bins = hybrid_fft_bins(n_fft)
            for _ in tqdm(range(num_samples), desc="Random samples", unit="sample"):
                if frontend == "librosa":
                    yield [np.random.rand(1, num_mels, spec_width, 1).astype(np.float32)]
                elif frontend == "hybrid":
                    yield [np.random.rand(1, fft_bins, spec_width, 1).astype(np.float32)]
                else:
                    yield [np.random.randn(1, T, 1).astype(np.float32)]

        def rep_data_gen_val():
            return rep_data_gen(num_samples=args.validate_samples)

    # Output path
    if not args.output_path:
        args.output_path = os.path.splitext(args.checkpoint_path)[0] + "_quantized.tflite"

    # Conversion is staged beside the destination and promoted atomically only
    # after every numerical quality gate passes.  A failed conversion must
    # never leave a release-looking .tflite artifact behind.
    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(output_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=".quantizing-", suffix=".tflite", dir=output_dir, delete=False
    ) as tmp_handle:
        tmp_path = tmp_handle.name
    report: dict = {
        "output_path": args.output_path,
        "quantization": args.quantization,
        "per_tensor": args.per_tensor,
        "quality_gate_passed": False,
    }

    try:
        convert_to_tflite(model, rep_data_gen, tmp_path, quantization=args.quantization, per_tensor=args.per_tensor)

        n_runs = max(1, args.batch_validate) if args.batch_validate > 0 else 1
        all_metrics: list[dict] = []
        for run_idx in range(n_runs):
            if n_runs > 1:
                print(f"\n--- Validation run {run_idx + 1}/{n_runs} ---")
            val_metrics = validate_models(model, tmp_path, rep_data_gen_val)
            all_metrics.append(val_metrics)

        # Aggregate metrics across runs. Input manifests are deterministic, so
        # repeated runs measure runtime repeatability rather than resampling.
        if n_runs > 1:
            print(f"\n--- Batch validation summary ({n_runs} runs) ---")
            for key in ["cosine_mean", "cosine_p05", "mse_mean", "mae_mean", "pearson_mean"]:
                vals = [m[key] for m in all_metrics]
                worst = min(vals) if "cosine" in key or "pearson" in key else max(vals)
                mean = np.mean(vals)
                print(f"  {key}: mean={mean:.6f}  worst={worst:.6f}")
            report["batch_validation"] = {"n_runs": n_runs, "all_metrics": all_metrics}
            val_metrics = dict(all_metrics[0])
            val_metrics["cosine_mean"] = min(m["cosine_mean"] for m in all_metrics)
            val_metrics["cosine_p05"] = min(m["cosine_p05"] for m in all_metrics)
        else:
            val_metrics = all_metrics[0]
        report["validation"] = val_metrics

        failures = []
        cos_mean = val_metrics["cosine_mean"]
        cos_p05 = val_metrics["cosine_p05"]
        if args.min_cosine_sim > 0 and cos_mean < args.min_cosine_sim:
            failures.append(f"mean cosine {cos_mean:.6f} < {args.min_cosine_sim:.4f}")
        if args.min_cosine_p05 > 0 and cos_p05 < args.min_cosine_p05:
            failures.append(f"p05 cosine {cos_p05:.6f} < {args.min_cosine_p05:.4f}")
        if failures:
            report["quality_gate_failures"] = failures
            _write_report(args.report_json, report)
            raise RuntimeError("Quantization quality check failed: " + "; ".join(failures))

        report["quality_gate_passed"] = True
        os.replace(tmp_path, args.output_path)
        tmp_path = ""
        print(f"TFLite model validated and saved to {args.output_path}")

        # Save validation data
        # Save labels only for a model that passed the gate.
        if cfg.get("class_names"):
            labels_path = os.path.splitext(args.output_path)[0] + "_labels.txt"
            with open(labels_path, "w", encoding="utf-8") as handle:
                handle.writelines(f"{name}\n" for name in cfg["class_names"])
            print(f"Labels saved to {labels_path}")

        validation_batches = [sample[0] for sample in rep_data_gen_val()]
        validation_data = np.concatenate(validation_batches, axis=0)
        if validation_data.shape[0] > 25:
            validation_data = pick_random_samples(validation_data, 25)
        val_path = os.path.splitext(args.output_path)[0] + "_validation_data.npz"
        np.savez_compressed(val_path, data=validation_data)
        print(f"Validation data saved to {val_path}")

        # ONNX export
        if args.export_onnx:
            onnx_path = os.path.splitext(args.output_path)[0] + ".onnx"
            try:
                import tf2onnx

                spec = (tf.TensorSpec(model.input_shape, tf.float32, name="input"),)
                tf2onnx.convert.from_keras(model, input_signature=spec, output_path=onnx_path)
                print(f"ONNX model saved to {onnx_path}")
                report["onnx_path"] = onnx_path
            except ImportError:
                raise RuntimeError("ONNX export requested but tf2onnx is not installed") from None
            except Exception as exc:
                raise RuntimeError(f"ONNX export failed: {exc}") from exc

        report["model_size_bytes"] = os.path.getsize(args.output_path)
        report["keras_size_bytes"] = os.path.getsize(args.checkpoint_path)
        report["compression_ratio"] = report["keras_size_bytes"] / max(report["model_size_bytes"], 1)
        report["config"] = cfg
        _write_report(args.report_json, report)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    main()
