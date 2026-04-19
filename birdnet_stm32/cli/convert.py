"""CLI entry point for TFLite conversion."""

import argparse
import os
import random

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from birdnet_stm32.audio.activity import pick_random_samples
from birdnet_stm32.conversion.quantize import convert_to_tflite, representative_data_gen
from birdnet_stm32.conversion.validate import validate_models
from birdnet_stm32.data.dataset import load_file_paths_from_directory
from birdnet_stm32.models.frontend import AudioFrontendLayer, normalize_frontend_name
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
        file_paths, classes = load_file_paths_from_directory(args.data_path_train)

        # Stratified sampling: balance classes in representative dataset
        from collections import defaultdict

        class_files: dict[str, list[str]] = defaultdict(list)
        for p in file_paths:
            cls = os.path.basename(os.path.dirname(p))
            class_files[cls].append(p)

        per_class = max(1, args.num_samples // max(len(class_files), 1))
        stratified_paths: list[str] = []
        for cls_name, paths in class_files.items():
            n = min(per_class, len(paths))
            stratified_paths.extend(random.sample(paths, n))
        random.shuffle(stratified_paths)
        # Cap at num_samples
        stratified_paths = stratified_paths[: args.num_samples]
        print(f"Representative dataset: {len(stratified_paths)} stratified samples from {len(class_files)} classes.")

        def rep_data_gen():
            return representative_data_gen(stratified_paths, cfg, num_samples=len(stratified_paths))

        # Validation uses a different subset
        val_paths_subset = random.sample(file_paths, min(args.validate_samples, len(file_paths)))

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
            fft_bins = n_fft // 2 + 1
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

    # Convert
    convert_to_tflite(model, rep_data_gen, args.output_path, quantization=args.quantization, per_tensor=args.per_tensor)
    print(f"TFLite model saved to {args.output_path}")

    # Validate (single run or batch)
    report: dict = {"output_path": args.output_path, "quantization": args.quantization, "per_tensor": args.per_tensor}

    n_runs = max(1, args.batch_validate) if args.batch_validate > 0 else 1
    all_metrics: list[dict] = []
    for run_idx in range(n_runs):
        if n_runs > 1:
            print(f"\n--- Validation run {run_idx + 1}/{n_runs} ---")
            random.seed(run_idx)
            np.random.seed(run_idx)
        val_metrics = validate_models(model, args.output_path, rep_data_gen_val)
        all_metrics.append(val_metrics)

    # Aggregate metrics across runs
    if n_runs > 1:
        print(f"\n--- Batch validation summary ({n_runs} runs) ---")
        for key in ["cosine_mean", "mse_mean", "mae_mean", "pearson_mean"]:
            vals = [m[key] for m in all_metrics]
            worst = min(vals) if "cosine" in key or "pearson" in key else max(vals)
            mean = np.mean(vals)
            print(f"  {key}: mean={mean:.6f}  worst={worst:.6f}")
        report["batch_validation"] = {"n_runs": n_runs, "all_metrics": all_metrics}
        # Use worst-case cosine for threshold check
        val_metrics = {"cosine_mean": min(m["cosine_mean"] for m in all_metrics)}
    else:
        val_metrics = all_metrics[0]
    report["validation"] = val_metrics

    # Reset seeds
    random.seed(42)
    np.random.seed(42)

    # Check cosine similarity threshold
    if args.min_cosine_sim > 0:
        cos_mean = val_metrics["cosine_mean"]
        if cos_mean < args.min_cosine_sim:
            raise RuntimeError(
                f"Quantization quality check failed: mean cosine similarity {cos_mean:.6f} "
                f"< threshold {args.min_cosine_sim:.4f}. "
                f"Consider using a more representative calibration dataset or a simpler model."
            )
        print(f"Cosine similarity check passed: {cos_mean:.6f} >= {args.min_cosine_sim:.4f}")

    # Save validation data
    validation_data = []
    for sample in rep_data_gen_val():
        validation_data.append(sample[0])
    validation_data = np.array(validation_data)
    if validation_data.shape[0] > 25:
        validation_data = pick_random_samples(validation_data, 25)
    val_path = os.path.splitext(args.output_path)[0] + "_validation_data.npz"
    np.savez_compressed(val_path, data=validation_data)
    print(f"Validation data saved to {val_path}")

    # ONNX export
    if args.export_onnx:
        onnx_path = os.path.splitext(args.checkpoint_path)[0] + ".onnx"
        try:
            import tf2onnx

            spec = (tf.TensorSpec(model.input_shape, tf.float32, name="input"),)
            model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=onnx_path)
            print(f"ONNX model saved to {onnx_path}")
            report["onnx_path"] = onnx_path
        except ImportError:
            print("[WARN] tf2onnx not installed. Skipping ONNX export (pip install tf2onnx).")
        except Exception as e:
            print(f"[WARN] ONNX export failed: {e}")

    # Save conversion report
    if args.report_json:
        import json

        report["model_size_bytes"] = os.path.getsize(args.output_path)
        report["keras_size_bytes"] = os.path.getsize(args.checkpoint_path)
        report["compression_ratio"] = report["keras_size_bytes"] / max(report["model_size_bytes"], 1)
        report["config"] = cfg
        with open(args.report_json, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Conversion report saved to {args.report_json}")


if __name__ == "__main__":
    main()
