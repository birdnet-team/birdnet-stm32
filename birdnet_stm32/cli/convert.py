"""CLI entry point for TFLite conversion."""

import argparse
import hashlib
import json
import os
import random
import tempfile
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from birdnet_stm32.audio.activity import pick_random_samples
from birdnet_stm32.conversion.quantize import (
    convert_to_tflite,
    representative_data_gen,
    stratified_sample_paths,
)
from birdnet_stm32.conversion.split import (
    count_parameters,
    embedding_dimension,
    find_embedding_layer,
    size_record,
    split_model,
)
from birdnet_stm32.conversion.validate import cosine_similarity, parity_metrics, validate_models
from birdnet_stm32.data.dataset import load_file_paths_from_directory
from birdnet_stm32.models.frontend import AudioFrontendLayer, hybrid_fft_bins, normalize_frontend_name
from birdnet_stm32.models.magnitude import MagnitudeScalingLayer
from birdnet_stm32.models.runners import ChainedTFLiteRunner, TFLiteRunner
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
        "--split_head",
        action="store_true",
        default=False,
        help=(
            "Also emit the model as a separate backbone and classifier head, so the head "
            "can be updated over a narrowband link without reflashing the backbone."
        ),
    )
    parser.add_argument(
        "--report_json",
        type=str,
        default="",
        help="Path to save a structured JSON conversion report.",
    )
    return parser.parse_args()


def _write_report(path: str, report: dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"Conversion report saved to {path}")


def _manifest_record(paths: list[str], root: str) -> dict:
    """Return a reproducible, compact identity for an audio path manifest."""
    relative_paths = [os.path.relpath(path, root).replace(os.sep, "/") for path in paths]
    payload = "\n".join(relative_paths).encode("utf-8")
    class_counts: dict[str, int] = defaultdict(int)
    for path in relative_paths:
        class_counts[path.split("/", 1)[0]] += 1
    return {
        "count": len(relative_paths),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "class_counts": dict(sorted(class_counts.items())),
    }


def _export_and_validate_onnx(model, output_path: str, validation_gen) -> dict:
    """Atomically export ONNX and require checker/runtime parity to pass."""
    try:
        import onnx
        import onnxruntime as ort
        import tf2onnx  # noqa: F401 - required by Keras' ONNX exporter
    except ImportError as exc:
        raise RuntimeError("ONNX export requires tf2onnx, onnx, and onnxruntime") from exc

    output_dir = os.path.dirname(os.path.abspath(output_path))
    with tempfile.NamedTemporaryFile(
        prefix=".exporting-onnx-",
        suffix=".onnx",
        dir=output_dir,
        delete=False,
    ) as handle:
        temporary_path = handle.name
    try:
        signature = [tf.TensorSpec(model.input_shape, tf.float32, name="input")]
        model.export(
            temporary_path,
            format="onnx",
            input_signature=signature,
            verbose=False,
        )
        onnx_model = onnx.load(temporary_path)
        onnx.checker.check_model(onnx_model, full_check=True)
        session = ort.InferenceSession(temporary_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        cosines: list[float] = []
        maximum_errors: list[float] = []
        squared_errors: list[float] = []
        for item in validation_gen():
            sample = np.asarray(item[0], dtype=np.float32)
            model_inputs = [sample] if isinstance(model._inputs_struct, (list, tuple)) else sample  # noqa: SLF001
            keras_output = np.asarray(model(model_inputs, training=False))
            onnx_output = np.asarray(session.run(None, {input_name: sample})[0])
            cosines.append(cosine_similarity(keras_output.ravel(), onnx_output.ravel()))
            maximum_errors.append(float(np.max(np.abs(keras_output - onnx_output))))
            squared_errors.append(float(np.mean((keras_output - onnx_output) ** 2)))
            if len(cosines) >= 16:
                break
        if not cosines:
            raise RuntimeError("ONNX validation generator yielded no samples")

        report = {
            "checker_passed": True,
            "runtime": "onnxruntime",
            "validation_samples": len(cosines),
            "cosine_mean": float(np.mean(cosines)),
            "cosine_min": float(np.min(cosines)),
            "max_abs_error": float(np.max(maximum_errors)),
            "mse_mean": float(np.mean(squared_errors)),
            "opset": max((item.version for item in onnx_model.opset_import), default=0),
        }
        if report["cosine_min"] < 0.9999 or report["max_abs_error"] > 1e-4:
            raise RuntimeError(f"ONNX runtime parity failed: {report}")
        os.replace(temporary_path, output_path)
        temporary_path = ""
        report["size_bytes"] = os.path.getsize(output_path)
        return report
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _convert_split_head(
    model,
    cfg: dict,
    args: argparse.Namespace,
    rep_data_gen,
    rep_data_gen_val,
) -> dict:
    """Convert and gate the backbone/classifier pair beside the whole model.

    The head is calibrated on embeddings produced by the *quantized* backbone,
    which is the exact input distribution it sees on the board. Both halves are
    staged and promoted together only after the chained pipeline clears the
    same parity gate the monolithic model had to clear.

    Args:
        model: Loaded monolithic Keras model.
        cfg: Model config dict.
        args: Parsed CLI arguments.
        rep_data_gen: Calibration data generator.
        rep_data_gen_val: Disjoint validation data generator.

    Returns:
        Report dict with parity metrics, sizes, and artifact paths.

    Raises:
        RuntimeError: If the chained pipeline fails the parity gate.
    """
    backbone, classifier = split_model(model)
    embedding_layer_name = find_embedding_layer(model).name
    base = os.path.splitext(args.output_path)[0]
    backbone_path = f"{base}_backbone.tflite"
    classifier_path = f"{base}_classifier.tflite"
    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    print(
        f"\nSplitting at '{embedding_layer_name}': backbone "
        f"{count_parameters(backbone):,} params -> {embedding_dimension(backbone)}-d embeddings, "
        f"classifier head {count_parameters(classifier):,} params"
    )

    staged: list[str] = []
    try:
        for _ in range(2):
            with tempfile.NamedTemporaryFile(
                prefix=".splitting-", suffix=".tflite", dir=output_dir, delete=False
            ) as handle:
                staged.append(handle.name)
        tmp_backbone, tmp_classifier = staged

        convert_to_tflite(
            backbone,
            rep_data_gen,
            tmp_backbone,
            quantization=args.quantization,
            per_tensor=args.per_tensor,
        )

        # Calibrate the head on what the quantized backbone actually emits, not
        # on float embeddings the board will never produce.
        backbone_runner = TFLiteRunner(tmp_backbone)
        embeddings = [backbone_runner.predict(np.asarray(sample[0], np.float32)) for sample in rep_data_gen()]
        if not embeddings:
            raise RuntimeError("Backbone produced no calibration embeddings for the classifier head")

        def head_rep_gen():
            for embedding in embeddings:
                yield [np.asarray(embedding, np.float32)]

        convert_to_tflite(
            classifier,
            head_rep_gen,
            tmp_classifier,
            quantization=args.quantization,
            per_tensor=args.per_tensor,
        )

        chained = ChainedTFLiteRunner(tmp_backbone, tmp_classifier)
        print("\n--- Backbone parity (Keras vs TFLite embeddings) ---")
        backbone_metrics = parity_metrics(
            lambda inputs: backbone(inputs, training=False).numpy(),
            backbone_runner.predict,
            rep_data_gen_val,
            label="backbone",
        )
        print("\n--- Chained parity (Keras whole model vs TFLite backbone + head) ---")
        chained_metrics = parity_metrics(
            lambda inputs: model(inputs, training=False).numpy(),
            chained.predict,
            rep_data_gen_val,
            label="chained",
        )

        failures = []
        if args.min_cosine_sim > 0 and chained_metrics["cosine_mean"] < args.min_cosine_sim:
            failures.append(f"chained mean cosine {chained_metrics['cosine_mean']:.6f} < {args.min_cosine_sim:.4f}")
        if args.min_cosine_p05 > 0 and chained_metrics["cosine_p05"] < args.min_cosine_p05:
            failures.append(f"chained p05 cosine {chained_metrics['cosine_p05']:.6f} < {args.min_cosine_p05:.4f}")
        if failures:
            raise RuntimeError("Split-model quality check failed: " + "; ".join(failures))

        os.replace(tmp_backbone, backbone_path)
        os.replace(tmp_classifier, classifier_path)
        staged = []
    finally:
        for path in staged:
            if os.path.exists(path):
                os.unlink(path)

    split_report = {
        "embedding_layer": embedding_layer_name,
        "embedding_dim": embedding_dimension(backbone),
        "backbone_params": count_parameters(backbone),
        "classifier_params": count_parameters(classifier),
        "backbone": size_record(backbone_path),
        "classifier": size_record(classifier_path),
        "backbone_validation": backbone_metrics,
        "chained_validation": chained_metrics,
        "quality_gate_passed": True,
    }

    # The head defines the class set, so it carries its own labels: an
    # over-the-air head update ships the species list with it.
    if cfg.get("class_names"):
        labels_path = os.path.splitext(classifier_path)[0] + "_labels.txt"
        with open(labels_path, "w", encoding="utf-8") as handle:
            handle.writelines(f"{name}\n" for name in cfg["class_names"])
        split_report["classifier_labels"] = os.path.basename(labels_path)

    head = split_report["classifier"]
    print(
        f"\nBackbone saved to {backbone_path} "
        f"({split_report['backbone']['size_bytes']:,} B, "
        f"{split_report['backbone']['gzip_bytes']:,} B gzipped)"
    )
    print(
        f"Classifier head saved to {classifier_path} "
        f"({head['size_bytes']:,} B, {head['gzip_bytes']:,} B gzipped, "
        f"{head['int8_sparsity']:.1%} of its INT8 weights are zero)"
    )
    return split_report


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
    data_manifests: dict[str, dict] = {}
    if os.path.isdir(args.data_path_train):
        configured_classes = cfg.get("class_names") or None
        file_paths, classes = load_file_paths_from_directory(args.data_path_train, classes=configured_classes)
        if not file_paths:
            raise ValueError("No training audio found for the classes in the model config.")

        class_count = len({os.path.basename(os.path.dirname(path)) for path in file_paths})
        stratified_paths = stratified_sample_paths(file_paths, args.num_samples, seed=42)
        if len(stratified_paths) != args.num_samples:
            raise ValueError(
                f"Requested {args.num_samples} calibration paths but only {len(stratified_paths)} are available."
            )
        print(f"Representative dataset: {len(stratified_paths)} stratified samples from {class_count} folders.")
        data_manifests["calibration"] = _manifest_record(stratified_paths, args.data_path_train)

        def rep_data_gen():
            return representative_data_gen(stratified_paths, cfg, num_samples=len(stratified_paths))

        # Calibration and validation must be disjoint; overlap makes parity
        # reports optimistic and invalidates a release gate.
        val_paths_subset = stratified_sample_paths(
            file_paths,
            args.validate_samples,
            seed=43,
            exclude=set(stratified_paths),
        )
        if not val_paths_subset:
            raise ValueError("No files remain for disjoint quantization validation.")
        if len(val_paths_subset) != args.validate_samples:
            raise ValueError(
                f"Requested {args.validate_samples} validation paths but only {len(val_paths_subset)} are available."
            )
        print(f"Validation dataset: {len(val_paths_subset)} disjoint stratified samples.")
        data_manifests["validation"] = _manifest_record(val_paths_subset, args.data_path_train)

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
        "data_manifests": data_manifests,
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

        # Backbone / classifier split. Opt-in: the firmware still runs a single
        # network, so the pair is produced only when it is asked for.
        if args.split_head:
            report["split"] = _convert_split_head(model, cfg, args, rep_data_gen, rep_data_gen_val)

        # ONNX export
        if args.export_onnx:
            onnx_path = os.path.splitext(args.output_path)[0] + ".onnx"
            report["onnx_validation"] = _export_and_validate_onnx(model, onnx_path, rep_data_gen_val)
            report["onnx_path"] = onnx_path
            print(f"ONNX model validated and saved to {onnx_path}")

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
