"""Device-facing quality gate for converted INT8 models.

The gate scores the deployed artifact without using a float model. It measures
correct top-1 detections and confident false alarms at configured operating
thresholds, both across chunks and equally across classes. Directory labels are
a stable release-to-release proxy, not a substitute for annotated soundscapes.
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from birdnet_stm32.conversion.quantize import representative_data_gen, stratified_sample_paths
from birdnet_stm32.data.dataset import load_file_paths_from_directory
from birdnet_stm32.models.frontend import hybrid_fft_bins
from birdnet_stm32.models.runners import TFLiteRunner
from birdnet_stm32.training.config import ModelConfig

DEFAULT_THRESHOLDS = (0.25, 0.5, 0.75)
REQUIRED_GATE_LIMITS = frozenset(
    {
        "min_detection_rate",
        "min_macro_detection_rate",
        "max_false_alarm_rate",
        "max_macro_false_alarm_rate",
        "max_negative_alarm_rate",
    }
)
REQUIRED_GATE_FIELDS = REQUIRED_GATE_LIMITS | {"threshold"}


def _threshold_key(value: float) -> str:
    """Return a stable, non-lossy JSON key for a confidence threshold."""
    normalized = 0.0 if value == 0 else float(value)
    return repr(normalized)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_record(paths: list[str], root: str | Path) -> dict[str, Any]:
    relative = [os.path.relpath(path, root).replace(os.sep, "/") for path in paths]
    counts: dict[str, int] = defaultdict(int)
    for path in relative:
        counts[path.split("/", 1)[0]] += 1
    return {
        "count": len(relative),
        "sha256": hashlib.sha256("\n".join(relative).encode()).hexdigest(),
        "class_counts": dict(sorted(counts.items())),
    }


def _expected_input_tail(cfg: dict[str, Any]) -> tuple[int, ...]:
    frontend = cfg["audio_frontend"]
    if frontend == "raw":
        return (int(cfg["sample_rate"] * cfg["chunk_duration"]), 1)
    if frontend == "hybrid":
        return (hybrid_fft_bins(int(cfg["fft_length"])), int(cfg["spec_width"]), 1)
    return (int(cfg["num_mels"]), int(cfg["spec_width"]), 1)


def _load_and_validate_int8_model(model_path: str | Path, cfg: dict[str, Any], classes: list[str]) -> TFLiteRunner:
    """Load a full-INT8 TFLite artifact and validate its public contract."""
    path = Path(model_path)
    if path.suffix.lower() != ".tflite":
        raise ValueError(f"Operational gating requires a .tflite artifact, got: {path.name}")
    runner = TFLiteRunner(str(path))
    interpreter = runner.interpreter
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_shape = tuple(int(value) for value in input_detail.get("shape_signature", input_detail["shape"]))
    output_shape = tuple(int(value) for value in output_detail.get("shape_signature", output_detail["shape"]))
    expected_tail = _expected_input_tail(cfg)

    if np.dtype(input_detail["dtype"]) != np.dtype(np.float32) or np.dtype(output_detail["dtype"]) != np.dtype(
        np.float32
    ):
        raise ValueError("Release INT8 models must retain float32 audio input and score output")
    if input_shape[1:] != expected_tail:
        raise ValueError(f"Model input {input_shape[1:]} does not match config {expected_tail}")
    if len(output_shape) != 2 or output_shape[-1] != len(classes):
        raise ValueError(
            f"Model exposes {output_shape[-1] if output_shape else 0} outputs but config defines {len(classes)} classes"
        )

    int8_activations = [
        detail
        for detail in interpreter.get_tensor_details()
        if np.dtype(detail["dtype"]) == np.dtype(np.int8)
        and len(detail.get("shape_signature", ())) > 0
        and int(detail["shape_signature"][0]) == -1
        and np.asarray(detail["quantization_parameters"]["scales"]).size > 0
    ]
    if not int8_activations:
        raise ValueError(
            "TFLite artifact has no dynamic-batch INT8 activations; it is not a full-INT8 deployment model"
        )
    return runner


def measure_operational(
    model_path: str | Path,
    model_config: str | Path,
    data_path: str | Path,
    num_chunks: int,
    seed: int,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Score one converted model on an exact, stratified chunk draw."""
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    thresholds = tuple(float(value) for value in thresholds)
    if not thresholds or any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("thresholds must contain values in [0, 1]")
    if len({_threshold_key(value) for value in thresholds}) != len(thresholds):
        raise ValueError("thresholds must not contain duplicates")

    config_path = Path(model_config)
    cfg = ModelConfig.load(config_path).to_dict()
    classes = list(cfg["class_names"])
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("Model config must define a nonempty, unique class order")
    runner = _load_and_validate_int8_model(model_path, cfg, classes)

    available_paths = load_file_paths_from_directory(str(data_path), classes=classes)[0]
    paths = stratified_sample_paths(available_paths, num_chunks, seed=seed)
    if len(paths) != num_chunks:
        raise ValueError(f"Requested {num_chunks} chunks but only {len(paths)} matching audio files are available")
    covered = {Path(path).parent.name for path in paths}
    missing = [name for name in classes if name not in covered]
    if missing:
        raise ValueError(f"Operational draw does not cover every output class; missing: {missing}")

    index = {name: position for position, name in enumerate(classes)}
    labels = np.asarray([index.get(Path(path).parent.name, -1) for path in paths], dtype=np.int64)
    if not np.any(labels < 0):
        raise ValueError("Operational draw contains no hard-negative noise/background chunks")
    input_hash = hashlib.sha256()
    score_batches: list[np.ndarray] = []
    pending: list[np.ndarray] = []
    generated = 0
    for sample in representative_data_gen(paths, cfg, num_samples=num_chunks):
        tensor = np.asarray(sample[0], dtype=np.float32)
        if tensor.shape != (1, *_expected_input_tail(cfg)):
            raise RuntimeError(f"Chunk generator returned unexpected input shape {tensor.shape}")
        input_hash.update(str(tensor.shape).encode())
        input_hash.update(tensor.tobytes())
        pending.append(tensor)
        generated += tensor.shape[0]
        if len(pending) == batch_size:
            score_batches.append(np.asarray(runner.predict(np.concatenate(pending, axis=0))))
            pending.clear()
    if pending:
        score_batches.append(np.asarray(runner.predict(np.concatenate(pending, axis=0))))
    if generated != len(paths):
        raise RuntimeError("Chunk generation skipped files; the draw would not be comparable across models")

    scores = np.concatenate(score_batches, axis=0)
    if scores.shape != (len(labels), len(classes)):
        raise RuntimeError(f"Model returned scores with unexpected shape {scores.shape}")
    if not np.isfinite(scores).all():
        raise RuntimeError("Model returned non-finite scores; refusing to evaluate a broken artifact")
    top1, peak = scores.argmax(1), scores.max(1)
    scored, negative = labels >= 0, labels < 0
    if not scored.any():
        raise RuntimeError("Draw contains no labelled chunks")

    result: dict[str, Any] = {
        "model": Path(model_path).name,
        "model_sha256": _sha256_file(model_path),
        "model_config": config_path.name,
        "model_config_sha256": _sha256_file(config_path),
        "class_order_sha256": hashlib.sha256(("\n".join(classes) + "\n").encode()).hexdigest(),
        "data_root": Path(data_path).name,
        "manifest": _manifest_record(paths, data_path),
        "input_tensors_sha256": input_hash.hexdigest(),
        "seed": seed,
        "chunks": int(len(labels)),
        "labelled_chunks": int(scored.sum()),
        "hard_negative_chunks": int(negative.sum()),
        "thresholds": {},
    }
    threshold_results: dict[str, Any] = result["thresholds"]
    for threshold in thresholds:
        threshold_results[_threshold_key(threshold)] = operating_point(labels, top1, peak, threshold)
    return result


def operating_point(labels: np.ndarray, top1: np.ndarray, peak: np.ndarray, threshold: float) -> dict[str, Any]:
    """Return detection and false-alarm rates at one confidence threshold."""
    labels, top1, peak = np.asarray(labels), np.asarray(top1), np.asarray(peak)
    if labels.ndim != 1 or labels.shape != top1.shape or labels.shape != peak.shape or not labels.size:
        raise ValueError("labels, top1, and peak must be nonempty, matching vectors")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    scored, negative = labels >= 0, labels < 0
    if not scored.any():
        raise ValueError("at least one labelled sample is required")
    confident = peak >= threshold
    hit = scored & confident & (top1 == labels)
    alarm = scored & confident & (top1 != labels)
    per_class_hit: dict[int, list[bool]] = defaultdict(list)
    per_class_alarm: dict[int, list[bool]] = defaultdict(list)
    for position in np.flatnonzero(scored):
        per_class_hit[int(labels[position])].append(bool(hit[position]))
        per_class_alarm[int(labels[position])].append(bool(alarm[position]))
    return {
        "detection_rate": float(hit.sum() / scored.sum()),
        "false_alarm_rate": float(alarm.sum() / scored.sum()),
        "macro_detection_rate": float(np.mean([np.mean(values) for values in per_class_hit.values()])),
        "macro_false_alarm_rate": float(np.mean([np.mean(values) for values in per_class_alarm.values()])),
        "negative_alarm_rate": (float((negative & confident).sum() / negative.sum()) if negative.any() else None),
        "detections": int(hit.sum()),
        "false_alarms": int(alarm.sum()),
        "classes_covered": len(per_class_hit),
    }


def summarize_draws(
    draws: list[dict[str, Any]], seeds: list[int], model_path: str | Path, data_path: str | Path, num_chunks: int
) -> dict[str, Any]:
    """Combine per-seed draws and report the spread a release floor must clear."""
    if not draws or len(draws) != len(seeds):
        raise ValueError("Exactly one completed draw is required for every seed")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be unique")
    if [draw.get("seed") for draw in draws] != seeds:
        raise RuntimeError("Draw seeds do not match the requested seed order")
    digests = {draw["model_sha256"] for draw in draws}
    config_digests = {draw.get("model_config_sha256") for draw in draws}
    if len(digests) != 1:
        raise RuntimeError("Draws do not share one model artifact")
    if None in config_digests or len(config_digests) != 1:
        raise RuntimeError("Draws do not share one model configuration")
    threshold_sets = {tuple(draw["thresholds"]) for draw in draws}
    if len(threshold_sets) != 1:
        raise RuntimeError("Draws do not share one threshold set")
    actual_counts = {draw["chunks"] for draw in draws}
    if actual_counts != {num_chunks}:
        raise RuntimeError(f"Draw cardinality does not match requested count {num_chunks}: {sorted(actual_counts)}")
    summary: dict[str, Any] = {
        "model": Path(model_path).name,
        "model_sha256": digests.pop(),
        "data_root": Path(data_path).name,
        "num_chunks": num_chunks,
        "seeds": list(seeds),
        "draws": draws,
    }
    summary["model_config_sha256"] = config_digests.pop()
    if len(draws) > 1:
        summary["across_seeds"] = {
            key: {
                metric: {
                    "mean": float(np.mean([draw["thresholds"][key][metric] for draw in draws])),
                    "std": float(np.std([draw["thresholds"][key][metric] for draw in draws], ddof=1)),
                    "min": float(np.min([draw["thresholds"][key][metric] for draw in draws])),
                    "max": float(np.max([draw["thresholds"][key][metric] for draw in draws])),
                }
                for metric in (
                    "detection_rate",
                    "false_alarm_rate",
                    "macro_detection_rate",
                    "macro_false_alarm_rate",
                    "negative_alarm_rate",
                )
                if all(draw["thresholds"][key][metric] is not None for draw in draws)
            }
            for key in draws[0]["thresholds"]
        }
    return summary


def evaluate_release_gate(summary: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Apply explicit release limits to the worst draw at one operating point."""
    missing = sorted(REQUIRED_GATE_FIELDS - profile.keys())
    if missing:
        raise ValueError(f"Gate profile is missing required limits: {missing}")
    threshold = float(profile["threshold"])
    if not 0 <= threshold <= 1:
        raise ValueError("Gate threshold must be in [0, 1]")
    key = _threshold_key(threshold)
    draws = summary.get("draws", [])
    if not draws or any(key not in draw["thresholds"] for draw in draws):
        raise ValueError(f"Operational results do not contain gate threshold {key}")

    for name in REQUIRED_GATE_LIMITS:
        value = float(profile[name])
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")

    observed: dict[str, float | None] = {
        "detection_rate": min(draw["thresholds"][key]["detection_rate"] for draw in draws),
        "macro_detection_rate": min(draw["thresholds"][key]["macro_detection_rate"] for draw in draws),
        "false_alarm_rate": max(draw["thresholds"][key]["false_alarm_rate"] for draw in draws),
        "macro_false_alarm_rate": max(draw["thresholds"][key]["macro_false_alarm_rate"] for draw in draws),
        "negative_alarm_rate": (
            max(draw["thresholds"][key]["negative_alarm_rate"] for draw in draws)
            if all(draw["thresholds"][key]["negative_alarm_rate"] is not None for draw in draws)
            else None
        ),
    }
    failures: list[str] = []
    comparisons = (
        ("detection_rate", "min_detection_rate", ">="),
        ("macro_detection_rate", "min_macro_detection_rate", ">="),
        ("false_alarm_rate", "max_false_alarm_rate", "<="),
        ("macro_false_alarm_rate", "max_macro_false_alarm_rate", "<="),
        ("negative_alarm_rate", "max_negative_alarm_rate", "<="),
    )
    for metric, limit_name, direction in comparisons:
        actual = observed[metric]
        limit = float(profile[limit_name])
        if actual is None:
            failures.append(f"{metric} unavailable: the draw contains no hard negatives")
        elif (direction == ">=" and actual < limit) or (direction == "<=" and actual > limit):
            failures.append(f"{metric} {actual:.6f} must be {direction} {limit:.6f}")
    return {
        "passed": not failures,
        "threshold": threshold,
        "limits": {name: float(profile[name]) for name in sorted(REQUIRED_GATE_LIMITS)},
        "worst_across_seeds": observed,
        "failures": failures,
    }
