"""Device-facing quality gate for converted INT8 models.

The gate scores the deployed artifact without using a float model. It works per
file, the way a recording is judged in the field: every file is cut into
overlapping chunks, each chunk is scored by the model, and the chunk scores are
pooled into one score per class. A single random chunk is not a fair test of a
catalog recording, because many chunks of it hold no call at all; pooling asks
whether the model finds the species anywhere in the file.

Two views of the pooled scores are reported. At configured operating
thresholds: correct top-1 detections and confident false alarms, both across
files and equally across classes, plus the alarm rate on hard negatives. And
threshold-free, as a ranked species list: how often the labelled species is
within the top k. Directory labels are a stable release-to-release proxy, not a
substitute for annotated soundscapes.
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from birdnet_stm32.conversion.quantize import stratified_sample_paths
from birdnet_stm32.data.dataset import load_file_paths_from_directory
from birdnet_stm32.evaluation.metrics import make_chunks_for_file
from birdnet_stm32.evaluation.pooling import pool_scores
from birdnet_stm32.models.frontend import hybrid_fft_bins
from birdnet_stm32.models.runners import TFLiteRunner, runner_for_config
from birdnet_stm32.training.config import ModelConfig

DEFAULT_THRESHOLDS = (0.25, 0.5, 0.75)
# Match the catalog evaluation: max-pool chunks that overlap by half their
# length (2.5 s chunks every 1.25 s).
DEFAULT_POOLING = "max"
VALID_POOLING = ("avg", "max", "lme")
# make_chunks_for_file reads at most this much of each file.
MAX_FILE_SECONDS = 60
TOP_K = (1, 3, 5)
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
# Optional: {"5": 0.8} requires the labelled species in the top 5 for 80% of
# files, averaged equally across classes, in the worst draw.
OPTIONAL_TOP_K_LIMITS = "min_macro_top_k_rates"


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
    # The contract checks above read the interpreter, so wrap only afterwards.
    return runner_for_config(runner, cfg)


def measure_operational(
    model_path: str | Path,
    model_config: str | Path,
    data_path: str | Path,
    num_files: int,
    seed: int,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    batch_size: int = 16,
    pooling: str = DEFAULT_POOLING,
    chunk_overlap: float | None = None,
) -> dict[str, Any]:
    """Score one converted model on an exact, stratified draw of whole files.

    Each file is chunked like the catalog evaluation (first
    ``MAX_FILE_SECONDS``; ``chunk_overlap`` seconds of overlap, by default half
    the chunk), every chunk is scored in bounded batches, and the chunk scores
    are pooled per file.
    """
    if num_files <= 0:
        raise ValueError("num_files must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if pooling not in VALID_POOLING:
        raise ValueError(f"pooling must be one of {VALID_POOLING}, got {pooling!r}")
    thresholds = tuple(float(value) for value in thresholds)
    if not thresholds or any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("thresholds must contain values in [0, 1]")
    if len({_threshold_key(value) for value in thresholds}) != len(thresholds):
        raise ValueError("thresholds must not contain duplicates")

    config_path = Path(model_config)
    cfg = ModelConfig.load(config_path).to_dict()
    if chunk_overlap is None:
        chunk_overlap = float(cfg["chunk_duration"]) / 2
    if not 0 <= chunk_overlap < float(cfg["chunk_duration"]):
        raise ValueError(f"chunk_overlap must be in [0, {cfg['chunk_duration']}) seconds, got {chunk_overlap}")
    classes = list(cfg["class_names"])
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("Model config must define a nonempty, unique class order")
    runner = _load_and_validate_int8_model(model_path, cfg, classes)

    available_paths = load_file_paths_from_directory(str(data_path), classes=classes)[0]
    paths = stratified_sample_paths(available_paths, num_files, seed=seed)
    if len(paths) != num_files:
        raise ValueError(f"Requested {num_files} files but only {len(paths)} matching audio files are available")
    covered = {Path(path).parent.name for path in paths}
    missing = [name for name in classes if name not in covered]
    if missing:
        raise ValueError(f"Operational draw does not cover every output class; missing: {missing}")

    index = {name: position for position, name in enumerate(classes)}
    labels = np.asarray([index.get(Path(path).parent.name, -1) for path in paths], dtype=np.int64)
    if not np.any(labels < 0):
        raise ValueError("Operational draw contains no hard-negative noise/background files")

    expected = _expected_input_tail(cfg)
    input_hash = hashlib.sha256()
    chunk_scores: list[list[np.ndarray]] = [[] for _ in paths]
    pending: list[np.ndarray] = []
    owners: list[int] = []

    def flush() -> None:
        scores = np.asarray(runner.predict(np.stack(pending, axis=0)))
        if scores.shape != (len(pending), len(classes)):
            raise RuntimeError(f"Model returned scores with unexpected shape {scores.shape}")
        if not np.isfinite(scores).all():
            raise RuntimeError("Model returned non-finite scores; refusing to evaluate a broken artifact")
        for owner, row in zip(owners, scores, strict=True):
            chunk_scores[owner].append(row)
        pending.clear()
        owners.clear()

    total_chunks = 0
    for position, path in enumerate(paths):
        chunks = make_chunks_for_file(
            path, cfg, cfg["audio_frontend"], cfg.get("mag_scale", "none"), int(cfg["fft_length"]), chunk_overlap
        )
        if not chunks:
            relative = os.path.relpath(path, data_path)
            raise RuntimeError(f"No audio chunks from {relative}; the draw would not be comparable across models")
        for chunk in chunks:
            tensor = np.asarray(chunk, dtype=np.float32)
            if tensor.shape != expected:
                raise RuntimeError(f"Chunking returned unexpected input shape {tensor.shape}, expected {expected}")
            input_hash.update(str(tensor.shape).encode())
            input_hash.update(tensor.tobytes())
            pending.append(tensor)
            owners.append(position)
            total_chunks += 1
            if len(pending) == batch_size:
                flush()
    if pending:
        flush()

    scores = np.stack([pool_scores(np.stack(rows), pooling) for rows in chunk_scores])
    top1, peak = scores.argmax(1), scores.max(1)
    scored, negative = labels >= 0, labels < 0
    if not scored.any():
        raise RuntimeError("Draw contains no labelled files")

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
        "pooling": pooling,
        "chunk_overlap": float(chunk_overlap),
        "max_file_seconds": MAX_FILE_SECONDS,
        "files": int(len(labels)),
        "labelled_files": int(scored.sum()),
        "hard_negative_files": int(negative.sum()),
        "chunks": int(total_chunks),
        "thresholds": {},
        "ranking": ranking_metrics(labels, scores),
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


def ranking_metrics(labels: np.ndarray, scores: np.ndarray, top_k: Sequence[int] = TOP_K) -> dict[str, Any]:
    """Score the pooled outputs as a ranked species list, without a threshold.

    The rank of the labelled species counts every class scoring at least as
    high, so ties go against it: an INT8 output that saturates several classes
    at the same code has not ranked the right one first.

    Args:
        labels: [N] class index per file, negative for hard negatives (ignored).
        scores: [N, C] pooled scores.
        top_k: List lengths to report; values above C are dropped.

    Returns:
        Micro and macro top-k rates, mean reciprocal rank and median rank.
    """
    labels, scores = np.asarray(labels), np.asarray(scores)
    if labels.ndim != 1 or scores.ndim != 2 or scores.shape[0] != labels.shape[0]:
        raise ValueError("labels must be [N] and scores [N, C]")
    scored = labels >= 0
    if not scored.any():
        raise ValueError("at least one labelled sample is required")
    rows, targets = scores[scored], labels[scored]
    target_scores = rows[np.arange(len(targets)), targets]
    ranks = (rows >= target_scores[:, None]).sum(axis=1)
    ks = [int(k) for k in top_k if 0 < int(k) <= scores.shape[1]]
    per_class: dict[int, list[int]] = defaultdict(list)
    for target, rank in zip(targets, ranks, strict=True):
        per_class[int(target)].append(int(rank))
    return {
        "top_k_rates": {str(k): float(np.mean(ranks <= k)) for k in ks},
        "macro_top_k_rates": {
            str(k): float(np.mean([np.mean(np.asarray(r) <= k) for r in per_class.values()])) for k in ks
        },
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
    }


def _spread(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def summarize_draws(
    draws: list[dict[str, Any]], seeds: list[int], model_path: str | Path, data_path: str | Path, num_files: int
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
    methods = {(draw.get("pooling"), draw.get("chunk_overlap")) for draw in draws}
    if len(methods) != 1:
        raise RuntimeError("Draws do not share one pooling method and chunk overlap")
    actual_counts = {draw["files"] for draw in draws}
    if actual_counts != {num_files}:
        raise RuntimeError(f"Draw cardinality does not match requested count {num_files}: {sorted(actual_counts)}")
    pooling, chunk_overlap = methods.pop()
    summary: dict[str, Any] = {
        "model": Path(model_path).name,
        "model_sha256": digests.pop(),
        "data_root": Path(data_path).name,
        "num_files": num_files,
        "pooling": pooling,
        "chunk_overlap": chunk_overlap,
        "seeds": list(seeds),
        "draws": draws,
    }
    summary["model_config_sha256"] = config_digests.pop()
    if len(draws) > 1:
        summary["across_seeds"] = {
            key: {
                metric: _spread([draw["thresholds"][key][metric] for draw in draws])
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
        if all("ranking" in draw for draw in draws):
            ranking: dict[str, Any] = {
                "mean_reciprocal_rank": _spread([draw["ranking"]["mean_reciprocal_rank"] for draw in draws])
            }
            for family in ("top_k_rates", "macro_top_k_rates"):
                ranking[family] = {
                    k: _spread([draw["ranking"][family][k] for draw in draws]) for k in draws[0]["ranking"][family]
                }
            summary["across_seeds_ranking"] = ranking
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
    top_k_limits = {str(k): float(v) for k, v in profile.get(OPTIONAL_TOP_K_LIMITS, {}).items()}
    for k, value in top_k_limits.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{OPTIONAL_TOP_K_LIMITS}[{k}] must be in [0, 1]")
        if any(k not in draw.get("ranking", {}).get("macro_top_k_rates", {}) for draw in draws):
            raise ValueError(f"Operational results do not report a top-{k} rate")

    observed: dict[str, Any] = {
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
    if top_k_limits:
        observed["macro_top_k_rates"] = {}
        for k, limit in sorted(top_k_limits.items(), key=lambda item: int(item[0])):
            actual = min(draw["ranking"]["macro_top_k_rates"][k] for draw in draws)
            observed["macro_top_k_rates"][k] = actual
            if actual < limit:
                failures.append(f"macro_top_{k}_rate {actual:.6f} must be >= {limit:.6f}")
    limits: dict[str, Any] = {name: float(profile[name]) for name in sorted(REQUIRED_GATE_LIMITS)}
    if top_k_limits:
        limits[OPTIONAL_TOP_K_LIMITS] = top_k_limits
    return {
        "passed": not failures,
        "threshold": threshold,
        "limits": limits,
        "worst_across_seeds": observed,
        "failures": failures,
    }
