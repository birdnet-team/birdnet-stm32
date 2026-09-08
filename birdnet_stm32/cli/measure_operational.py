"""CLI for the device-facing INT8 release gate."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from birdnet_stm32.evaluation.operational import (
    DEFAULT_THRESHOLDS,
    evaluate_release_gate,
    measure_operational,
    summarize_draws,
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, required=True, help="Converted full-INT8 .tflite to gate")
    parser.add_argument("--model_config", type=Path, required=True)
    parser.add_argument("--data_path_test", type=Path, required=True, help="Class-directory root to draw chunks from")
    parser.add_argument("--num_chunks", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=16, help="Bounded TFLite inference batch size")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[7, 11, 23],
        help="Pinned draw seeds; compared models must use the identical list",
    )
    parser.add_argument(
        "--gate_profile",
        type=Path,
        required=True,
        help="JSON file containing threshold and required detection/alarm limits",
    )
    parser.add_argument(
        "--report_json", type=Path, required=True, help="Write the complete, artifact-bound report here"
    )
    return parser.parse_args()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write a report beside its destination and promote it atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as handle:
            temporary_path = handle.name
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def main() -> None:
    args = get_args()
    profile_bytes = args.gate_profile.read_bytes()
    profile = json.loads(profile_bytes)
    if not isinstance(profile, dict):
        raise ValueError("Gate profile must be a JSON object")
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("--seeds must contain unique values")
    if "threshold" not in profile:
        raise ValueError("Gate profile is missing required field: threshold")
    gate_threshold = float(profile["threshold"])
    thresholds = tuple(sorted({*DEFAULT_THRESHOLDS, gate_threshold}))
    draws = [
        measure_operational(
            args.model_path,
            args.model_config,
            args.data_path_test,
            args.num_chunks,
            seed,
            thresholds=thresholds,
            batch_size=args.batch_size,
        )
        for seed in args.seeds
    ]
    summary = summarize_draws(draws, args.seeds, args.model_path, args.data_path_test, args.num_chunks)
    summary["gate_profile"] = args.gate_profile.name
    summary["gate_profile_sha256"] = hashlib.sha256(profile_bytes).hexdigest()
    summary["gate"] = evaluate_release_gate(summary, profile)
    _write_json_atomic(args.report_json, summary)
    print(f"Operational report written to {args.report_json}")
    print(json.dumps(summary["gate"], indent=2))
    if not summary["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
