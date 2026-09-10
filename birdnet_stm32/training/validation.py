"""Exact task validation and selection of the actual converted INT8 artifact."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

from birdnet_stm32.conversion.quantize import convert_to_tflite, stratified_sample_paths
from birdnet_stm32.evaluation.metrics import evaluate, macro_cmap
from birdnet_stm32.models.runners import KerasRunner, TFLiteRunner

# Seed of the validation subset draw. Fixed so every run, arm and epoch that asks
# for the same subset size scores the identical files.
VALIDATION_SUBSET_SEED = 1234


def stratified_validation_subset(val_paths: list[str], subset: int, *, seed: int = VALIDATION_SUBSET_SEED) -> list[str]:
    """Fixed, class-balanced draw of validation files for checkpoint selection.

    Classes are visited round-robin in a seeded order, so the draw is as even
    across classes as their folder sizes allow: a class with fewer files than
    its share contributes all of them and the remainder is spread over the
    rest. With 101 validation folders of which the smallest holds 13 files,
    ``subset=2513`` gives exactly 25 files per class and all 13 of the small
    one.

    Subset cMAP is biased upward -- fewer negatives per class to rank against --
    and is not comparable to full-manifest numbers. It preserves the gap between
    checkpoints well enough to choose between arms and coarse checkpoints;
    confirm a final selection on the full manifest.

    Args:
        val_paths: All validation file paths, one folder per class.
        subset: Files to draw. 0, or a value at least ``len(val_paths)``, keeps
            every file.
        seed: Draw seed.

    Returns:
        The selected paths.

    Raises:
        ValueError: If ``subset`` is too small to cover every folder.
        RuntimeError: If the draw fails to cover every folder.
    """
    if not subset or subset >= len(val_paths):
        return list(val_paths)
    folders = {os.path.basename(os.path.dirname(path)) for path in val_paths}
    if subset < len(folders):
        raise ValueError(f"validation subset {subset} cannot cover all {len(folders)} validation folders")
    chosen = stratified_sample_paths(val_paths, subset, seed=seed)
    if {os.path.basename(os.path.dirname(path)) for path in chosen} != folders:
        raise RuntimeError("Validation subset failed to cover every validation folder")
    return chosen


class ExactCmap(tf.keras.callbacks.Callback):
    """Exact chunk AP for library callers without a file validation manifest."""

    def __init__(self, dataset, steps):
        super().__init__()
        self.dataset, self.steps = dataset, steps

    def on_epoch_end(self, epoch, logs=None):
        labels, scores = [], []
        for batch in self.dataset.take(self.steps):
            x, y = batch[:2]
            labels.append(np.asarray(y))
            scores.append(np.asarray(self.model(x, training=False)))
        logs["val_cmap"] = macro_cmap(np.concatenate(labels), np.concatenate(scores))


class FileCmap(tf.keras.callbacks.Callback):
    """Use the CLI evaluator's windows, labels, pooling, and exact AP."""

    def __init__(self, files, classes, cfg, *, overlap=None, pooling="max", batch_size=64):
        super().__init__()
        self.files = sorted(files)
        self.classes = list(classes)
        self.cfg = cfg
        self.overlap = cfg["chunk_duration"] / 2 if overlap is None else overlap
        self.pooling, self.batch_size = pooling, batch_size
        if not self.files:
            raise ValueError("File validation requires a nonempty manifest")
        if not 0 <= self.overlap < cfg["chunk_duration"]:
            raise ValueError("Validation overlap must be >= 0 and less than chunk duration")

    def score(self, runner):
        metrics, records, labels, scores = evaluate(
            runner,
            self.files,
            self.classes,
            self.cfg,
            pooling=self.pooling,
            overlap=self.overlap,
            batch_size=self.batch_size,
        )
        if len(records) != len(self.files):
            raise RuntimeError("Validation skipped files; refusing to change the selection manifest")
        return macro_cmap(labels, scores)

    def on_epoch_end(self, epoch, logs=None):
        logs["val_cmap"] = self.score(KerasRunner(self.model))


class Int8Selection(FileCmap):
    """Keep matching Keras/TFLite files selected only by INT8 validation cMAP.

    Epoch zero is eligible, so fine-tuning cannot silently replace a better
    starting checkpoint. These are development artifacts, not release approval.
    The report is written last and records hashes of both selected files.
    """

    def __init__(self, deployment, teacher, calibration, checkpoint_path, sync, **kwargs):
        super().__init__(**kwargs)
        self.deployment, self.teacher = deployment, teacher
        self.calibration = calibration
        self.checkpoint_path = Path(checkpoint_path)
        self.int8_path = self.checkpoint_path.with_name(self.checkpoint_path.stem + "_INT8.tflite")
        self.report_path = self.checkpoint_path.with_name(self.checkpoint_path.stem + "_selection.json")
        self.sync = sync
        self.best = -float("inf")
        self.records = []
        self.float_reference = None
        calibration_hash = hashlib.sha256()
        for item in self.calibration:
            tensor = np.asarray(item[0], dtype=np.float32)
            calibration_hash.update(str(tensor.shape).encode())
            calibration_hash.update(tensor.tobytes())
        self.calibration_hash = calibration_hash.hexdigest()

    def on_train_begin(self, logs=None):
        self.float_reference = self.score(KerasRunner(self.teacher))
        self._evaluate(0, {})

    def on_epoch_end(self, epoch, logs=None):
        self.sync()
        self._evaluate(epoch + 1, logs)

    def _evaluate(self, epoch, logs):
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".int8-selection-", dir=self.checkpoint_path.parent) as directory:
            candidate = Path(directory) / "candidate.tflite"
            data = convert_to_tflite(
                self.deployment,
                lambda: iter(self.calibration),
                str(candidate),
            )
            int8_cmap = self.score(TFLiteRunner(str(candidate)))
            float_cmap = self.score(KerasRunner(self.deployment))
            if not np.isfinite(int8_cmap) or not np.isfinite(float_cmap):
                raise RuntimeError("Non-finite validation cMAP")
            logs.update(val_int8_cmap=int8_cmap, val_deployment_cmap=float_cmap)
            record = {
                "epoch": epoch,
                "int8_cmap": int8_cmap,
                "deployment_float_cmap": float_cmap,
                "drop_from_original_float": self.float_reference - int8_cmap,
            }
            self.records.append(record)
            if int8_cmap > self.best:
                temporary_checkpoint = Path(directory) / "candidate.keras"
                self.deployment.save(temporary_checkpoint)
                checkpoint_hash = hashlib.sha256(temporary_checkpoint.read_bytes()).hexdigest()
                os.replace(temporary_checkpoint, self.checkpoint_path)
                os.replace(candidate, self.int8_path)
                self.best = int8_cmap
                self.selected = {
                    **record,
                    "keras_sha256": checkpoint_hash,
                    "tflite_sha256": hashlib.sha256(data).hexdigest(),
                }
            report = {
                "status": "development_only",
                "metric": "exact_file_macro_average_precision",
                "selection_metric": "val_int8_cmap",
                "original_float_cmap": self.float_reference,
                "pooling": self.pooling,
                "overlap_seconds": self.overlap,
                "validation_files": len(self.files),
                "classes": self.classes,
                "validation_manifest_sha256": hashlib.sha256("\n".join(self.files).encode()).hexdigest(),
                "calibration_tensors": len(self.calibration),
                "calibration_sha256": self.calibration_hash,
                "selected": self.selected,
                "epochs": self.records,
            }
            temporary_report = Path(directory) / "selection.json"
            temporary_report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            os.replace(temporary_report, self.report_path)
        print(f"[INT8 validation] epoch={epoch} cMAP={int8_cmap:.6f} best={self.best:.6f}")
