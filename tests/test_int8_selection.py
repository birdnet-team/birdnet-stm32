"""Regressions for selecting deployed task quality rather than QAT proxies."""

import hashlib
import json

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from birdnet_stm32.evaluation.metrics import macro_cmap
from birdnet_stm32.training.trainer import monitor_mode, train_model
from birdnet_stm32.training.validation import Int8Selection


def test_exact_ap_keeps_low_probability_ranking_and_full_class_contract():
    labels = np.array([[1, 0, 1], [0, 0, 1]])
    scores = np.array([[0.0002, 0, 0.1], [0.0001, 0, 0.2]])
    assert macro_cmap(labels, scores) == pytest.approx(2 / 3)
    assert monitor_mode("val_int8_cmap") == "max"
    with pytest.raises(ValueError, match="finite"):
        macro_cmap(labels, np.full((2, 3), np.nan))


@pytest.mark.parametrize("scores", [[0.8, 0.7, 0.6], [0.5, 0.8, 0.7]])
def test_selection_keeps_matching_best_artifact_including_epoch_zero(tmp_path, monkeypatch, scores):
    """Lower simulated loss or a later epoch must never overwrite better INT8."""
    from birdnet_stm32.training import validation

    model = tf.keras.Sequential([tf.keras.Input((2,)), tf.keras.layers.Dense(2, activation="sigmoid")])
    teacher = tf.keras.models.clone_model(model)
    data = [[np.zeros((1, 2), np.float32)]]
    path = tmp_path / "model_qat.keras"
    selector = Int8Selection(
        model,
        teacher,
        data,
        path,
        sync=lambda: None,
        files=["a.wav"],
        classes=["a", "b"],
        cfg={"chunk_duration": 2.5},
    )
    blobs = iter([b"epoch0", b"epoch1", b"epoch2"])

    def convert(model, rep, output):
        blob = next(blobs)
        from pathlib import Path

        Path(output).write_bytes(blob)
        return blob

    monkeypatch.setattr(validation, "convert_to_tflite", convert)
    monkeypatch.setattr(validation, "TFLiteRunner", lambda path: "int8")
    int8_scores = iter(scores)
    monkeypatch.setattr(selector, "score", lambda runner: next(int8_scores) if runner == "int8" else 0.9)
    selector.on_train_begin()
    for epoch in range(2):
        model.layers[0].bias.assign([epoch + 1.0, epoch + 1.0])
        logs = {"val_pr_auc": 0.99 - epoch * 0.01}
        selector.on_epoch_end(epoch, logs)
        assert logs["val_int8_cmap"] == scores[epoch + 1]
    selected_epoch = int(np.argmax(scores))
    report = json.loads(selector.report_path.read_text())
    assert report["selected"]["epoch"] == selected_epoch
    assert selector.int8_path.read_bytes() == f"epoch{selected_epoch}".encode()
    assert report["selected"]["keras_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    kept = tf.keras.models.load_model(path, compile=False)
    np.testing.assert_allclose(kept.layers[0].bias, [selected_epoch] * 2)
    assert report["selected"]["drop_from_original_float"] == pytest.approx(0.9 - max(scores))


def test_exact_metric_runs_before_checkpoint_selection(tmp_path, monkeypatch):
    """Exercise real callback ordering and shared checkpoint persistence."""
    from birdnet_stm32.training import trainer

    monkeypatch.setattr(trainer, "_save_training_curves", lambda *args: None)
    model = tf.keras.Sequential([tf.keras.Input((2,)), tf.keras.layers.Dense(2, activation="sigmoid")])
    samples = np.array([[1, 0], [0, 1]], np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((samples, samples)).batch(2).repeat()
    path = tmp_path / "best.keras"
    history = train_model(model, dataset, dataset, epochs=1, steps_per_epoch=1, val_steps=1, checkpoint_path=str(path))
    assert np.isfinite(history.history["val_cmap"][0])
    assert "val_pr_auc" in history.history
    assert path.exists()


@pytest.mark.integration
def test_qat_cli_saves_a_real_selected_int8_pair(tmp_path, monkeypatch):
    """Small synthetic audio run through calibration, clipping, QAT and export."""
    import soundfile as sf

    from birdnet_stm32.cli.train import get_args
    from birdnet_stm32.models.frontend import AudioFrontendLayer
    from birdnet_stm32.models.magnitude import MagnitudeScalingLayer
    from birdnet_stm32.training.config import ModelConfig
    from birdnet_stm32.training.qat import run_qat

    for split in ["train", "validation"]:
        for index, name in enumerate(["a", "b"]):
            folder = tmp_path / split / name
            folder.mkdir(parents=True)
            signal = np.sin(2 * np.pi * (400 + index * 600) * np.arange(2000) / 8000)
            sf.write(folder / "sample.wav", signal, 8000)
    inputs = tf.keras.Input((2000, 1))
    x = AudioFrontendLayer(
        mode="raw",
        mel_bins=8,
        spec_width=8,
        sample_rate=8000,
        chunk_duration=0.25,
        mag_scale="pwl",
        name="audio_frontend",
    )(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(2, activation="sigmoid", name="pred")(x))
    path = tmp_path / "model.keras"
    model.save(path)
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    ModelConfig(
        sample_rate=8000,
        num_mels=8,
        spec_width=8,
        chunk_duration=0.25,
        audio_frontend="raw",
        num_classes=2,
        class_names=["a", "b"],
    ).save(tmp_path / "model_model_config.json")
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--qat",
            "--checkpoint_path",
            str(path),
            "--data_path_train",
            str(tmp_path / "train"),
            "--data_path_val",
            str(tmp_path / "validation"),
            "--qat_calibration_samples",
            "2",
            "--qat_calibration_percentile",
            "99.9",
            "--epochs",
            "1",
            "--batch_size",
            "2",
            "--num_workers",
            "1",
            "--max_chunks_per_file",
            "1",
            "--prefetch_batches",
            "1",
        ],
    )
    run_qat(get_args())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source_hash
    report = json.loads((tmp_path / "model_qat_selection.json").read_text())
    assert len(report["epochs"]) == 2
    assert report["selected"]["int8_cmap"] == max(row["int8_cmap"] for row in report["epochs"])
    kept_path = tmp_path / "model_qat.keras"
    kept = tf.keras.models.load_model(
        kept_path,
        compile=False,
        custom_objects={
            "AudioFrontendLayer": AudioFrontendLayer,
            "MagnitudeScalingLayer": MagnitudeScalingLayer,
        },
    )
    assert kept.get_layer("audio_frontend").activation_bounds
    assert hashlib.sha256(kept_path.read_bytes()).hexdigest() == report["selected"]["keras_sha256"]
    assert (
        hashlib.sha256((tmp_path / "model_qat_INT8.tflite").read_bytes()).hexdigest()
        == report["selected"]["tflite_sha256"]
    )


def test_validation_subset_draw_is_fixed_and_covers_every_class():
    """Selection compares checkpoints, so the draw must never move.

    The subset exists because selection converts and scores an INT8 model every
    epoch; on the full manifest that dominates the run. A draw that changed
    between arms or epochs would make the comparison meaningless, and one that
    dropped classes would silently stop gating the long tail that macro AP
    exists to protect.
    """
    from birdnet_stm32.conversion.quantize import stratified_sample_paths

    paths = [f"/data/{name}/{index:04d}.wav" for name in (f"class{i:03d}" for i in range(40)) for index in range(25)]
    first = stratified_sample_paths(paths, 200, seed=1234)
    second = stratified_sample_paths(paths, 200, seed=1234)

    assert first == second
    assert len(first) == 200
    assert len({path.split("/")[2] for path in first}) == 40
    # A subset at least as large as the manifest must not silently shrink it.
    assert sorted(stratified_sample_paths(paths, len(paths), seed=1234)) == sorted(paths)


def test_validation_subset_defaults_to_the_whole_manifest():
    """Opting in is explicit; an unset flag must not quietly change selection."""
    import sys

    from birdnet_stm32.cli.train import get_args

    argv = sys.argv
    try:
        sys.argv = ["birdnet_stm32", "--data_path_train", "/tmp/train"]
        assert get_args().validation_subset == 0
    finally:
        sys.argv = argv
