"""Integration test: train → convert → evaluate end-to-end on synthetic data."""

import os

import numpy as np
import pytest
import soundfile as sf

tf = pytest.importorskip("tensorflow", reason="TensorFlow required for integration tests")


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Create a tiny synthetic dataset (2 classes × 5 files × 3s)."""
    sr = 22050
    duration = 3.0
    classes = ["class_a", "class_b"]

    for split in ("train", "test"):
        for cls in classes:
            cls_dir = tmp_path / split / cls
            cls_dir.mkdir(parents=True)
            for i in range(5):
                freq = 1000.0 if cls == "class_a" else 3000.0
                t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
                audio = 0.5 * np.sin(2 * np.pi * (freq + i * 100) * t)
                sf.write(str(cls_dir / f"sample_{i}.wav"), audio, sr)

    return str(tmp_path / "train"), str(tmp_path / "test"), classes


@pytest.mark.integration
@pytest.mark.slow
class TestTrainToEval:
    """End-to-end pipeline: train a tiny model → convert → evaluate."""

    def test_pipeline(self, synthetic_dataset, tmp_path):
        """Full pipeline should complete without errors and produce valid metrics."""
        from birdnet_stm32.audio.io import load_audio_file
        from birdnet_stm32.audio.spectrogram import get_spectrogram_from_audio
        from birdnet_stm32.data.dataset import load_file_paths_from_directory
        from birdnet_stm32.evaluation.metrics import evaluate
        from birdnet_stm32.models.dscnn import build_dscnn_model
        from birdnet_stm32.models.runners import TFLiteRunner

        train_dir, test_dir, classes = synthetic_dataset

        # ── 1. Build model ──────────────────────────────────────────────
        num_mels = 32
        spec_width = 64
        sr = 22050
        cd = 3

        model = build_dscnn_model(
            num_mels=num_mels,
            spec_width=spec_width,
            sample_rate=sr,
            chunk_duration=cd,
            embeddings_size=64,
            num_classes=len(classes),
            audio_frontend="precomputed",
            alpha=0.25,
            depth_multiplier=1,
            mag_scale="none",
        )
        model.compile(optimizer="adam", loss="categorical_crossentropy")

        # ── 2. Quick train (2 epochs) ──────────────────────────────────
        train_paths, train_classes = load_file_paths_from_directory(train_dir, classes=classes)
        X_train, y_train = [], []
        for p in train_paths:
            chunks = load_audio_file(p, sample_rate=sr, chunk_duration=cd)
            if len(chunks) == 0:
                continue
            spec = get_spectrogram_from_audio(
                chunks[0], sample_rate=sr, n_fft=256, mel_bins=num_mels, spec_width=spec_width
            )
            X_train.append(spec[:, :, np.newaxis])
            label = os.path.basename(os.path.dirname(p))
            y = np.zeros(len(classes), dtype=np.float32)
            y[classes.index(label)] = 1.0
            y_train.append(y)

        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        model.fit(X_train, y_train, epochs=2, batch_size=4, verbose=0)

        # ── 3. Save and convert to TFLite ──────────────────────────────
        keras_path = str(tmp_path / "model.keras")
        model.save(keras_path)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        def rep_gen():
            for x in X_train[:5]:
                yield [x[np.newaxis]]

        converter.representative_dataset = rep_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        tflite_bytes = converter.convert()

        tflite_path = str(tmp_path / "model.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_bytes)

        # ── 4. Evaluate with TFLite ────────────────────────────────────
        test_paths, _ = load_file_paths_from_directory(test_dir, classes=classes)
        cfg = {
            "sample_rate": sr,
            "chunk_duration": cd,
            "num_mels": num_mels,
            "spec_width": spec_width,
            "fft_length": 256,
            "audio_frontend": "precomputed",
            "mag_scale": "none",
        }
        runner = TFLiteRunner(tflite_path)
        metrics, per_file, y_true, y_scores = evaluate(
            runner,
            test_paths,
            classes,
            cfg,
            pooling="avg",
        )

        # Verify all metric keys present and values are valid
        assert "roc-auc" in metrics
        assert "f1" in metrics
        assert "cmAP" in metrics
        assert y_true.shape[0] > 0
        assert y_scores.shape == y_true.shape
        # Even a tiny model should produce some valid (not NaN) metrics
        assert np.isfinite(metrics["cmAP"])
