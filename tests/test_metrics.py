"""Unit tests for evaluation metrics."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow required for metrics tests")

from birdnet_stm32.evaluation.metrics import evaluate


class FakeRunner:
    """Fake model runner that returns fixed predictions."""

    def __init__(self, scores: np.ndarray):
        self.scores = scores
        self._idx = 0

    def predict(self, x_batch: np.ndarray) -> np.ndarray:
        batch_size = x_batch.shape[0]
        out = np.tile(self.scores[self._idx % len(self.scores)], (batch_size, 1))
        self._idx += batch_size
        return out.astype(np.float32)


class TestEvaluateMetrics:
    """Tests for the evaluate function using synthetic data."""

    @pytest.fixture
    def eval_setup(self, tmp_path):
        """Create a minimal dataset and config for evaluation."""
        import soundfile as sf

        sr = 22050
        duration = 3.0
        classes = ["bird_a", "bird_b"]
        files = []
        for cls in classes:
            cls_dir = tmp_path / cls
            cls_dir.mkdir()
            for i in range(3):
                path = cls_dir / f"sample_{i}.wav"
                audio = np.random.default_rng(i).standard_normal(int(sr * duration)).astype(np.float32)
                sf.write(str(path), audio, sr)
                files.append(str(path))

        cfg = {
            "sample_rate": sr,
            "chunk_duration": int(duration),
            "num_mels": 64,
            "spec_width": 256,
            "fft_length": 512,
            "audio_frontend": "precomputed",
            "mag_scale": "none",
        }
        return files, classes, cfg

    def test_perfect_predictions(self, eval_setup):
        """Perfect predictions should yield high metrics."""
        files, classes, cfg = eval_setup
        # bird_a files get [1,0], bird_b files get [0,1]
        scores = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=np.float32)
        runner = FakeRunner(scores)
        metrics, per_file, y_true, y_scores = evaluate(
            runner,
            files,
            classes,
            cfg,
            pooling="avg",
            batch_size=64,
        )
        assert "roc-auc" in metrics
        assert "f1" in metrics
        assert "cmAP" in metrics
        assert metrics["f1"] > 0.0
        assert len(per_file) == len(files)
        assert y_true.shape[0] == len(files)
        assert y_scores.shape[0] == len(files)

    def test_metrics_keys(self, eval_setup):
        """All expected metric keys should be present."""
        files, classes, cfg = eval_setup
        scores = np.array([[0.5, 0.5]], dtype=np.float32)
        runner = FakeRunner(scores)
        metrics, _, _, _ = evaluate(runner, files, classes, cfg, pooling="avg")
        for key in ("roc-auc", "f1", "precision", "recall", "cmAP", "mAP", "ap_per_class"):
            assert key in metrics, f"Missing metric key: {key}"

    def test_no_valid_files_raises(self, tmp_path):
        """Evaluate with no matching files should raise RuntimeError."""
        cfg = {
            "sample_rate": 22050,
            "chunk_duration": 3,
            "num_mels": 64,
            "spec_width": 256,
            "fft_length": 512,
            "audio_frontend": "precomputed",
            "mag_scale": "none",
        }
        runner = FakeRunner(np.array([[0.5, 0.5]]))
        with pytest.raises(RuntimeError, match="No valid test samples"):
            evaluate(runner, [], ["a", "b"], cfg, pooling="avg")
