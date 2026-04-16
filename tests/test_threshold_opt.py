"""Unit tests for per-class threshold optimization."""

import numpy as np

from birdnet_stm32.evaluation.metrics import optimize_thresholds


class TestOptimizeThresholds:
    """Tests for optimize_thresholds."""

    def test_perfect_separation(self):
        """Well-separated scores should yield high thresholds for positives."""
        y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
        y_scores = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]], dtype=np.float32)
        classes = ["a", "b"]
        result = optimize_thresholds(y_true, y_scores, classes)
        assert set(result.keys()) == {"a", "b"}
        assert 0.0 < result["a"] < 1.0
        assert 0.0 < result["b"] < 1.0

    def test_returns_default_for_empty_class(self):
        """Classes with no positive samples should get threshold 0.5."""
        y_true = np.array([[0, 1], [0, 1]], dtype=np.float32)
        y_scores = np.array([[0.3, 0.9], [0.2, 0.8]], dtype=np.float32)
        classes = ["empty_class", "present_class"]
        result = optimize_thresholds(y_true, y_scores, classes)
        assert result["empty_class"] == 0.5

    def test_all_classes_present(self):
        """All class names should appear in the result dict."""
        n_classes = 5
        classes = [f"cls_{i}" for i in range(n_classes)]
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=(20, n_classes)).astype(np.float32)
        y_scores = rng.uniform(0, 1, size=(20, n_classes)).astype(np.float32)
        result = optimize_thresholds(y_true, y_scores, classes)
        assert len(result) == n_classes
        for cls in classes:
            assert cls in result
            assert 0.0 <= result[cls] <= 1.0
