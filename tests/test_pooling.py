"""Unit tests for score pooling methods."""

import numpy as np
import pytest

from birdnet_stm32.evaluation.pooling import lme_pooling, pool_scores


class TestPoolScores:
    """Tests for pool_scores dispatcher."""

    def test_average(self):
        """Average pooling should compute the mean."""
        scores = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=np.float32)
        pooled = pool_scores(scores, method="avg")
        np.testing.assert_allclose(pooled, [0.4, 0.6])

    def test_max(self):
        """Max pooling should take element-wise max."""
        scores = np.array([[0.1, 0.9], [0.7, 0.3]], dtype=np.float32)
        pooled = pool_scores(scores, method="max")
        np.testing.assert_allclose(pooled, [0.7, 0.9])

    def test_empty(self):
        """Empty input should return zeros."""
        scores = np.zeros((0, 5), dtype=np.float32)
        pooled = pool_scores(scores, method="avg")
        np.testing.assert_array_equal(pooled, np.zeros(5))

    def test_invalid_method(self):
        """Invalid method should raise ValueError."""
        scores = np.ones((3, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="Unsupported"):
            pool_scores(scores, method="invalid")

    def test_wrong_ndim(self):
        """1-D input should raise ValueError."""
        with pytest.raises(ValueError, match="must be"):
            pool_scores(np.ones(5), method="avg")


class TestLmePooling:
    """Tests for log-mean-exponential pooling."""

    def test_single_row(self):
        """Single-row input should return the row itself."""
        scores = np.array([[0.5, 0.3]], dtype=np.float32)
        pooled = lme_pooling(scores, beta=10.0)
        np.testing.assert_allclose(pooled, [0.5, 0.3], atol=1e-5)

    def test_high_beta_approaches_max(self):
        """At high beta, LME should approximate max pooling."""
        scores = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
        pooled = lme_pooling(scores, beta=100.0)
        np.testing.assert_allclose(pooled, [0.8, 0.9], atol=0.05)
