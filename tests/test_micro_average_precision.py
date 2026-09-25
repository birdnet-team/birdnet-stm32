"""Tests for the pooled precision-recall metric used on field recordings.

`window_cmap_present` averages a PR curve per species, so a bird heard twice
weighs as much as one heard four hundred times. `micro_average_precision` pools
every window-class decision into one curve, which is closer to what a deployed
recorder produces.
"""

import numpy as np
import pytest

from birdnet_stm32.evaluation.metrics import class_average_precision, micro_average_precision


class TestMicroAveragePrecision:
    def test_a_perfect_ranking_scores_one(self):
        labels = np.array([[1, 0], [0, 1], [0, 0]])
        scores = np.array([[0.9, 0.1], [0.2, 0.8], [0.05, 0.05]])
        assert micro_average_precision(labels, scores) == pytest.approx(1.0)

    def test_it_matches_the_macro_metric_when_classes_are_identical(self):
        rng = np.random.default_rng(0)
        column_labels = (rng.random(200) > 0.7).astype(int)
        column_scores = rng.random(200)
        labels = np.stack([column_labels] * 3, axis=1)
        scores = np.stack([column_scores] * 3, axis=1)
        macro = float(np.mean(class_average_precision(labels, scores)))
        assert micro_average_precision(labels, scores) == pytest.approx(macro, abs=1e-6)

    def test_it_weights_the_common_class_where_macro_does_not(self):
        """One rare species scored perfectly, one common species scored badly."""
        rng = np.random.default_rng(1)
        rare = np.zeros(200, dtype=int)
        rare[:2] = 1
        rare_scores = np.where(rare == 1, 1.0, rng.random(200) * 0.1)
        common = (rng.random(200) > 0.5).astype(int)
        common_scores = rng.random(200)
        labels = np.stack([rare, common], axis=1)
        scores = np.stack([rare_scores, common_scores], axis=1)
        macro = float(np.mean(class_average_precision(labels, scores)))
        assert micro_average_precision(labels, scores) < macro

    def test_no_positives_anywhere_is_undefined_rather_than_zero(self):
        labels = np.zeros((5, 2), dtype=int)
        assert np.isnan(micro_average_precision(labels, np.random.default_rng(0).random((5, 2))))

    def test_it_refuses_the_same_malformed_input_the_macro_metric_does(self):
        with pytest.raises(ValueError, match="nonempty"):
            micro_average_precision(np.zeros((0, 2)), np.zeros((0, 2)))
        with pytest.raises(ValueError, match="finite"):
            micro_average_precision(np.array([[1, 0]]), np.array([[np.nan, 0.1]]))
        with pytest.raises(ValueError, match="binary"):
            micro_average_precision(np.array([[2, 0]]), np.array([[0.5, 0.1]]))
