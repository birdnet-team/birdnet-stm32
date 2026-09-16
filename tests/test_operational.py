"""The device-facing INT8 release gate.

This measurement decides whether a conversion ships, so the arithmetic is
pinned here rather than only exercised during a release.
"""

from pathlib import Path

import numpy as np
import pytest

from birdnet_stm32.evaluation.operational import (
    evaluate_release_gate,
    operating_point,
    ranking_metrics,
    summarize_draws,
)


class TestOperatingPoint:
    def test_counts_only_confident_predictions(self):
        """A correct prediction below threshold is not a detection.

        The device applies a threshold before it reports anything, so a chunk
        the model gets right but is unsure about is not a delivered detection.
        """
        labels = np.array([0, 0, 1, 1])
        top1 = np.array([0, 0, 1, 1])
        peak = np.array([0.9, 0.4, 0.8, 0.1])
        point = operating_point(labels, top1, peak, 0.5)
        assert point["detections"] == 2
        assert point["detection_rate"] == pytest.approx(0.5)
        assert point["false_alarms"] == 0

    def test_a_confident_wrong_prediction_is_a_false_alarm(self):
        labels = np.array([0, 1])
        top1 = np.array([1, 1])
        peak = np.array([0.9, 0.9])
        point = operating_point(labels, top1, peak, 0.5)
        assert point["detections"] == 1
        assert point["false_alarms"] == 1
        assert point["false_alarm_rate"] == pytest.approx(0.5)

    def test_macro_rate_gives_a_thin_class_equal_weight(self):
        """Micro rate is dominated by well-supplied classes; macro is not.

        The long tail is what a species-expansion release adds, so a metric
        that averages over chunks would let those classes fail unnoticed.
        """
        # Class 0 has eight chunks and all succeed; class 1 has one and fails.
        labels = np.array([0] * 8 + [1])
        top1 = np.array([0] * 8 + [0])
        peak = np.full(9, 0.9)
        point = operating_point(labels, top1, peak, 0.5)
        assert point["detection_rate"] == pytest.approx(8 / 9)
        assert point["macro_detection_rate"] == pytest.approx(0.5)
        assert point["classes_covered"] == 2

    def test_hard_negatives_are_excluded_from_the_denominators(self):
        """They have no correct class, so they can only raise false positives."""
        labels = np.array([0, 0, -1, -1])
        top1 = np.array([0, 0, 3, 3])
        peak = np.array([0.9, 0.9, 0.9, 0.1])
        point = operating_point(labels, top1, peak, 0.5)
        assert point["detection_rate"] == pytest.approx(1.0)
        assert point["false_alarm_rate"] == pytest.approx(0.0)
        assert point["negative_alarm_rate"] == pytest.approx(0.5)

    def test_negative_alarm_rate_is_absent_without_hard_negatives(self):
        point = operating_point(np.array([0, 1]), np.array([0, 1]), np.array([0.9, 0.9]), 0.5)
        assert point["negative_alarm_rate"] is None

    def test_a_uniformly_more_confident_model_cannot_win_on_detections_alone(self):
        """Why the alarm rate is gated alongside the detection rate."""
        labels = np.array([0, 1, 2, 3])
        top1 = np.array([0, 1, 0, 0])
        timid = operating_point(labels, top1, np.array([0.9, 0.9, 0.4, 0.4]), 0.5)
        brash = operating_point(labels, top1, np.array([0.9, 0.9, 0.9, 0.9]), 0.5)
        assert brash["detections"] == timid["detections"]
        assert brash["false_alarms"] > timid["false_alarms"]


class TestSummarizeDraws:
    @staticmethod
    def draw(sha, detection, seed):
        return {
            "model_sha256": sha,
            "model_config_sha256": "f" * 64,
            "seed": seed,
            "files": 10,
            "pooling": "max",
            "chunk_overlap": 1.25,
            "thresholds": {
                "0.5": {
                    "detection_rate": detection,
                    "false_alarm_rate": 0.1,
                    "macro_detection_rate": detection,
                    "macro_false_alarm_rate": 0.1,
                    "negative_alarm_rate": 0.0,
                }
            },
        }

    def test_reports_the_spread_a_floor_has_to_clear(self):
        draws = [self.draw("a" * 64, value, seed) for value, seed in zip((0.20, 0.30), (7, 11), strict=True)]
        summary = summarize_draws(draws, [7, 11], "m.tflite", "catalog_test", 10)
        spread = summary["across_seeds"]["0.5"]["macro_detection_rate"]
        assert spread["mean"] == pytest.approx(0.25)
        assert spread["min"] == pytest.approx(0.20)
        assert spread["max"] == pytest.approx(0.30)
        assert spread["std"] > 0

    def test_draws_of_different_artifacts_are_refused(self):
        """Averaging two models' scores would silently invent a third."""
        draws = [self.draw("a" * 64, 0.2, 7), self.draw("b" * 64, 0.3, 11)]
        with pytest.raises(RuntimeError, match="one model artifact"):
            summarize_draws(draws, [7, 11], "m.tflite", "catalog_test", 10)

    def test_draws_pooled_differently_are_refused(self):
        """Max- and average-pooled rates are different quantities."""
        other = self.draw("a" * 64, 0.3, 11)
        other["pooling"] = "avg"
        with pytest.raises(RuntimeError, match="pooling"):
            summarize_draws([self.draw("a" * 64, 0.2, 7), other], [7, 11], "m.tflite", "catalog_test", 10)

    def test_the_summary_carries_the_artifact_hash_and_seeds(self):
        """A release gate binds the report to the file and the pinned seeds."""
        summary = summarize_draws([self.draw("c" * 64, 0.2, 7)], [7], "m.tflite", "catalog_test", 10)
        assert summary["model_sha256"] == "c" * 64
        assert summary["seeds"] == [7]
        assert "across_seeds" not in summary


class TestReleaseGate:
    PROFILE = {
        "threshold": 0.5,
        "min_detection_rate": 0.2,
        "min_macro_detection_rate": 0.2,
        "max_false_alarm_rate": 0.2,
        "max_macro_false_alarm_rate": 0.2,
        "max_negative_alarm_rate": 0.2,
    }

    @staticmethod
    def summary(detection=0.3, false_alarm=0.1, negative_alarm=0.1):
        point = {
            "detection_rate": detection,
            "macro_detection_rate": detection,
            "false_alarm_rate": false_alarm,
            "macro_false_alarm_rate": false_alarm,
            "negative_alarm_rate": negative_alarm,
        }
        return {"draws": [{"thresholds": {"0.5": point}}]}

    def test_passes_only_when_every_limit_clears(self):
        assert evaluate_release_gate(self.summary(), self.PROFILE)["passed"] is True
        failed = evaluate_release_gate(self.summary(detection=0.1), self.PROFILE)
        assert failed["passed"] is False
        assert "detection_rate" in failed["failures"][0]

    def test_requires_hard_negatives(self):
        failed = evaluate_release_gate(self.summary(negative_alarm=None), self.PROFILE)
        assert failed["passed"] is False
        assert any("hard negatives" in reason for reason in failed["failures"])

    def test_requires_complete_profile(self):
        with pytest.raises(ValueError, match="missing required"):
            evaluate_release_gate(self.summary(), {"threshold": 0.5})

    def test_requires_an_explicit_operating_threshold(self):
        profile = {key: value for key, value in self.PROFILE.items() if key != "threshold"}
        with pytest.raises(ValueError, match="threshold"):
            evaluate_release_gate(self.summary(), profile)


def _fixture(tmp_path, folders, class_names):
    from birdnet_stm32.training.config import ModelConfig

    data = tmp_path / "test"
    for folder in folders:
        path = data / folder
        path.mkdir(parents=True)
        (path / "sample.wav").write_bytes(b"audio")
    config = tmp_path / "model_config.json"
    ModelConfig(
        sample_rate=8,
        num_mels=2,
        spec_width=2,
        fft_length=4,
        chunk_duration=1,
        audio_frontend="raw",
        num_classes=len(class_names),
        class_names=class_names,
    ).save(config)
    model = tmp_path / "model.tflite"
    model.write_bytes(b"model")
    return data, config, model


def test_measurement_streams_bounded_batches_and_hashes_contract(tmp_path, monkeypatch):
    from birdnet_stm32.evaluation import operational

    data, config, model = _fixture(tmp_path, ("a", "b", "background"), ["a", "b"])

    class Runner:
        batch_sizes = []

        def predict(self, batch):
            self.batch_sizes.append(len(batch))
            return np.tile([[0.9, 0.1]], (len(batch), 1))

    runner = Runner()
    monkeypatch.setattr(operational, "_load_and_validate_int8_model", lambda *args: runner)
    # Two chunks per file: batches of 2 over 3 files x 2 chunks.
    monkeypatch.setattr(
        operational, "make_chunks_for_file", lambda *args: [np.zeros((8, 1), np.float32) for _ in range(2)]
    )

    result = operational.measure_operational(model, config, data, 3, seed=7, batch_size=2)
    assert runner.batch_sizes == [2, 2, 2]
    assert result["files"] == 3
    assert result["chunks"] == 6
    assert result["pooling"] == "max"
    assert result["manifest"]["count"] == 3
    assert len(result["model_config_sha256"]) == 64
    assert len(result["input_tensors_sha256"]) == 64

    with pytest.raises(ValueError, match="only 3"):
        operational.measure_operational(model, config, data, 4, seed=7, batch_size=2)


def test_a_call_anywhere_in_the_file_counts(tmp_path, monkeypatch):
    """The point of file-level scoring: a call in the last chunk is found.

    Scored on its first chunk alone, the bird file below is a miss; pooled over
    the file, it is a confident, correct detection.
    """
    from birdnet_stm32.evaluation import operational

    data, config, model = _fixture(tmp_path, ("bird", "background"), ["bird"])
    chunks = {
        "bird": [np.full((8, 1), 0.0, np.float32), np.full((8, 1), 0.0, np.float32), np.full((8, 1), 1.0, np.float32)],
        "background": [np.full((8, 1), 0.0, np.float32)] * 3,
    }

    class Runner:
        @staticmethod
        def predict(batch):
            # Score = 0.9 on a chunk holding the call, 0.1 elsewhere.
            return np.where(batch.reshape(len(batch), -1)[:, :1] > 0.5, 0.9, 0.1).astype(np.float32)

    monkeypatch.setattr(operational, "_load_and_validate_int8_model", lambda *args: Runner())
    monkeypatch.setattr(operational, "make_chunks_for_file", lambda path, *args: chunks[Path(path).parent.name])

    result = operational.measure_operational(model, config, data, 2, seed=7, thresholds=(0.5,))
    assert result["thresholds"]["0.5"]["detection_rate"] == pytest.approx(1.0)
    assert result["thresholds"]["0.5"]["negative_alarm_rate"] == pytest.approx(0.0)
    assert result["ranking"]["top_k_rates"]["1"] == pytest.approx(1.0)

    average = operational.measure_operational(model, config, data, 2, seed=7, thresholds=(0.5,), pooling="avg")
    assert average["thresholds"]["0.5"]["detection_rate"] == pytest.approx(0.0)


def test_a_file_that_yields_no_audio_is_refused(tmp_path, monkeypatch):
    """Silently dropping it would make two models' draws incomparable."""
    from birdnet_stm32.evaluation import operational

    data, config, model = _fixture(tmp_path, ("bird", "background"), ["bird"])
    monkeypatch.setattr(operational, "_load_and_validate_int8_model", lambda *args: object())
    monkeypatch.setattr(operational, "make_chunks_for_file", lambda *args: [])
    with pytest.raises(RuntimeError, match="No audio chunks"):
        operational.measure_operational(model, config, data, 2, seed=7)


def test_measurement_rejects_non_finite_model_scores(tmp_path, monkeypatch):
    from birdnet_stm32.evaluation import operational

    data, config, model = _fixture(tmp_path, ("bird", "background"), ["bird"])

    class Runner:
        @staticmethod
        def predict(batch):
            return np.full((len(batch), 1), np.nan, dtype=np.float32)

    monkeypatch.setattr(operational, "_load_and_validate_int8_model", lambda *args: Runner())
    monkeypatch.setattr(operational, "make_chunks_for_file", lambda *args: [np.zeros((8, 1), np.float32)])

    with pytest.raises(RuntimeError, match="non-finite scores"):
        operational.measure_operational(model, config, data, 2, seed=7)


class TestRanking:
    def test_top_k_counts_the_target_anywhere_in_the_list(self):
        labels = np.array([0, 1, 2])
        scores = np.array(
            [
                [0.9, 0.1, 0.0, 0.0],  # rank 1
                [0.9, 0.5, 0.1, 0.0],  # rank 2
                [0.9, 0.8, 0.7, 0.1],  # rank 3
            ]
        )
        ranking = ranking_metrics(labels, scores, top_k=(1, 3, 5))
        assert ranking["top_k_rates"] == {"1": pytest.approx(1 / 3), "3": pytest.approx(1.0)}
        assert ranking["mean_reciprocal_rank"] == pytest.approx((1 + 1 / 2 + 1 / 3) / 3)
        assert ranking["median_rank"] == pytest.approx(2)

    def test_ties_go_against_the_target(self):
        """Two classes saturated at the same INT8 code: neither is ranked first."""
        ranking = ranking_metrics(np.array([0]), np.array([[1.0, 1.0, 0.0]]), top_k=(1, 2))
        assert ranking["top_k_rates"]["1"] == pytest.approx(0.0)
        assert ranking["top_k_rates"]["2"] == pytest.approx(1.0)

    def test_hard_negatives_are_not_ranked_and_macro_weighs_classes_equally(self):
        labels = np.array([0, 0, 0, 1, -1])
        scores = np.array([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1]])
        ranking = ranking_metrics(labels, scores, top_k=(1,))
        assert ranking["top_k_rates"]["1"] == pytest.approx(3 / 4)
        assert ranking["macro_top_k_rates"]["1"] == pytest.approx(0.5)


def test_gate_can_require_a_top_k_rate():
    point = {
        "detection_rate": 0.9,
        "macro_detection_rate": 0.9,
        "false_alarm_rate": 0.0,
        "macro_false_alarm_rate": 0.0,
        "negative_alarm_rate": 0.0,
    }
    summary = {"draws": [{"thresholds": {"0.5": point}, "ranking": {"macro_top_k_rates": {"5": 0.7}}}]}
    profile = {**TestReleaseGate.PROFILE, "min_macro_top_k_rates": {"5": 0.8}}
    gate = evaluate_release_gate(summary, profile)
    assert gate["passed"] is False
    assert gate["failures"] == ["macro_top_5_rate 0.700000 must be >= 0.800000"]
    assert evaluate_release_gate(summary, {**profile, "min_macro_top_k_rates": {"5": 0.6}})["passed"] is True
    with pytest.raises(ValueError, match="top-3"):
        evaluate_release_gate(summary, {**profile, "min_macro_top_k_rates": {"3": 0.6}})
