"""Unit tests for Phase 10 evaluation reporting features.

Tests bootstrap AP CI, DET curve, species report CSV, benchmark JSON,
DET plot, and HTML report generation.
"""

import json

import numpy as np
import pytest

from birdnet_stm32.evaluation.metrics import bootstrap_ap_ci, compute_det_curve
from birdnet_stm32.evaluation.reporting import (
    print_ascii_det_curve,
    save_benchmark_json,
    save_det_curve_plot,
    save_html_report,
    save_species_report_csv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_data():
    """Synthetic binary classification data (2 classes, 50 samples)."""
    rng = np.random.default_rng(0)
    n, c = 50, 2
    y_true = np.zeros((n, c), dtype=np.float32)
    for i in range(n):
        y_true[i, i % c] = 1.0
    y_scores = y_true * 0.8 + rng.uniform(0, 0.2, size=(n, c)).astype(np.float32)
    classes = ["bird_a", "bird_b"]
    return y_true, y_scores, classes


@pytest.fixture
def three_class_data():
    """Synthetic 3-class data with clear separability."""
    rng = np.random.default_rng(7)
    n, c = 60, 3
    y_true = np.zeros((n, c), dtype=np.float32)
    for i in range(n):
        y_true[i, i % c] = 1.0
    noise = rng.uniform(0, 0.15, size=(n, c)).astype(np.float32)
    y_scores = y_true * 0.85 + noise
    classes = ["species_x", "species_y", "species_z"]
    return y_true, y_scores, classes


# ---------------------------------------------------------------------------
# bootstrap_ap_ci
# ---------------------------------------------------------------------------

class TestBootstrapApCi:
    """Tests for bootstrap_ap_ci."""

    def test_output_structure(self, binary_data):
        y_true, y_scores, classes = binary_data
        result = bootstrap_ap_ci(y_true, y_scores, classes, n_bootstrap=100)
        assert len(result) == len(classes)
        for row in result:
            for key in ("class", "ap", "ci_lower", "ci_upper", "n_positive", "n_total"):
                assert key in row, f"Missing key: {key}"

    def test_ci_bounds_contain_point(self, binary_data):
        """CI lower <= AP <= CI upper for each class."""
        y_true, y_scores, classes = binary_data
        result = bootstrap_ap_ci(y_true, y_scores, classes, n_bootstrap=500)
        for row in result:
            assert row["ci_lower"] <= row["ap"] + 1e-6
            assert row["ci_upper"] >= row["ap"] - 1e-6

    def test_high_confidence_wider(self, binary_data):
        """99% CI should be at least as wide as 90% CI."""
        y_true, y_scores, classes = binary_data
        ci_90 = bootstrap_ap_ci(y_true, y_scores, classes, n_bootstrap=500, confidence=0.90)
        ci_99 = bootstrap_ap_ci(y_true, y_scores, classes, n_bootstrap=500, confidence=0.99)
        for r90, r99 in zip(ci_90, ci_99):
            w90 = r90["ci_upper"] - r90["ci_lower"]
            w99 = r99["ci_upper"] - r99["ci_lower"]
            assert w99 >= w90 - 1e-6

    def test_all_positive_degeneracy(self):
        """When all samples are positive for a class, CI collapses to point."""
        n, c = 10, 1
        y_true = np.ones((n, c), dtype=np.float32)
        y_scores = np.ones((n, c), dtype=np.float32) * 0.9
        result = bootstrap_ap_ci(y_true, y_scores, ["only_class"], n_bootstrap=100)
        assert len(result) == 1
        assert result[0]["ci_lower"] == result[0]["ci_upper"]

    def test_no_positive_degeneracy(self):
        """When there are no positives, CI collapses."""
        n, c = 10, 1
        y_true = np.zeros((n, c), dtype=np.float32)
        y_scores = np.random.default_rng(1).uniform(0, 1, (n, c)).astype(np.float32)
        result = bootstrap_ap_ci(y_true, y_scores, ["empty_class"], n_bootstrap=100)
        assert len(result) == 1
        assert result[0]["n_positive"] == 0

    def test_reproducibility(self, binary_data):
        """Same seed should produce identical results."""
        y_true, y_scores, classes = binary_data
        r1 = bootstrap_ap_ci(y_true, y_scores, classes, n_bootstrap=200, seed=123)
        r2 = bootstrap_ap_ci(y_true, y_scores, classes, n_bootstrap=200, seed=123)
        for a, b in zip(r1, r2):
            assert a["ci_lower"] == b["ci_lower"]
            assert a["ci_upper"] == b["ci_upper"]

    def test_three_classes(self, three_class_data):
        """Verify with 3 classes."""
        y_true, y_scores, classes = three_class_data
        result = bootstrap_ap_ci(y_true, y_scores, classes, n_bootstrap=200)
        assert len(result) == 3
        for row in result:
            assert -1e-9 <= row["ap"] <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# compute_det_curve
# ---------------------------------------------------------------------------

class TestComputeDetCurve:
    """Tests for compute_det_curve."""

    def test_output_shapes(self, binary_data):
        y_true, y_scores, _ = binary_data
        far, frr, thresholds = compute_det_curve(y_true, y_scores)
        assert far.shape == frr.shape == thresholds.shape
        assert len(far) > 0

    def test_far_frr_range(self, binary_data):
        y_true, y_scores, _ = binary_data
        far, frr, _ = compute_det_curve(y_true, y_scores)
        assert np.all(far >= 0) and np.all(far <= 1)
        assert np.all(frr >= 0) and np.all(frr <= 1)

    def test_perfect_predictions(self):
        """Perfect scores should have a point at (FAR=0, FRR=0)."""
        y_true = np.array([1, 1, 0, 0], dtype=np.float32)
        y_scores = np.array([0.9, 0.8, 0.1, 0.2], dtype=np.float32)
        far, frr, _ = compute_det_curve(y_true, y_scores)
        # At some threshold both FAR and FRR should be 0
        min_sum = np.min(far + frr)
        assert min_sum < 0.01

    def test_no_positive_degeneracy(self):
        """All negatives should return degenerate but valid arrays."""
        y_true = np.zeros(10, dtype=np.float32)
        y_scores = np.random.default_rng(2).uniform(0, 1, 10).astype(np.float32)
        far, frr, thresholds = compute_det_curve(y_true, y_scores)
        assert len(far) >= 1

    def test_no_negative_degeneracy(self):
        """All positives should return degenerate but valid arrays."""
        y_true = np.ones(10, dtype=np.float32)
        y_scores = np.random.default_rng(3).uniform(0, 1, 10).astype(np.float32)
        far, frr, thresholds = compute_det_curve(y_true, y_scores)
        assert len(far) >= 1


# ---------------------------------------------------------------------------
# save_species_report_csv
# ---------------------------------------------------------------------------

class TestSaveSpeciesReportCsv:
    """Tests for save_species_report_csv."""

    def test_csv_created(self, tmp_path, binary_data):
        y_true, y_scores, classes = binary_data
        species_data = bootstrap_ap_ci(y_true, y_scores, classes, n_bootstrap=50)
        out = str(tmp_path / "species.csv")
        save_species_report_csv(species_data, out)
        assert (tmp_path / "species.csv").exists()

    def test_csv_content(self, tmp_path, binary_data):
        y_true, y_scores, classes = binary_data
        species_data = bootstrap_ap_ci(y_true, y_scores, classes, n_bootstrap=50)
        out = str(tmp_path / "species.csv")
        save_species_report_csv(species_data, out)
        with open(out) as f:
            lines = f.readlines()
        assert lines[0].strip() == "class,ap,ci_lower,ci_upper,n_positive,n_total"
        assert len(lines) == len(classes) + 1  # header + data rows

    def test_csv_sorted_descending(self, tmp_path, three_class_data):
        y_true, y_scores, classes = three_class_data
        species_data = bootstrap_ap_ci(y_true, y_scores, classes, n_bootstrap=50)
        out = str(tmp_path / "species.csv")
        save_species_report_csv(species_data, out)
        with open(out) as f:
            lines = f.readlines()[1:]  # skip header
        aps = [float(line.split(",")[1]) for line in lines]
        assert aps == sorted(aps, reverse=True)


# ---------------------------------------------------------------------------
# save_benchmark_json
# ---------------------------------------------------------------------------

class TestSaveBenchmarkJson:
    """Tests for save_benchmark_json."""

    def test_json_created(self, tmp_path):
        metrics = {"roc-auc": 0.95, "cmAP": 0.88, "f1": 0.80, "total_chunks": 100}
        out = str(tmp_path / "benchmark.json")
        save_benchmark_json(metrics, ["a", "b"], "model.tflite", out)
        assert (tmp_path / "benchmark.json").exists()

    def test_json_structure(self, tmp_path):
        metrics = {"roc-auc": 0.95, "cmAP": 0.88, "f1": 0.80, "total_chunks": 100}
        out = str(tmp_path / "benchmark.json")
        save_benchmark_json(metrics, ["a", "b"], "model.tflite", out)
        with open(out) as f:
            data = json.load(f)
        assert "model_path" in data
        assert "num_classes" in data
        assert "metrics" in data
        assert data["num_classes"] == 2

    def test_json_with_species(self, tmp_path, binary_data):
        y_true, y_scores, classes = binary_data
        species_data = bootstrap_ap_ci(y_true, y_scores, classes, n_bootstrap=50)
        metrics = {"roc-auc": 0.9, "cmAP": 0.8, "f1": 0.7}
        out = str(tmp_path / "bench.json")
        save_benchmark_json(metrics, classes, "m.tflite", out, species_data=species_data)
        with open(out) as f:
            data = json.load(f)
        assert "species" in data
        assert len(data["species"]) == len(classes)

    def test_json_with_config(self, tmp_path):
        metrics = {"roc-auc": 0.85}
        cfg = {"sample_rate": 22050, "audio_frontend": "hybrid"}
        out = str(tmp_path / "bench.json")
        save_benchmark_json(metrics, ["a"], "m.tflite", out, config=cfg)
        with open(out) as f:
            data = json.load(f)
        assert data["config"]["sample_rate"] == 22050

    def test_ap_per_class_excluded(self, tmp_path):
        """ap_per_class should not appear in JSON metrics."""
        metrics = {"roc-auc": 0.9, "ap_per_class": [0.8, 0.9]}
        out = str(tmp_path / "bench.json")
        save_benchmark_json(metrics, ["a", "b"], "m.tflite", out)
        with open(out) as f:
            data = json.load(f)
        assert "ap_per_class" not in data["metrics"]


# ---------------------------------------------------------------------------
# print_ascii_det_curve
# ---------------------------------------------------------------------------

class TestPrintAsciiDetCurve:
    """Tests for print_ascii_det_curve."""

    def test_no_error(self, binary_data, capsys):
        y_true, y_scores, _ = binary_data
        far, frr, _ = compute_det_curve(y_true, y_scores)
        print_ascii_det_curve(far, frr)
        captured = capsys.readouterr()
        assert "DET Curve" in captured.out

    def test_custom_bins(self, binary_data, capsys):
        y_true, y_scores, _ = binary_data
        far, frr, _ = compute_det_curve(y_true, y_scores)
        print_ascii_det_curve(far, frr, bins=5, width=20)
        captured = capsys.readouterr()
        assert "FRR" in captured.out


# ---------------------------------------------------------------------------
# save_det_curve_plot
# ---------------------------------------------------------------------------

class TestSaveDetCurvePlot:
    """Tests for save_det_curve_plot."""

    def test_plot_created(self, tmp_path, binary_data):
        y_true, y_scores, _ = binary_data
        far, frr, _ = compute_det_curve(y_true, y_scores)
        out = str(tmp_path / "det.png")
        save_det_curve_plot(far, frr, out)
        assert (tmp_path / "det.png").exists()
        assert (tmp_path / "det.png").stat().st_size > 0


# ---------------------------------------------------------------------------
# save_html_report
# ---------------------------------------------------------------------------

class TestSaveHtmlReport:
    """Tests for save_html_report."""

    def test_html_created(self, tmp_path, binary_data):
        y_true, y_scores, classes = binary_data
        metrics = {"roc-auc": 0.95, "cmAP": 0.88, "f1": 0.80}
        out = str(tmp_path / "report.html")
        save_html_report(metrics, classes, y_true, y_scores, "model.tflite", out)
        assert (tmp_path / "report.html").exists()

    def test_html_contains_metrics(self, tmp_path, binary_data):
        y_true, y_scores, classes = binary_data
        metrics = {"roc-auc": 0.95, "cmAP": 0.88, "f1": 0.80}
        out = str(tmp_path / "report.html")
        save_html_report(metrics, classes, y_true, y_scores, "model.tflite", out)
        html = (tmp_path / "report.html").read_text()
        assert "Evaluation Report" in html
        assert "roc-auc" in html
        assert "cmAP" in html

    def test_html_with_species(self, tmp_path, binary_data):
        y_true, y_scores, classes = binary_data
        species_data = bootstrap_ap_ci(y_true, y_scores, classes, n_bootstrap=50)
        metrics = {"roc-auc": 0.9, "cmAP": 0.8}
        out = str(tmp_path / "report.html")
        save_html_report(metrics, classes, y_true, y_scores, "m.tflite", out, species_data=species_data)
        html = (tmp_path / "report.html").read_text()
        assert "Per-Species" in html

    def test_html_has_model_name(self, tmp_path, binary_data):
        y_true, y_scores, classes = binary_data
        metrics = {"roc-auc": 0.9}
        out = str(tmp_path / "report.html")
        save_html_report(metrics, classes, y_true, y_scores, "/path/to/my_model.tflite", out)
        html = (tmp_path / "report.html").read_text()
        assert "my_model.tflite" in html


# ---------------------------------------------------------------------------
# Latency measurement in evaluate()
# ---------------------------------------------------------------------------

class TestLatencyMeasurement:
    """Tests for latency measurement in evaluate()."""

    @pytest.fixture
    def eval_setup(self, tmp_path):
        """Minimal eval setup for latency testing."""
        import soundfile as sf

        sr = 22050
        duration = 3.0
        classes = ["bird_a", "bird_b"]
        files = []
        for cls in classes:
            cls_dir = tmp_path / cls
            cls_dir.mkdir()
            for i in range(2):
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
            "audio_frontend": "librosa",
            "mag_scale": "none",
        }
        return files, classes, cfg

    def test_latency_keys_present(self, eval_setup):
        """When measure_latency=True, latency keys should appear in metrics."""
        from birdnet_stm32.evaluation.metrics import evaluate

        files, classes, cfg = eval_setup
        scores = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)

        class FakeRunner:
            def __init__(self):
                self._idx = 0

            def predict(self, x):
                out = np.tile(scores[self._idx % 2], (x.shape[0], 1))
                self._idx += x.shape[0]
                return out.astype(np.float32)

        metrics, _, _, _ = evaluate(
            FakeRunner(), files, classes, cfg, pooling="avg", measure_latency=True
        )
        assert "latency_mean_ms" in metrics
        assert "latency_median_ms" in metrics
        assert "latency_p95_ms" in metrics
        assert "latency_p99_ms" in metrics
        assert metrics["latency_mean_ms"] >= 0

    def test_latency_keys_absent_when_disabled(self, eval_setup):
        """When measure_latency=False (default), no latency keys."""
        from birdnet_stm32.evaluation.metrics import evaluate

        files, classes, cfg = eval_setup
        scores = np.array([[0.5, 0.5]], dtype=np.float32)

        class FakeRunner:
            def predict(self, x):
                return np.tile(scores[0], (x.shape[0], 1)).astype(np.float32)

        metrics, _, _, _ = evaluate(FakeRunner(), files, classes, cfg, pooling="avg")
        assert "latency_mean_ms" not in metrics
