"""Host x board parity in the board test.

The board test exists to prove the device computes what the host computes, so
the comparison itself -- including the firmware's tie order -- is pinned here.
"""

import numpy as np
import pytest

from birdnet_stm32.deploy.board_test import (
    compare_board_host,
    firmware_top_k,
    load_truth,
    parse_serial_output,
)

LABELS = ["a", "b", "c", "d"]


class TestFirmwareTopK:
    def test_orders_by_score_and_stops_below_threshold(self):
        assert firmware_top_k(np.array([0.1, 0.9, 0.005, 0.5]), 5, 0.01) == [1, 3, 0]

    def test_ties_go_to_the_lower_index_like_the_firmware(self):
        """Saturated INT8 outputs tie; numpy's argsort would not order them the same."""
        assert firmware_top_k(np.array([0.2, 0.996, 0.996, 0.996]), 3, 0.01) == [1, 2, 3]

    def test_respects_k(self):
        assert firmware_top_k(np.array([0.4, 0.3, 0.2, 0.1]), 2, 0.01) == [0, 1]


def _board(name, *detections):
    return {"file": name, "detections": [{"label": label, "score": score} for label, score in detections]}


def _compare(board, host, **kwargs):
    options = {"top_k": 3, "threshold": 0.01, "tolerance": 0.05}
    options.update(kwargs)
    return compare_board_host(board, host, LABELS, **options)


class TestCompareBoardHost:
    def test_same_top1_within_tolerance_passes(self):
        host = {"F1.WAV": np.array([0.9, 0.05, 0.02, 0.0])}
        parity = _compare([_board("F1.WAV", ("a", 0.898), ("b", 0.051))], host)
        assert parity["passed"] is True
        assert parity["rows"][0]["agreement"] == "match"
        assert parity["max_score_diff"] == pytest.approx(0.002, abs=1e-6)

    def test_a_different_top1_is_a_mismatch(self):
        host = {"F1.WAV": np.array([0.9, 0.05, 0.0, 0.0])}
        parity = _compare([_board("F1.WAV", ("b", 0.9))], host)
        assert parity["passed"] is False
        assert parity["rows"][0]["agreement"] == "mismatch"

    def test_a_near_tie_is_not_a_disagreement(self):
        """The host scores the board's pick within tolerance of its own top-1."""
        host = {"F1.WAV": np.array([0.60, 0.58, 0.0, 0.0])}
        parity = _compare([_board("F1.WAV", ("b", 0.61), ("a", 0.59))], host)
        assert parity["rows"][0]["agreement"] == "tie"
        assert parity["passed"] is True

    def test_right_label_wrong_score_fails(self):
        """Same ranking with a drifted score is still the device computing something else."""
        host = {"F1.WAV": np.array([0.9, 0.0, 0.0, 0.0])}
        parity = _compare([_board("F1.WAV", ("a", 0.7))], host)
        assert parity["rows"][0]["agreement"] == "match"
        assert parity["passed"] is False

    def test_the_old_broken_raw_model_fails(self):
        """The failure this test missed before: one constant answer for every file."""
        host = {
            "F1.WAV": np.array([0.99, 0.0, 0.0, 0.0]),
            "F2.WAV": np.array([0.0, 0.98, 0.0, 0.0]),
        }
        board = [_board("F1.WAV", ("d", 0.08)), _board("F2.WAV", ("d", 0.08))]
        parity = _compare(board, host)
        assert parity["passed"] is False
        assert parity["agreeing"] == 0

    def test_counts_correct_detections_against_truth(self):
        host = {"F1.WAV": np.array([0.9, 0.0, 0.0, 0.0]), "F2.WAV": np.array([0.0, 0.0, 0.8, 0.0])}
        board = [_board("F1.WAV", ("a", 0.9)), _board("F2.WAV", ("c", 0.8))]
        parity = _compare(board, host, truth={"F1.WAV": "a", "F2.WAV": "b"})
        assert parity["board_correct"] == 1
        assert parity["host_correct"] == 1
        assert parity["labelled_files"] == 2

    def test_unknown_board_label_is_an_error(self):
        with pytest.raises(ValueError, match="does not have"):
            _compare([_board("F1.WAV", ("zebra", 0.9))], {"F1.WAV": np.zeros(4)})


def test_load_truth_reads_the_sd_card_manifest(tmp_path):
    audio = tmp_path / "sdcard" / "audio"
    audio.mkdir(parents=True)
    (tmp_path / "sdcard" / "manifest.csv").write_text("file,true_species,host_top1\n01_ncxx.wav,cardinal,cardinal\n")
    assert load_truth(audio) == {"01_NCXX.WAV": "cardinal"}
    assert load_truth(tmp_path) == {}


def test_parses_the_firmware_output_format():
    lines = [
        "[1/2] 01_NCXX.WAV",
        "  [WAV] 24000 Hz, 16-bit, 1 ch, 60000 samples",
        "  [BENCH] read=40ms stft=0ms npu=12ms total=52ms",
        "  01_NCXX.WAV:",
        "    [1] northern_cardinal: 98.0%",
        "    [2] blue_jay: 1.2%",
        "[2/2] 02_BJXX.WAV",
        "  02_BJXX.WAV:",
        "    [1] blue_jay: 99.6%",
        "=== DONE ===",
        "Processed: 2 / 2 files (0 errors)",
    ]
    parsed = parse_serial_output(lines, [], top_k=5, threshold=0.01)
    assert [r["file"] for r in parsed["results"]] == ["01_NCXX.WAV", "02_BJXX.WAV"]
    assert parsed["results"][0]["detections"] == [
        {"label": "northern_cardinal", "score": pytest.approx(0.98)},
        {"label": "blue_jay", "score": pytest.approx(0.012)},
    ]
    assert parsed["processed"] == 2
