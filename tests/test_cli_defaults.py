"""Guards on the CLI default settings that decide what a plain run produces.

These are deliberate product choices rather than incidental argparse values, so
a change to any of them should be a change to this file too.
"""

import re
import sys
from pathlib import Path
from unittest import mock

import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow required for CLI imports")

from birdnet_stm32.cli.convert import get_args as convert_args
from birdnet_stm32.cli.evaluate import get_args as evaluate_args
from birdnet_stm32.cli.train import get_args as train_args


def _train(*argv):
    with mock.patch("sys.argv", ["train", "--data_path_train", "data/train", *argv]):
        return train_args()


def _convert(*argv):
    with mock.patch("sys.argv", ["convert", "--checkpoint_path", "model.keras", *argv]):
        return convert_args()


def _evaluate(*argv):
    with mock.patch(
        "sys.argv",
        ["evaluate", "--model_path", "model.tflite", "--data_path_test", "data/test", *argv],
    ):
        return evaluate_args()


class TestCompressionStepDefaults:
    """Compression steps are opt-in and run one at a time."""

    def test_no_compression_step_runs_unless_asked(self):
        args = _train()
        assert args.qat is False
        assert args.linear_probe is False

    def test_compression_steps_are_mutually_exclusive(self):
        """Each step consumes a converged checkpoint and writes another."""
        with pytest.raises(SystemExit, match="mutually exclusive"):
            _train("--qat", "--linear_probe")

    def test_qat_calibration_defaults_match_the_converter(self):
        """QAT and conversion must observe the same activation statistics."""
        args = _train("--qat")
        assert args.qat_calibration_samples == 1024
        assert args.qat_calibration_percentile == 100.0

    def test_validation_subset_applies_to_all_training_and_is_nonnegative(self):
        assert _train("--validation_subset", "2513").validation_subset == 2513
        assert _train("--qat", "--validation_subset", "2513").validation_subset == 2513
        with pytest.raises(SystemExit):
            _train("--validation_subset", "-1")
        with pytest.raises(SystemExit):
            _train("--qat", "--validation_subset", "-1")

    def test_removed_options_are_gone(self):
        """Pruning and Optuna tuning were never used by any release.

        They are removed rather than kept as dead flags, so a run script that
        still passes them fails loudly instead of silently doing nothing.
        """
        args = _train()
        for name in ("prune", "tune", "n_trials", "qat_preserve_sparsity", "prune_head"):
            assert not hasattr(args, name), name
        with pytest.raises(SystemExit):
            _train("--prune")
        with pytest.raises(SystemExit):
            _train("--tune")


class TestConversionDefaults:
    """Conversion emits one model unless the split pair is requested."""

    def test_head_separation_is_off_by_default(self):
        assert _convert().split_head is False

    def test_head_separation_is_opt_in(self):
        assert _convert("--split_head").split_head is True

    def test_parity_gates_keep_their_thresholds(self):
        args = _convert()
        assert args.min_cosine_sim == pytest.approx(0.95)
        assert args.min_cosine_p05 == pytest.approx(0.90)
        assert args.quantization == "ptq"


class TestEvaluationDefaults:
    """Evaluation runs a single model unless a head is supplied to chain."""

    def test_no_classifier_head_by_default(self):
        assert _evaluate().classifier_path == ""

    def test_classifier_head_can_be_chained(self):
        assert _evaluate("--classifier_path", "head.tflite").classifier_path == "head.tflite"


class TestDocumentedArguments:
    """The argument reference must match the parser.

    A table that drifts from the code is worse than no table: it documents
    options that silently do nothing, which is exactly what the 1.2.0 cleanup
    removed. Pinning it here keeps the two in step.
    """

    DOC = Path(__file__).resolve().parents[1] / "docs" / "training.md"

    @staticmethod
    def parser_options() -> set[str]:
        import argparse

        from birdnet_stm32.cli.train import get_args

        recorded: set[str] = set()
        original = argparse.ArgumentParser.add_argument

        def capture(self, *args, **kwargs):
            recorded.update(a for a in args if isinstance(a, str) and a.startswith("--"))
            return original(self, *args, **kwargs)

        argparse.ArgumentParser.add_argument = capture
        try:
            with mock.patch.object(sys, "argv", ["train", "--data_path_train", "d"]):
                get_args()
        finally:
            argparse.ArgumentParser.add_argument = original
        return recorded

    def documented_options(self) -> set[str]:
        rows = re.findall(r"^\| `(--[a-z_0-9]+)`", self.DOC.read_text(), re.M)
        return set(rows)

    def test_every_documented_option_exists(self):
        undefined = self.documented_options() - self.parser_options()
        assert not undefined, f"documented but not accepted by the parser: {sorted(undefined)}"

    def test_every_option_is_documented(self):
        undocumented = self.parser_options() - self.documented_options() - {"--help"}
        assert not undocumented, f"accepted by the parser but undocumented: {sorted(undocumented)}"
