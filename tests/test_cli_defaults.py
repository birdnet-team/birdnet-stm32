"""Guards on the CLI default settings that decide what a plain run produces.

These are deliberate product choices rather than incidental argparse values, so
a change to any of them should be a change to this file too.
"""

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
    """Compression steps are opt-in; their internals are on once selected."""

    def test_no_compression_step_runs_unless_asked(self):
        args = _train()
        assert args.prune is False
        assert args.qat is False
        assert args.linear_probe is False
        assert args.tune is False

    def test_pruning_covers_the_classifier_head_by_default(self):
        """The head is the artifact a split export ships, so it is pruned."""
        assert _train("--prune").prune_head is True
        assert _train("--prune", "--no_prune_head").prune_head is False

    def test_head_follows_the_shared_sparsity_target_by_default(self):
        assert _train("--prune").prune_head_sparsity == -1.0
        assert _train("--prune", "--prune_head_sparsity", "0.75").prune_head_sparsity == pytest.approx(0.75)

    def test_pruning_defaults_are_conservative(self):
        args = _train("--prune")
        assert args.prune_final_sparsity == pytest.approx(0.5)
        assert args.prune_scope == "layerwise"
        assert args.prune_ramp_fraction == pytest.approx(0.5)
        assert args.prune_max_auc_drop == pytest.approx(0.005)

    def test_qat_preserves_pruning_masks_by_default(self):
        assert _train("--qat").qat_preserve_sparsity is True
        assert _train("--qat", "--no_qat_preserve_sparsity").qat_preserve_sparsity is False

    def test_compression_steps_are_mutually_exclusive(self):
        with pytest.raises(SystemExit, match="mutually exclusive"):
            _train("--prune", "--qat")


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
