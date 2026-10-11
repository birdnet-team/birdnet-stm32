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
        assert not hasattr(args, "qat_calibration_percentile")

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


class TestBestPathDefaults:
    """A plain run reproduces the release recipe (docs/dev/int8-parity-plan.md)."""

    def test_training_defaults_are_the_release_recipe(self):
        args = _train()
        assert args.audio_frontend == "raw"
        assert args.mag_scale == "pwl"
        assert args.input_compression == "none"
        assert args.sample_rate == 24000
        assert args.chunk_duration == pytest.approx(2.5)
        assert args.embeddings_size == 1024
        assert args.alpha == pytest.approx(1.5)
        assert args.dw_kernel_size == 5
        assert args.stage_widths == [32, 64, 128, 256]
        assert args.spec_width == 448
        assert args.raw_exposure_gain == pytest.approx(16.0)
        assert args.max_chunks_per_file == 1
        assert args.class_cap_per_epoch == 3000
        assert args.epochs == 50
        assert args.learning_rate == pytest.approx(2e-4)
        assert args.warmup_epochs == pytest.approx(0.5)

    def test_frontend_sets_its_geometry(self):
        hybrid = _train("--audio_frontend", "hybrid")
        assert (hybrid.spec_width, hybrid.raw_exposure_gain) == (384, 1.0)
        assert _train("--audio_frontend", "librosa").spec_width == 256
        assert _train("--spec_width", "512").spec_width == 512
        assert _train("--raw_exposure_gain", "1").raw_exposure_gain == 1.0

    def test_a_teacher_cache_turns_on_the_teacher(self, tmp_path):
        plain = _train()
        assert (plain.teacher_weight, plain.crop_policy) == (0.0, "energy")
        taught = _train("--teacher_cache", str(tmp_path))
        assert (taught.teacher_weight, taught.crop_policy) == (0.5, "teacher")
        own = _train("--teacher_cache", str(tmp_path), "--teacher_weight", "0.3", "--crop_policy", "energy")
        assert (own.teacher_weight, own.crop_policy) == (0.3, "energy")

    def test_qat_trains_in_float32_uncompiled(self):
        args = _train("--qat")
        assert args.mixed_precision is False and args.jit_compile is False

    def test_qat_gets_its_own_schedule_and_range_refresh(self):
        args = _train("--qat")
        assert args.epochs == 8
        assert args.learning_rate == pytest.approx(2e-5)
        assert args.qat_range_refresh is True
        assert _train("--qat", "--no-qat_range_refresh").qat_range_refresh is False

    def test_linear_probe_keeps_its_rate(self):
        assert _train("--linear_probe").learning_rate == pytest.approx(1e-3)

    def test_explicit_schedule_wins(self):
        args = _train("--qat", "--epochs", "3", "--learning_rate", "1e-4")
        assert (args.epochs, args.learning_rate) == (3, pytest.approx(1e-4))

    def test_disproven_options_are_gone(self):
        with pytest.raises(SystemExit):
            _train("--mag_scale", "cpwl")
        for flag in (
            ["--upsample_ratio", "0.5"],
            ["--raw_exposure_mode", "channels"],
            ["--dw_weight_decay", "0"],
            ["--dw_activation", "leaky_relu"],
            ["--frontend_trainable"],
            ["--mixed_precision"],
            ["--jit_compile"],
        ):
            with pytest.raises(SystemExit):
                _train(*flag)
        with pytest.raises(SystemExit):
            _convert("--quantization", "dynamic")
        with pytest.raises(SystemExit):
            _train("--qat", "--qat_calibration_percentile", "99.9")


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

    def test_logit_output_by_default(self):
        assert _convert().output_activation == "logit"


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


def test_steps_per_epoch_defaults_to_one_pass(monkeypatch, tmp_path):
    """0 (the default) keeps one pass over the training files per epoch; a value fixes it."""
    import sys

    from birdnet_stm32.cli import train

    monkeypatch.setattr(sys, "argv", ["train", "--data_path_train", str(tmp_path)])
    assert train.get_args().steps_per_epoch == 0
    monkeypatch.setattr(sys, "argv", ["train", "--data_path_train", str(tmp_path), "--steps_per_epoch", "2850"])
    assert train.get_args().steps_per_epoch == 2850


def test_warmup_epochs_defaults_to_half_an_epoch_and_may_be_set(monkeypatch, tmp_path):
    """The 2.0 recipe warms up over half an epoch (one pass over a large dataset); any length can be set."""
    import sys

    from birdnet_stm32.cli import train

    monkeypatch.setattr(sys, "argv", ["train", "--data_path_train", str(tmp_path)])
    assert train.get_args().warmup_epochs == 0.5
    monkeypatch.setattr(sys, "argv", ["train", "--data_path_train", str(tmp_path), "--warmup_epochs", "2"])
    assert train.get_args().warmup_epochs == 2


class TestFilterbankDesign:
    """Every frontend trains: the raw filterbank's release options only go into raw models."""

    @pytest.mark.parametrize("frontend", ["raw", "hybrid", "librosa"])
    def test_each_frontend_gets_a_valid_config(self, frontend):
        from birdnet_stm32.cli.train import filterbank_design
        from birdnet_stm32.training.config import ModelConfig

        raw_magnitude, raw_bank = filterbank_design(frontend)
        ModelConfig(audio_frontend=frontend, raw_magnitude=raw_magnitude, raw_bank=raw_bank, num_classes=3)

    def test_raw_gets_the_release_design(self):
        from birdnet_stm32.cli.train import filterbank_design
        from birdnet_stm32.models.frontend import RELEASE_RAW_BANK, RELEASE_RAW_MAGNITUDE

        assert filterbank_design("raw") == (RELEASE_RAW_MAGNITUDE, RELEASE_RAW_BANK)
