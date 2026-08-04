"""Unit tests for training utilities."""

import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow required for trainer tests")

from birdnet_stm32.training.trainer import VALID_OPTIMIZERS, WarmupCosineDecay, _build_optimizer


class TestBuildOptimizer:
    """Tests for _build_optimizer."""

    def _make_schedule(self):
        return tf.keras.optimizers.schedules.CosineDecay(1e-3, 100)

    def test_adam(self):
        """'adam' should return an Adam optimizer."""
        opt = _build_optimizer("adam", self._make_schedule())
        assert isinstance(opt, tf.keras.optimizers.Adam)

    def test_sgd(self):
        """'sgd' should return an SGD optimizer."""
        opt = _build_optimizer("sgd", self._make_schedule())
        assert isinstance(opt, tf.keras.optimizers.SGD)

    def test_adamw(self):
        """'adamw' should return an AdamW optimizer."""
        opt = _build_optimizer("adamw", self._make_schedule(), weight_decay=1e-4)
        assert isinstance(opt, tf.keras.optimizers.AdamW)

    def test_invalid_raises(self):
        """Invalid optimizer name should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid optimizer"):
            _build_optimizer("rmsprop", self._make_schedule())

    def test_valid_optimizers_constant(self):
        """VALID_OPTIMIZERS should contain expected names."""
        assert set(VALID_OPTIMIZERS) == {"adam", "sgd", "adamw"}


class TestWarmupCosineDecay:
    """Tests for the warmup + cosine learning-rate schedule."""

    def test_warmup_ramps_up_to_peak(self):
        """LR climbs across warmup and reaches the peak at its end."""
        sched = WarmupCosineDecay(initial_learning_rate=1.0, decay_steps=100, warmup_steps=10)
        lrs = [float(sched(s)) for s in range(10)]
        assert lrs == sorted(lrs)
        assert lrs[0] < 0.2
        assert float(sched(9)) == pytest.approx(1.0, rel=1e-5)

    def test_decays_to_zero_at_end(self):
        """Cosine tail ends at (numerically) zero."""
        sched = WarmupCosineDecay(initial_learning_rate=1.0, decay_steps=100, warmup_steps=10)
        assert float(sched(100)) == pytest.approx(0.0, abs=1e-6)

    def test_no_warmup_starts_at_peak(self):
        """With warmup disabled the schedule starts at the peak rate."""
        sched = WarmupCosineDecay(initial_learning_rate=0.01, decay_steps=50, warmup_steps=0)
        assert float(sched(0)) == pytest.approx(0.01, rel=1e-5)

    def test_offset_continues_the_schedule(self):
        """A resumed run picks up where the previous one stopped.

        Without the offset, resuming would restart the cosine at the peak LR
        and undo the decay already applied.
        """
        fresh = WarmupCosineDecay(initial_learning_rate=1.0, decay_steps=100, warmup_steps=10)
        resumed = WarmupCosineDecay(initial_learning_rate=1.0, decay_steps=100, warmup_steps=10, offset_steps=60)
        assert float(resumed(0)) == pytest.approx(float(fresh(60)), rel=1e-5)
        assert float(resumed(0)) < float(fresh(0)) or float(fresh(0)) < 0.2

    def test_get_config_roundtrip(self):
        """Config carries every field needed to rebuild the schedule."""
        sched = WarmupCosineDecay(initial_learning_rate=0.5, decay_steps=20, warmup_steps=4, offset_steps=8)
        rebuilt = WarmupCosineDecay(**sched.get_config())
        assert float(rebuilt(3)) == pytest.approx(float(sched(3)), rel=1e-6)
