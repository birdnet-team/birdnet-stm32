"""Tests for models that emit logits instead of probabilities.

Quantizing probabilities puts every score on the INT8 1/256 grid, which floors
everything below ~0.002 and ties the rest. Emitting logits and applying the
sigmoid afterwards measured +0.021 (raw) and +0.044 (hybrid) catalog cMAP.
"""

import numpy as np
import pytest

from birdnet_stm32.training.config import ModelConfig

tf = pytest.importorskip("tensorflow", reason="TensorFlow required")

from birdnet_stm32.models.runners import SigmoidRunner, runner_for_config  # noqa: E402


class FakeRunner:
    """Returns whatever it is given, so the wrapper's arithmetic is visible."""

    def __init__(self, output):
        self.output = np.asarray(output, dtype=np.float32)
        self.calls = 0

    def predict(self, x):
        self.calls += 1
        return self.output


class TestConfigField:
    def test_defaults_to_sigmoid_so_existing_configs_load_unchanged(self):
        assert ModelConfig().output_activation == "sigmoid"
        assert ModelConfig.from_dict({"num_classes": 0}).output_activation == "sigmoid"

    def test_accepts_logit(self):
        assert ModelConfig(output_activation="logit").output_activation == "logit"

    def test_rejects_anything_else(self):
        with pytest.raises(ValueError, match="output_activation"):
            ModelConfig(output_activation="softmax")

    def test_round_trips_through_json(self, tmp_path):
        path = tmp_path / "cfg.json"
        ModelConfig(output_activation="logit").save(path)
        assert ModelConfig.load(path).output_activation == "logit"
        assert '"output_activation": "logit"' in path.read_text()


class TestRunnerWrapping:
    def test_logit_config_applies_the_sigmoid(self):
        inner = FakeRunner([[0.0, 2.0, -2.0]])
        wrapped = runner_for_config(inner, {"output_activation": "logit"})
        assert isinstance(wrapped, SigmoidRunner)
        np.testing.assert_allclose(wrapped.predict(None), [[0.5, 0.880797, 0.119203]], rtol=1e-5)

    def test_sigmoid_config_returns_the_runner_untouched(self):
        inner = FakeRunner([[0.25]])
        assert runner_for_config(inner, {"output_activation": "sigmoid"}) is inner

    def test_missing_field_and_no_config_leave_the_runner_alone(self):
        inner = FakeRunner([[0.25]])
        assert runner_for_config(inner, {}) is inner
        assert runner_for_config(inner, None) is inner

    def test_wrapped_output_is_a_valid_probability(self):
        rng = np.random.default_rng(0)
        logits = rng.standard_normal((4, 100)).astype(np.float32) * 12.0
        out = runner_for_config(FakeRunner(logits), {"output_activation": "logit"}).predict(None)
        assert out.shape == logits.shape
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_the_sigmoid_preserves_ranking(self):
        """cMAP is rank-based, so wrapping must not reorder anything."""
        logits = np.array([[-8.0, 3.0, -1.0, 7.5]], dtype=np.float32)
        out = runner_for_config(FakeRunner(logits), {"output_activation": "logit"}).predict(None)
        assert list(np.argsort(out[0])) == list(np.argsort(logits[0]))


class TestStrippingTheHead:
    """What `convert --output_activation logit` does to the graph."""

    def _model(self, activation="sigmoid"):
        inputs = tf.keras.Input(shape=(4,))
        outputs = tf.keras.layers.Dense(3, activation=activation, name="pred")(inputs)
        return tf.keras.Model(inputs, outputs)

    def test_linear_head_emits_logits_whose_sigmoid_is_the_original_output(self):
        model = self._model()
        x = np.random.default_rng(0).standard_normal((2, 4)).astype(np.float32)
        probabilities = np.asarray(model(x, training=False))

        model.get_layer("pred").activation = tf.keras.activations.linear
        logits = np.asarray(model(x, training=False))

        assert not np.allclose(logits, probabilities)
        np.testing.assert_allclose(1.0 / (1.0 + np.exp(-logits)), probabilities, atol=1e-5)

    def test_a_model_without_a_sigmoid_head_is_refused(self):
        model = self._model(activation="relu")
        head = model.layers[-1]
        assert head.activation.__name__ != "sigmoid"  # what convert checks before stripping
