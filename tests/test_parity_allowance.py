"""Tests for the board-versus-host parity allowance on a logit model.

A probability output sat on a 1/256 grid, 0.004 steps, negligible beside the
0.031-0.036 board-versus-host frontend disagreement the 1.4 releases measured.
A logit output's grid is uniform in logits instead: one step spans
`scale * p * (1 - p)` in probability, up to `scale / 4` at p = 0.5, which is
0.03 at the release scale. Comparing that against a flat 0.05 printed
"tolerance 0.05" beside a measured 0.0715 and read like a failure that passed.
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow required")

from birdnet_stm32.deploy.board_test import int8_output_step, output_grid_parity_allowance  # noqa: E402


@pytest.fixture(scope="module")
def int8_model(tmp_path_factory):
    """A tiny INT8 model with float I/O, like every released bundle."""
    inputs = tf.keras.Input(shape=(8,))
    outputs = tf.keras.layers.Dense(4)(inputs)
    model = tf.keras.Model(inputs, outputs)
    rng = np.random.default_rng(0)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    converter.representative_dataset = lambda: ([rng.standard_normal((1, 8)).astype(np.float32) * 3] for _ in range(16))
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    path = tmp_path_factory.mktemp("parity") / "model.tflite"
    path.write_bytes(converter.convert())
    return str(path)


class TestInt8OutputStep:
    def test_it_finds_the_grid_behind_a_float_output(self, int8_model):
        """The output tensor is float32, so the grid is the tensor before it."""
        step = int8_output_step(int8_model)
        assert step is not None and step > 0


class TestAllowance:
    def test_a_probability_model_keeps_the_historical_allowance(self, int8_model):
        assert output_grid_parity_allowance(int8_model, {"output_activation": "sigmoid"}) == 0.05
        assert output_grid_parity_allowance(int8_model, {}) == 0.05

    def test_a_logit_model_gets_a_quarter_step_more(self, int8_model):
        step = int8_output_step(int8_model)
        allowance = output_grid_parity_allowance(int8_model, {"output_activation": "logit"})
        assert allowance == pytest.approx(0.05 + step / 4.0)
        assert allowance > 0.05

    def test_the_base_is_the_caller_s_to_choose(self, int8_model):
        assert output_grid_parity_allowance(int8_model, {"output_activation": "sigmoid"}, base=0.1) == 0.1
