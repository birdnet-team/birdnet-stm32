"""N6-B0 variants for the 2.5 size study: stage tables and channel attention, through config, CLI, save and INT8."""

import numpy as np
import pytest
import tensorflow as tf
from test_dscnn import _raw_model

from birdnet_stm32.models.dscnn import build_dscnn_model
from birdnet_stm32.models.efficientnet import ChannelAttention, parse_blocks
from birdnet_stm32.training.config import ModelConfig

HALF = "1,3,1,16,1/6,3,1,24,2/6,5,1,40,2/6,3,2,80,2/6,5,2,112,1/6,5,2,192,2/6,3,1,320,1"  # B0 at half depth


def small_model(blocks=None, attention=()):
    """A spectrogram-input N6-B0 small enough to convert in a test."""
    return build_dscnn_model(
        num_mels=64,
        spec_width=256,
        sample_rate=16000,
        chunk_duration=2,
        embeddings_size=32,
        num_classes=3,
        audio_frontend="librosa",
        mag_scale="none",
        backbone="efficientnet_b0_n6",
        b0_blocks=blocks or parse_blocks("1,3,1,8,1/4,3,1,16,2/4,3,1,24,2/4,3,1,32,2/4,3,1,48,2"),
        attention_stages=attention,
    )


class TestParseBlocks:
    def test_reads_a_table(self):
        blocks = parse_blocks(HALF)
        assert len(blocks) == 7 and blocks[3] == (6, 3, 2, 80, 2)

    @pytest.mark.parametrize("spec", ["", "6,3,2,24", "6,4,2,24,2", "6,3,2,24,3", "6,3,0,24,2", "a,3,2,24,2"])
    def test_rejects_bad_stages(self, spec):
        with pytest.raises(ValueError):
            parse_blocks(spec)


class TestStageTable:
    def test_replaces_the_backbone_table(self):
        model = _raw_model(backbone="efficientnet_b0_n6", b0_blocks=parse_blocks(HALF))
        adds = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Add)]
        assert len(adds) == 3  # 10 blocks; only the second blocks of stages 4-6 keep their shape
        assert model.get_layer("block4b_project_conv").filters == 80
        assert "block4c_project_conv" not in {layer.name for layer in model.layers}

    def test_dscnn_takes_no_stage_table_or_attention(self):
        with pytest.raises(ValueError, match="EfficientNet"):
            _raw_model(backbone="dscnn", b0_blocks=parse_blocks(HALF))
        with pytest.raises(ValueError, match="EfficientNet"):
            _raw_model(backbone="dscnn", attention_stages=[7])


class TestChannelAttention:
    def test_blocks_follow_the_chosen_stages(self):
        model = _raw_model(backbone="efficientnet_b0_n6", attention_stages=[3, 7])
        attn = [layer for layer in model.layers if isinstance(layer, ChannelAttention)]
        assert [layer.name for layer in attn] == ["attn3_attn", "attn7_attn"]
        assert attn[1].heads == 320 // 32 and attn[0].heads == 1  # 40 channels: one head of 40
        assert model.output_shape == (None, 5)
        with pytest.raises(ValueError, match="attention_stages"):
            _raw_model(backbone="efficientnet_b0_n6", attention_stages=[8])

    def test_mixes_channels_per_head(self):
        layer = ChannelAttention(2)
        rng = np.random.default_rng(0)
        q, k, v = (tf.constant(rng.standard_normal((2, 3, 4, 8)), tf.float32) for _ in range(3))
        out = layer([q, k, v]).numpy()
        assert out.shape == (2, 3, 4, 8)
        # Head 0 (channels 0-3) only mixes v's channels 0-3: changing v's head-1 channels leaves it alone.
        v2 = v.numpy().copy()
        v2[..., 4:] += 1.0
        out2 = layer([q, k, tf.constant(v2)]).numpy()
        np.testing.assert_allclose(out[..., :4], out2[..., :4], atol=1e-6)
        assert not np.allclose(out[..., 4:], out2[..., 4:])

    def test_round_trips_through_save_and_qat_cloning(self, tmp_path):
        from birdnet_stm32.models.runners import load_keras_model

        model = small_model(attention=[4, 5])
        x = np.random.default_rng(0).random((2, 64, 256, 1)).astype(np.float32)
        path = str(tmp_path / "m.keras")
        model.save(path)
        np.testing.assert_allclose(load_keras_model(path)(x), model(x), atol=1e-5)
        clone = tf.keras.models.clone_model(
            model, clone_function=lambda layer: layer.__class__.from_config(layer.get_config())
        )
        clone.set_weights(model.get_weights())
        np.testing.assert_allclose(clone(x), model(x), atol=1e-5)

    def test_qat_graph_builds_and_tracks_the_float_model(self):
        from birdnet_stm32.training.qat import build_qat_model, calibrate_activation_ranges

        model = small_model(attention=[4, 5])
        rng = np.random.default_rng(1)
        x = rng.random((4, 64, 256, 1)).astype(np.float32)
        data = tf.data.Dataset.from_tensor_slices((x, np.zeros((4, 3), np.float32))).batch(2)
        qat = build_qat_model(model, calibrate_activation_ranges(model, data, max_samples=4), logit_output=False)
        out, ref = np.asarray(qat(x)), np.asarray(model(x))
        assert out.shape == ref.shape and np.isfinite(out).all()
        assert np.max(np.abs(out - ref)) < 0.1  # INT8-simulated, close to float

    def test_converts_to_int8(self):
        from birdnet_stm32.conversion.quantize import convert_to_tflite

        model = small_model(attention=[5])
        v = model.get_layer("attn5_v")  # non-negative weights on a ReLU6 input: v is >= 0, a zero point of -128
        v.set_weights([np.abs(w) for w in v.get_weights()])
        rng = np.random.default_rng(0)

        def rep():
            for _ in range(8):
                yield [rng.random((1, 64, 256, 1)).astype(np.float32)]

        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/m.tflite"
            convert_to_tflite(model, rep, path)
            interp = tf.lite.Interpreter(path)
            interp.allocate_tensors()
            details = interp._get_ops_details()  # noqa: SLF001
            interp.set_tensor(interp.get_input_details()[0]["index"], rng.random((1, 64, 256, 1)).astype(np.float32))
            interp.invoke()
            out = interp.get_tensor(interp.get_output_details()[0]["index"])
            zero_points = {t["index"]: t["quantization"][1] for t in interp.get_tensor_details()}
        assert np.isfinite(out).all() and out.shape == (1, 3)
        assert {"BATCH_MATMUL", "SOFTMAX"} <= {op["op_name"] for op in details}
        # XNNPACK's INT8 BATCH_MATMUL ignores its right-hand zero point; the converter centres those operands on 0.
        matmuls = [op for op in details if op["op_name"] == "BATCH_MATMUL"]
        assert len(matmuls) == 2 and all(zero_points[op["inputs"][1]] == 0 for op in matmuls)


class TestConfigAndFlags:
    def test_config_records_and_checks_them(self, tmp_path):
        cfg = ModelConfig(
            backbone="efficientnet_b0_n6", b0_blocks=[list(b) for b in parse_blocks(HALF)], attention_stages=[7]
        )
        cfg.save(tmp_path / "c.json")
        back = ModelConfig.load(tmp_path / "c.json")
        assert back.b0_blocks[3] == [6, 3, 2, 80, 2] and back.attention_stages == [7]
        assert ModelConfig.load(tmp_path / "c.json").backbone == "efficientnet_b0_n6"
        with pytest.raises(ValueError):
            ModelConfig(backbone="dscnn", attention_stages=[7])
        with pytest.raises(ValueError):
            ModelConfig(backbone="efficientnet_b0_n6", b0_blocks=[[6, 3, 2, 80]])

    def _args(self, monkeypatch, tmp_path, *extra):
        import sys

        from birdnet_stm32.cli import train

        monkeypatch.setattr(sys, "argv", ["train", "--data_path_train", str(tmp_path), *extra])
        return train.get_args()

    def test_train_flags(self, monkeypatch, tmp_path):
        args = self._args(
            monkeypatch,
            tmp_path,
            "--backbone",
            "efficientnet_b0_n6",
            "--b0_blocks",
            HALF,
            "--attention_stages",
            "3",
            "5",
            "7",
        )
        assert args.b0_blocks[0] == [1, 3, 1, 16, 1] and args.attention_stages == [3, 5, 7]
        assert self._args(monkeypatch, tmp_path).b0_blocks == []
        for extra in (["--b0_blocks", "6,3,2"], ["--backbone", "dscnn", "--attention_stages", "7"]):
            with pytest.raises(SystemExit):
                self._args(monkeypatch, tmp_path, *extra)
