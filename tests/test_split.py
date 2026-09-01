"""Tests for splitting a model into a backbone and a swappable classifier head."""

import gzip
import os

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow required for split tests")

from birdnet_stm32.conversion.quantize import convert_to_tflite
from birdnet_stm32.conversion.split import (
    count_parameters,
    embedding_dimension,
    find_embedding_layer,
    gzip_file,
    size_record,
    split_model,
    weight_sparsity,
)
from birdnet_stm32.models.runners import ChainedTFLiteRunner, TFLiteRunner, load_model_runner


def _tiny_model(seed: int = 0, pooling: str = "gap", width: int = 16) -> tf.keras.Model:
    """Build a small classifier with the same head shape as the DS-CNN."""
    tf.keras.utils.set_random_seed(seed)
    inputs = tf.keras.Input(shape=(8, 8, 4), name="input")
    x = tf.keras.layers.Conv2D(width, 3, padding="same", activation="relu", name="stem_conv")(inputs)
    if pooling == "flatten":
        x = tf.keras.layers.Flatten(name="flatten")(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout")(x)
    outputs = tf.keras.layers.Dense(6, activation="sigmoid", name="pred")(x)
    return tf.keras.Model(inputs, outputs, name="tiny")


# ---------------------------------------------------------------------------
# Split point
# ---------------------------------------------------------------------------


class TestFindEmbeddingLayer:
    """The split point is the pooling layer that yields the embedding."""

    def test_finds_global_average_pooling(self):
        model = _tiny_model()
        assert find_embedding_layer(model) is model.get_layer("gap")

    def test_finds_flatten(self):
        model = _tiny_model(pooling="flatten")
        assert find_embedding_layer(model) is model.get_layer("flatten")

    def test_finds_custom_attention_pooling_by_name(self):
        inputs = tf.keras.Input(shape=(4,), name="input")
        x = tf.keras.layers.Dense(8, name="attn_pool")(inputs)
        outputs = tf.keras.layers.Dense(2, activation="sigmoid", name="pred")(x)
        model = tf.keras.Model(inputs, outputs)
        assert find_embedding_layer(model).name == "attn_pool"

    def test_raises_without_a_pooling_layer(self):
        inputs = tf.keras.Input(shape=(4,), name="input")
        outputs = tf.keras.layers.Dense(2, activation="sigmoid", name="pred")(inputs)
        with pytest.raises(ValueError, match="embedding layer"):
            find_embedding_layer(tf.keras.Model(inputs, outputs))


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


class TestSplitModel:
    """The two halves must reproduce the original model exactly."""

    def test_shapes_and_naming(self):
        model = _tiny_model()
        backbone, classifier = split_model(model)

        assert backbone.output_shape == (None, 16)
        assert classifier.input_shape == (None, 16)
        assert classifier.output_shape == (None, 6)
        assert classifier.inputs[0].name == "embeddings"
        assert embedding_dimension(backbone) == 16

    def test_chained_output_matches_the_source_model(self):
        model = _tiny_model()
        backbone, classifier = split_model(model)
        inputs = np.random.default_rng(1).standard_normal((4, 8, 8, 4)).astype(np.float32)

        chained = classifier(backbone(inputs, training=False), training=False).numpy()
        np.testing.assert_allclose(chained, model(inputs, training=False).numpy(), atol=1e-6)

    def test_head_carries_its_own_weights(self):
        """The head is rebuilt, so editing it must not disturb the source."""
        model = _tiny_model()
        _backbone, classifier = split_model(model)
        original = model.get_layer("pred").get_weights()

        head_dense = classifier.get_layer("pred")
        head_dense.set_weights([np.zeros_like(w) for w in head_dense.get_weights()])

        np.testing.assert_allclose(model.get_layer("pred").get_weights()[0], original[0])

    def test_source_model_graph_is_not_mutated(self):
        model = _tiny_model()
        before = [len(layer._outbound_nodes) for layer in model.layers]  # noqa: SLF001
        split_model(model)
        after = [len(layer._outbound_nodes) for layer in model.layers]  # noqa: SLF001
        assert before == after

    def test_parameter_split_accounts_for_every_weight(self):
        model = _tiny_model()
        backbone, classifier = split_model(model)
        assert count_parameters(backbone) + count_parameters(classifier) == count_parameters(model)

    def test_head_holds_only_the_classifier_weights(self):
        model = _tiny_model()
        _backbone, classifier = split_model(model)
        assert count_parameters(classifier) == 16 * 6 + 6

    def test_rejects_a_branched_head(self):
        inputs = tf.keras.Input(shape=(8, 8, 4), name="input")
        x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(inputs)
        left = tf.keras.layers.Dense(6, name="left")(x)
        right = tf.keras.layers.Dense(6, name="right")(x)
        outputs = tf.keras.layers.Add(name="pred")([left, right])
        with pytest.raises(ValueError, match="single chain|branched"):
            split_model(tf.keras.Model(inputs, outputs))

    def test_rejects_a_model_that_is_all_backbone(self):
        inputs = tf.keras.Input(shape=(8, 8, 4), name="input")
        outputs = tf.keras.layers.GlobalAveragePooling2D(name="gap")(inputs)
        with pytest.raises(ValueError, match="no classifier head"):
            split_model(tf.keras.Model(inputs, outputs))


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------


class TestPackaging:
    """The head is shipped compressed, so its measurements must be right."""

    def test_gzip_roundtrip(self, tmp_path):
        source = tmp_path / "model.bin"
        payload = b"birdnet" * 500
        source.write_bytes(payload)

        archive = gzip_file(str(source))

        assert archive == str(source) + ".gz"
        with gzip.open(archive, "rb") as handle:
            assert handle.read() == payload

    def test_gzip_is_byte_stable_across_runs(self, tmp_path):
        """An OTA update should be diffable, so the archive must be reproducible."""
        source = tmp_path / "model.bin"
        source.write_bytes(b"birdnet" * 500)

        first = (tmp_path / "a.gz").as_posix()
        second = (tmp_path / "b.gz").as_posix()
        gzip_file(str(source), first)
        gzip_file(str(source), second)

        with open(first, "rb") as a, open(second, "rb") as b:
            assert a.read() == b.read()

    def test_sparse_head_compresses_better_than_a_dense_one(self, tmp_path):
        """This is the whole point of pruning the head before shipping it."""
        # A 64-wide embedding gives the head 384 INT8 weights, above the
        # floor below which weight_sparsity ignores a tensor.
        dense = _tiny_model(seed=2, width=64)
        sparse = _tiny_model(seed=2, width=64)
        weights, bias = sparse.get_layer("pred").get_weights()
        weights = weights.copy()
        weights[: weights.shape[0] // 2] = 0.0
        sparse.get_layer("pred").set_weights([weights, bias])

        rng = np.random.default_rng(3)
        samples = rng.standard_normal((16, 8, 8, 4)).astype(np.float32)
        sizes = {}
        for tag, model in (("dense", dense), ("sparse", sparse)):
            _backbone, classifier = split_model(model)
            embeddings = _backbone.predict(samples, verbose=0)

            def gen(embeddings=embeddings):
                for row in embeddings:
                    yield [row[None]]

            path = str(tmp_path / f"{tag}.tflite")
            convert_to_tflite(classifier, gen, path)
            sizes[tag] = size_record(path)

        assert sizes["sparse"]["int8_sparsity"] > sizes["dense"]["int8_sparsity"]
        assert sizes["sparse"]["gzip_bytes"] < sizes["dense"]["gzip_bytes"]

    def test_size_record_reports_both_sizes(self, tmp_path):
        model = _tiny_model()
        _backbone, classifier = split_model(model)
        rng = np.random.default_rng(4)
        embeddings = rng.standard_normal((16, 16)).astype(np.float32)

        def gen():
            for row in embeddings:
                yield [row[None]]

        path = str(tmp_path / "head.tflite")
        convert_to_tflite(classifier, gen, path)
        record = size_record(path)

        assert record["size_bytes"] == os.path.getsize(path)
        assert record["gzip_bytes"] == os.path.getsize(path + ".gz")
        assert record["gzip_ratio"] > 1.0
        assert 0.0 <= record["int8_sparsity"] <= 1.0

    def test_weight_sparsity_ignores_tiny_tensors(self, tmp_path):
        model = _tiny_model()
        _backbone, classifier = split_model(model)
        rng = np.random.default_rng(5)
        embeddings = rng.standard_normal((8, 16)).astype(np.float32)

        def gen():
            for row in embeddings:
                yield [row[None]]

        path = str(tmp_path / "head.tflite")
        convert_to_tflite(classifier, gen, path)
        # 16x6 = 96 INT8 weights, below the default 256-element floor.
        assert weight_sparsity(path)["int8_weights"] == 0
        assert weight_sparsity(path, min_tensor_size=16)["int8_weights"] == 96


# ---------------------------------------------------------------------------
# Chained inference
# ---------------------------------------------------------------------------


class TestChainedRunner:
    """Evaluation and deployment must be able to run the split pair as one."""

    def _convert_pair(self, tmp_path) -> tuple[tf.keras.Model, str, str, np.ndarray]:
        model = _tiny_model(seed=6)
        backbone, classifier = split_model(model)
        rng = np.random.default_rng(7)
        samples = rng.standard_normal((16, 8, 8, 4)).astype(np.float32)

        def backbone_gen():
            for row in samples:
                yield [row[None]]

        backbone_path = str(tmp_path / "bb.tflite")
        convert_to_tflite(backbone, backbone_gen, backbone_path)

        embeddings = TFLiteRunner(backbone_path)

        def head_gen():
            for row in samples:
                yield [embeddings.predict(row[None])]

        classifier_path = str(tmp_path / "head.tflite")
        convert_to_tflite(classifier, head_gen, classifier_path)
        return model, backbone_path, classifier_path, samples

    def test_chained_matches_the_monolithic_model(self, tmp_path):
        model, backbone_path, classifier_path, samples = self._convert_pair(tmp_path)
        runner = ChainedTFLiteRunner(backbone_path, classifier_path)

        chained = runner.predict(samples[:4])
        reference = model(samples[:4], training=False).numpy()

        assert chained.shape == reference.shape
        np.testing.assert_allclose(chained, reference, atol=0.05)

    def test_load_model_runner_chains_when_given_a_head(self, tmp_path):
        _model, backbone_path, classifier_path, samples = self._convert_pair(tmp_path)
        runner = load_model_runner(backbone_path, classifier_path)

        assert isinstance(runner, ChainedTFLiteRunner)
        assert runner.predict(samples[:2]).shape == (2, 6)

    def test_load_model_runner_stays_single_without_a_head(self, tmp_path):
        _model, backbone_path, _classifier_path, _samples = self._convert_pair(tmp_path)
        assert isinstance(load_model_runner(backbone_path), TFLiteRunner)

    def test_head_cannot_be_chained_onto_a_keras_backbone(self, tmp_path):
        with pytest.raises(ValueError, match="only be chained onto a .tflite"):
            load_model_runner("model.keras", str(tmp_path / "head.tflite"))


# ---------------------------------------------------------------------------
# Conversion pipeline
# ---------------------------------------------------------------------------


class TestSplitConversion:
    """The CLI step must gate the chained pair and promote it atomically."""

    def _args(self, tmp_path, **overrides):
        import argparse

        defaults = dict(
            output_path=str(tmp_path / "model.tflite"),
            quantization="ptq",
            per_tensor=False,
            min_cosine_sim=0.90,
            min_cosine_p05=0.80,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def _generators(self, samples):
        def gen():
            for row in samples:
                yield [row[None]]

        return gen

    def test_emits_a_gated_and_measured_pair(self, tmp_path):
        from birdnet_stm32.cli.convert import _convert_split_head

        model = _tiny_model(seed=8, width=64)
        samples = np.random.default_rng(9).standard_normal((24, 8, 8, 4)).astype(np.float32)
        gen = self._generators(samples)
        cfg = {"class_names": [f"species_{i}" for i in range(6)]}

        report = _convert_split_head(model, cfg, self._args(tmp_path), gen, gen)

        backbone_path = tmp_path / "model_backbone.tflite"
        classifier_path = tmp_path / "model_classifier.tflite"
        assert backbone_path.is_file() and classifier_path.is_file()
        assert (tmp_path / "model_classifier.tflite.gz").is_file()
        assert (tmp_path / "model_classifier_labels.txt").read_text().split() == cfg["class_names"]

        assert report["embedding_layer"] == "gap"
        assert report["embedding_dim"] == 64
        assert report["classifier_params"] == 64 * 6 + 6
        assert report["chained_validation"]["cosine_mean"] > 0.9
        assert report["classifier"]["gzip_bytes"] < report["classifier"]["size_bytes"]
        assert report["quality_gate_passed"] is True

    def test_failed_gate_leaves_no_artifacts_behind(self, tmp_path):
        """A half-valid pair must never be promoted."""
        from birdnet_stm32.cli.convert import _convert_split_head

        model = _tiny_model(seed=10, width=64)
        samples = np.random.default_rng(11).standard_normal((16, 8, 8, 4)).astype(np.float32)
        gen = self._generators(samples)
        # No quantized model can reach a cosine of 1.1.
        args = self._args(tmp_path, min_cosine_sim=1.1)

        with pytest.raises(RuntimeError, match="Split-model quality check failed"):
            _convert_split_head(model, {}, args, gen, gen)

        assert not list(tmp_path.glob("*.tflite"))
        assert not list(tmp_path.glob(".splitting-*"))
