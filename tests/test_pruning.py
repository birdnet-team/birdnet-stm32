"""Tests for gradual magnitude pruning."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow required for pruning tests")

from birdnet_stm32.training.pruning import (
    MAX_LAYER_SPARSITY,
    GradualPruningScheduler,
    SparsityMaskEnforcer,
    apply_masks_to_export,
    build_pruning_model,
    classifier_head_layer,
    collect_sparsity_masks,
    compute_masks,
    evaluate_accuracy_gate,
    macro_roc_auc,
    mask_sparsity,
    polynomial_sparsity,
    select_prunable_layers,
    sparsity_report,
)


def _tiny_model(seed: int = 0) -> tf.keras.Model:
    """Build a model whose two pointwise convs clear the default size floor.

    ``stem_conv`` (288 weights) stays below it, ``dw`` and ``pred`` are exempt
        by type — the same exemptions that hold for a real DS-CNN.
    """
    tf.keras.utils.set_random_seed(seed)
    inputs = tf.keras.Input(shape=(8, 8, 4), name="input")
    x = tf.keras.layers.Conv2D(8, 3, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = tf.keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False, name="dw")(x)
    x = tf.keras.layers.Conv2D(128, 1, padding="same", use_bias=False, name="expand")(x)
    x = tf.keras.layers.Conv2D(64, 1, padding="same", use_bias=False, name="project")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    outputs = tf.keras.layers.Dense(3, activation="sigmoid", name="pred")(x)
    return tf.keras.Model(inputs, outputs, name="tiny")


class _StubModel:
    """Return fixed scores so the accuracy gate can be checked exactly."""

    def __init__(self, scores: np.ndarray):
        self.scores = scores

    def predict(self, inputs, batch_size=None, verbose=0):
        """Ignore the inputs and return the pre-set score matrix."""
        return self.scores[: len(inputs)]


# ---------------------------------------------------------------------------
# Layer selection
# ---------------------------------------------------------------------------


class TestLayerSelection:
    """Only redundant, large, non-depthwise conv kernels may be pruned."""

    def test_selects_large_pointwise_convs_only(self):
        names = [layer.name for layer in select_prunable_layers(_tiny_model())]
        assert names == ["expand", "project"]

    def test_depthwise_is_exempt_by_type(self):
        """No size floor can pull a depthwise kernel back in."""
        names = [layer.name for layer in select_prunable_layers(_tiny_model(), min_layer_params=1)]
        assert "dw" not in names  # 3x3xC depthwise: tiny and highly sensitive

    def test_classifier_head_is_prunable_by_default(self):
        """Conversion ships the head separately, so its size is the OTA cost."""
        names = [layer.name for layer in select_prunable_layers(_tiny_model(), min_layer_params=1)]
        assert "pred" in names

    def test_classifier_head_can_be_left_dense(self):
        names = [layer.name for layer in select_prunable_layers(_tiny_model(), min_layer_params=1, include_head=False)]
        assert "pred" not in names

    def test_head_still_obeys_the_size_floor(self):
        """The 192-weight head of this model is below the default floor."""
        names = [layer.name for layer in select_prunable_layers(_tiny_model())]
        assert "pred" not in names

    def test_non_output_dense_layers_stay_exempt(self):
        """Squeeze-and-excite gates are Dense but are not the shipped head."""
        inputs = tf.keras.Input(shape=(4,), name="input")
        gate = tf.keras.layers.Dense(64, activation="relu", name="se_reduce")(inputs)
        outputs = tf.keras.layers.Dense(3, activation="sigmoid", name="pred")(gate)
        model = tf.keras.Model(inputs, outputs)

        names = [layer.name for layer in select_prunable_layers(model, min_layer_params=1)]
        assert names == ["pred"]

    def test_classifier_head_layer_finds_the_output_dense(self):
        model = _tiny_model()
        assert classifier_head_layer(model) is model.get_layer("pred")

    def test_classifier_head_layer_is_none_without_a_dense_output(self):
        inputs = tf.keras.Input(shape=(4, 4, 2), name="input")
        outputs = tf.keras.layers.GlobalAveragePooling2D(name="gap")(inputs)
        assert classifier_head_layer(tf.keras.Model(inputs, outputs)) is None

    def test_nested_dense_without_a_functional_output_is_ignored(self):
        """Attention pooling owns a Dense helper whose ``.output`` is undefined."""
        from birdnet_stm32.models.blocks import AttentionPooling

        inputs = tf.keras.Input(shape=(4, 4, 8), name="input")
        x = AttentionPooling(name="attn_pool")(inputs)
        outputs = tf.keras.layers.Dense(3, activation="sigmoid", name="pred")(x)
        model = tf.keras.Model(inputs, outputs)

        assert classifier_head_layer(model) is model.get_layer("pred")
        assert [layer.name for layer in select_prunable_layers(model, min_layer_params=1)] == ["pred"]

    def test_small_kernels_are_exempt(self):
        """The stem holds 144 weights and stays dense at the default floor."""
        names = [layer.name for layer in select_prunable_layers(_tiny_model())]
        assert "stem_conv" not in names

    def test_min_layer_params_widens_the_selection(self):
        names = [layer.name for layer in select_prunable_layers(_tiny_model(), min_layer_params=16)]
        assert "stem_conv" in names

    def test_audio_frontend_kernels_are_exempt(self):
        """Frontend filterbanks are signal-processing filters, not capacity."""
        from birdnet_stm32.models.frontend import AudioFrontendLayer

        inputs = tf.keras.Input(shape=(2000, 1), name="raw_audio_input")
        x = AudioFrontendLayer(
            mode="raw",
            mel_bins=32,
            spec_width=16,
            sample_rate=8000,
            chunk_duration=0.25,
            mag_scale="pwl",
            name="audio_frontend",
        )(inputs)
        x = tf.keras.layers.Conv2D(64, 1, padding="same", use_bias=False, name="expand")(x)
        x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
        outputs = tf.keras.layers.Dense(2, activation="sigmoid", name="pred")(x)
        model = tf.keras.Model(inputs, outputs)

        # The floor is lowered so only the frontend exemption can exclude the
        # frontend's own (large) filterbank kernels.
        names = [layer.name for layer in select_prunable_layers(model, min_layer_params=16)]
        assert names == ["expand", "pred"]
        assert not any(name.startswith("audio_frontend") for name in names)


# ---------------------------------------------------------------------------
# Sparsity schedule
# ---------------------------------------------------------------------------


class TestPolynomialSparsity:
    """The cubic ramp must be monotone and hit both endpoints exactly."""

    def test_endpoints(self):
        assert polynomial_sparsity(0, 0, 100, 0.8) == pytest.approx(0.0)
        assert polynomial_sparsity(100, 0, 100, 0.8) == pytest.approx(0.8)
        assert polynomial_sparsity(500, 0, 100, 0.8) == pytest.approx(0.8)

    def test_monotone_non_decreasing(self):
        values = [polynomial_sparsity(step, 0, 100, 0.8) for step in range(101)]
        assert all(b >= a - 1e-9 for a, b in zip(values, values[1:], strict=False))

    def test_front_loaded(self):
        """More than half the sparsity arrives in the first half of the ramp."""
        assert polynomial_sparsity(50, 0, 100, 0.8) > 0.4

    def test_begin_step_delays_the_ramp(self):
        assert polynomial_sparsity(10, 20, 100, 0.8) == pytest.approx(0.0)

    def test_degenerate_ramp_returns_final(self):
        assert polynomial_sparsity(11, 10, 10, 0.6) == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Mask computation
# ---------------------------------------------------------------------------


class TestComputeMasks:
    """Masks must hit the requested sparsity and keep the largest weights."""

    def test_layerwise_hits_target_per_layer(self):
        kernels = {
            "a": np.random.default_rng(1).standard_normal((10, 10)).astype(np.float32),
            "b": np.random.default_rng(2).standard_normal((20, 20)).astype(np.float32),
        }
        masks = compute_masks(kernels, 0.5, scope="layerwise")
        for mask in masks.values():
            assert mask.mean() == pytest.approx(0.5, abs=0.01)

    def test_smallest_magnitudes_are_removed(self):
        kernel = np.array([[0.01, -5.0, 0.02, 3.0]], dtype=np.float32)
        mask = compute_masks({"a": kernel}, 0.5)["a"]
        np.testing.assert_array_equal(mask, [[0.0, 1.0, 0.0, 1.0]])

    def test_zero_sparsity_keeps_everything(self):
        kernel = np.random.default_rng(3).standard_normal((8, 8)).astype(np.float32)
        assert compute_masks({"a": kernel}, 0.0)["a"].all()

    def test_global_scope_shifts_budget_to_the_redundant_layer(self):
        """A layer of small weights should absorb more than its uniform share."""
        kernels = {
            "small": np.full((32, 32), 0.001, dtype=np.float32),
            "large": np.full((32, 32), 1.0, dtype=np.float32),
        }
        masks = compute_masks(kernels, 0.5, scope="global")
        assert 1.0 - masks["small"].mean() > 0.9
        assert 1.0 - masks["large"].mean() < 0.1

    def test_global_scope_respects_the_per_layer_ceiling(self):
        kernels = {
            "small": np.full((32, 32), 0.001, dtype=np.float32),
            "large": np.full((32, 32), 1.0, dtype=np.float32),
        }
        masks = compute_masks(kernels, 0.5, scope="global")
        # Without the ceiling the shared threshold would strip this layer bare.
        assert 1.0 - masks["small"].mean() <= MAX_LAYER_SPARSITY + 0.005

    def test_global_scope_hits_the_overall_target(self):
        rng = np.random.default_rng(4)
        kernels = {name: rng.standard_normal((16, 16)).astype(np.float32) for name in "abc"}
        masks = compute_masks(kernels, 0.7, scope="global")
        assert mask_sparsity(masks) == pytest.approx(0.7, abs=0.01)

    def test_global_scope_redistributes_budget_from_a_capped_layer(self):
        kernels = {
            "small": np.full((32, 32), 0.001, dtype=np.float32),
            "large": np.full((32, 32), 1.0, dtype=np.float32),
        }
        masks = compute_masks(kernels, 0.5, scope="global")

        assert mask_sparsity(masks) == pytest.approx(0.5, abs=0.001)
        assert 1.0 - masks["small"].mean() <= MAX_LAYER_SPARSITY + 0.005
        assert 1.0 - masks["large"].mean() > 0.0

    def test_global_scope_rejects_an_infeasible_target(self):
        kernel = np.ones((16, 16), dtype=np.float32)
        with pytest.raises(ValueError, match="exceeds the per-layer ceiling"):
            compute_masks({"a": kernel}, 0.8, scope="global", max_layer_sparsity=0.5)

    def test_invalid_scope_rejected(self):
        with pytest.raises(ValueError, match="Invalid prune scope"):
            compute_masks({"a": np.ones((4, 4), np.float32)}, 0.5, scope="magic")

    def test_out_of_range_sparsity_rejected(self):
        with pytest.raises(ValueError, match="Target sparsity"):
            compute_masks({"a": np.ones((4, 4), np.float32)}, 1.0)

    def test_override_gives_one_layer_its_own_sparsity(self):
        rng = np.random.default_rng(8)
        kernels = {name: rng.standard_normal((16, 16)).astype(np.float32) for name in ("body", "head")}
        masks = compute_masks(kernels, 0.5, overrides={"head": 0.8})

        assert 1.0 - masks["body"].mean() == pytest.approx(0.5, abs=0.01)
        assert 1.0 - masks["head"].mean() == pytest.approx(0.8, abs=0.01)

    def test_override_layer_is_excluded_from_global_ranking(self):
        """An overridden layer must not also absorb part of the global budget."""
        kernels = {
            "body": np.full((32, 32), 1.0, dtype=np.float32),
            "head": np.full((32, 32), 0.001, dtype=np.float32),
        }
        masks = compute_masks(kernels, 0.5, scope="global", overrides={"head": 0.25})

        assert 1.0 - masks["head"].mean() == pytest.approx(0.25, abs=0.01)
        assert 1.0 - masks["body"].mean() == pytest.approx(0.5, abs=0.01)

    def test_unknown_override_names_are_ignored(self):
        kernel = np.random.default_rng(9).standard_normal((16, 16)).astype(np.float32)
        masks = compute_masks({"a": kernel}, 0.5, overrides={"missing": 0.9})
        assert 1.0 - masks["a"].mean() == pytest.approx(0.5, abs=0.01)

    def test_out_of_range_override_rejected(self):
        with pytest.raises(ValueError, match="Override sparsity"):
            compute_masks({"a": np.ones((8, 8), np.float32)}, 0.5, overrides={"a": 1.0})


# ---------------------------------------------------------------------------
# Masked training graph
# ---------------------------------------------------------------------------


class TestPruningGraph:
    """The masked graph shares weights and leaves the deployment model clean."""

    def test_graph_shares_kernels_and_owns_the_masks(self):
        model = _tiny_model()
        prunable = select_prunable_layers(model)
        pruned, wrappers = build_pruning_model(model, prunable)

        assert set(wrappers) == {"expand", "project"}
        assert wrappers["expand"].target is model.get_layer("expand")
        # Masks live on the wrapper, never on the deployment layer.
        assert not any("pruning_mask" in weight.path for weight in model.weights)
        assert pruned.output_shape == model.output_shape

    def test_mask_changes_the_forward_pass_only_when_applied(self):
        model = _tiny_model()
        pruned, wrappers = build_pruning_model(model, select_prunable_layers(model))
        inputs = np.random.default_rng(5).standard_normal((2, 8, 8, 4)).astype(np.float32)

        np.testing.assert_allclose(pruned(inputs).numpy(), model(inputs).numpy(), atol=1e-5)
        wrappers["expand"].mask.assign(np.zeros(wrappers["expand"].mask.shape, np.float32))
        assert not np.allclose(pruned(inputs).numpy(), model(inputs).numpy(), atol=1e-5)

    def test_dense_weights_survive_masking(self):
        """Revival at the next update requires the variable to stay dense."""
        model = _tiny_model()
        _pruned, wrappers = build_pruning_model(model, select_prunable_layers(model))
        wrappers["expand"].mask.assign(np.zeros(wrappers["expand"].mask.shape, np.float32))
        assert np.count_nonzero(model.get_layer("expand").kernel.numpy()) > 0

    def test_masked_weights_receive_a_straight_through_gradient(self):
        model = _tiny_model()
        pruned, wrappers = build_pruning_model(model, select_prunable_layers(model))
        wrappers["expand"].mask.assign(np.zeros(wrappers["expand"].mask.shape, np.float32))
        inputs = np.random.default_rng(11).standard_normal((2, 8, 8, 4)).astype(np.float32)

        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(pruned(inputs))
        gradient = tape.gradient(loss, model.get_layer("expand").kernel)

        assert gradient is not None
        assert np.count_nonzero(gradient.numpy()) > 0

    def test_export_bakes_masks_without_touching_the_trained_model(self):
        model = _tiny_model()
        export = _tiny_model(seed=1)
        prunable = select_prunable_layers(model)
        _pruned, wrappers = build_pruning_model(model, prunable)
        masks = {name: np.zeros(wrapper.mask.shape, np.float32) for name, wrapper in wrappers.items()}
        masks["expand"] = compute_masks({"expand": model.get_layer("expand").kernel.numpy()}, 0.5)["expand"]
        masks["project"] = np.ones(wrappers["project"].mask.shape, np.float32)

        apply_masks_to_export(model, export, masks)

        exported = export.get_layer("expand").kernel.numpy()
        assert 1.0 - np.count_nonzero(exported) / exported.size == pytest.approx(0.5, abs=0.01)
        # Non-pruned layers are copied verbatim, and the source stays dense.
        np.testing.assert_allclose(export.get_layer("pred").get_weights()[0], model.get_layer("pred").get_weights()[0])
        assert np.count_nonzero(model.get_layer("expand").kernel.numpy()) == exported.size

    def test_dense_head_is_masked_through_matmul(self):
        """The wrapper must run Dense heads, not only convolutions."""
        model = _tiny_model()
        prunable = select_prunable_layers(model, min_layer_params=1)
        pruned, wrappers = build_pruning_model(model, prunable)
        assert "pred" in wrappers

        inputs = np.random.default_rng(10).standard_normal((2, 8, 8, 4)).astype(np.float32)
        np.testing.assert_allclose(pruned(inputs).numpy(), model(inputs).numpy(), atol=1e-5)
        wrappers["pred"].mask.assign(np.zeros(wrappers["pred"].mask.shape, np.float32))
        # With every head weight masked the output collapses to sigmoid(bias).
        masked = pruned(inputs).numpy()
        expected = 1.0 / (1.0 + np.exp(-model.get_layer("pred").get_weights()[1]))
        np.testing.assert_allclose(masked, np.tile(expected, (2, 1)), atol=1e-5)

    def test_export_rejects_a_mismatched_model(self):
        model = _tiny_model()
        other = tf.keras.Sequential([tf.keras.layers.Dense(2, input_shape=(4,))])
        with pytest.raises(ValueError, match="layer for layer"):
            apply_masks_to_export(model, other, {})


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class TestGradualPruningScheduler:
    """The ramp must reach the target sparsity and then hold it fixed."""

    def _scheduler(self, model, **kwargs):
        _pruned, wrappers = build_pruning_model(model, select_prunable_layers(model))
        defaults = dict(begin_step=0, end_step=10, final_sparsity=0.5, frequency=2)
        defaults.update(kwargs)
        return GradualPruningScheduler(wrappers, **defaults)

    def test_reaches_the_target_and_freezes(self):
        model = _tiny_model()
        scheduler = self._scheduler(model)
        for _ in range(20):
            scheduler.on_train_batch_end(0)
        assert scheduler.current_sparsity == pytest.approx(0.5, abs=0.01)
        assert scheduler._frozen is True  # noqa: SLF001

    def test_sparsity_grows_during_the_ramp(self):
        model = _tiny_model()
        scheduler = self._scheduler(model, end_step=100, frequency=10)
        seen = []
        for _ in range(100):
            scheduler.on_train_batch_end(0)
            seen.append(scheduler.current_sparsity)
        assert seen[9] > 0.0
        assert seen[-1] > seen[9]
        assert all(b >= a - 1e-9 for a, b in zip(seen, seen[1:], strict=False))

    def test_frozen_masks_ignore_later_weight_changes(self):
        model = _tiny_model()
        scheduler = self._scheduler(model, end_step=4, frequency=2)
        for _ in range(4):
            scheduler.on_train_batch_end(0)
        frozen = scheduler.masks()["expand"].copy()

        expand = model.get_layer("expand")
        expand.kernel.assign(np.zeros(expand.kernel.shape, np.float32))
        for _ in range(20):
            scheduler.on_train_batch_end(0)
        np.testing.assert_array_equal(scheduler.masks()["expand"], frozen)

    def test_head_override_reaches_its_own_final_sparsity(self):
        model = _tiny_model()
        prunable = select_prunable_layers(model, min_layer_params=1)
        _pruned, wrappers = build_pruning_model(model, prunable)
        scheduler = GradualPruningScheduler(
            wrappers,
            begin_step=0,
            end_step=10,
            final_sparsity=0.5,
            frequency=2,
            final_overrides={"pred": 0.8},
        )
        for _ in range(12):
            scheduler.on_train_batch_end(0)

        masks = scheduler.masks()
        assert 1.0 - masks["pred"].mean() == pytest.approx(0.8, abs=0.02)
        assert 1.0 - masks["expand"].mean() == pytest.approx(0.5, abs=0.01)

    def test_head_override_ramps_rather_than_jumping(self):
        model = _tiny_model()
        prunable = select_prunable_layers(model, min_layer_params=1)
        _pruned, wrappers = build_pruning_model(model, prunable)
        scheduler = GradualPruningScheduler(
            wrappers,
            begin_step=0,
            end_step=200,
            final_sparsity=0.5,
            frequency=1,
            final_overrides={"pred": 0.8},
        )
        scheduler.on_train_batch_end(0)
        assert 1.0 - scheduler.masks()["pred"].mean() < 0.8

    def test_masks_are_recomputed_from_current_magnitudes(self):
        """A weight that grew back must be able to re-enter the network."""
        model = _tiny_model()
        scheduler = self._scheduler(model, end_step=1000, frequency=1, final_sparsity=0.5)
        scheduler.on_train_batch_end(0)

        expand = model.get_layer("expand")
        revived = np.zeros(expand.kernel.shape, np.float32)
        revived.reshape(-1)[0] = 100.0
        expand.kernel.assign(revived)
        for _ in range(9):
            scheduler.on_train_batch_end(0)
        assert scheduler.masks()["expand"].reshape(-1)[0] == 1.0


# ---------------------------------------------------------------------------
# Sparsity preservation for downstream steps
# ---------------------------------------------------------------------------


class TestSparsityPreservation:
    """A later fine-tuning step must not refill the pruned slots."""

    def test_masks_recovered_from_a_pruned_checkpoint(self):
        model = _tiny_model()
        expand = model.get_layer("expand")
        kernel = expand.kernel.numpy()
        kernel.reshape(-1)[: kernel.size // 2] = 0.0
        expand.kernel.assign(kernel)

        masks = collect_sparsity_masks(model)
        assert set(masks) == {"expand"}
        assert masks["expand"].mean() == pytest.approx(0.5, abs=0.01)

    def test_dense_checkpoint_yields_no_masks(self):
        assert collect_sparsity_masks(_tiny_model()) == {}

    def test_small_pruned_classifier_head_is_recovered(self):
        model = _tiny_model()
        head = model.get_layer("pred")
        kernel = head.kernel.numpy()
        kernel.reshape(-1)[: kernel.size // 2] = 0.0
        head.kernel.assign(kernel)

        masks = collect_sparsity_masks(model)

        assert "pred" in masks
        assert masks["pred"].mean() == pytest.approx(0.5, abs=0.01)

    def test_enforcer_restores_zeros_after_a_weight_update(self):
        model = _tiny_model()
        expand = model.get_layer("expand")
        kernel = expand.kernel.numpy()
        kernel.reshape(-1)[: kernel.size // 2] = 0.0
        expand.kernel.assign(kernel)
        enforcer = SparsityMaskEnforcer(model, collect_sparsity_masks(model))

        expand.kernel.assign(np.ones(expand.kernel.shape, np.float32))
        enforcer.on_train_batch_end(0)

        restored = expand.kernel.numpy().reshape(-1)
        assert np.count_nonzero(restored[: kernel.size // 2]) == 0
        assert np.count_nonzero(restored[kernel.size // 2 :]) == kernel.size - kernel.size // 2


# ---------------------------------------------------------------------------
# Reporting and the accuracy gate
# ---------------------------------------------------------------------------


class TestReportingAndGate:
    """The step must measure what it produced and refuse a real regression."""

    def test_sparsity_report_counts_zeros(self):
        model = _tiny_model()
        expand = model.get_layer("expand")
        kernel = expand.kernel.numpy()
        kernel.reshape(-1)[: kernel.size // 2] = 0.0
        expand.kernel.assign(kernel)
        masks = {"expand": (kernel != 0).astype(np.float32)}

        report = sparsity_report(model, masks)
        assert report["prunable_params"] == kernel.size
        assert report["prunable_sparsity"] == pytest.approx(0.5, abs=0.01)
        assert report["layers"][0]["layer"] == "expand"
        assert 0.0 < report["model_sparsity"] < report["prunable_sparsity"]

    def test_macro_roc_auc_skips_single_valued_classes(self):
        labels = np.array([[1, 0], [0, 0], [1, 0], [0, 0]], dtype=np.float32)
        scores = np.array([[0.9, 0.5], [0.1, 0.5], [0.8, 0.5], [0.2, 0.5]], dtype=np.float32)
        assert macro_roc_auc(labels, scores) == pytest.approx(1.0)

    def test_macro_roc_auc_is_nan_without_a_scorable_class(self):
        labels = np.zeros((4, 2), dtype=np.float32)
        scores = np.full((4, 2), 0.5, dtype=np.float32)
        assert np.isnan(macro_roc_auc(labels, scores))

    def _gate_dataset(self):
        rng = np.random.default_rng(6)
        inputs = rng.standard_normal((8, 8, 8, 4)).astype(np.float32)
        labels = np.tile(np.array([[1, 0, 1], [0, 1, 0]], np.float32), (4, 1))
        return tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(4)

    def test_gate_passes_for_an_identical_model(self):
        model = _tiny_model()
        result = evaluate_accuracy_gate(model, model, self._gate_dataset(), 8, 0.005, 4)
        assert result["passed"] is True
        assert result["roc_auc_drop"] == pytest.approx(0.0)
        assert result["samples"] == 8

    def test_gate_fails_when_the_pruned_model_regresses(self):
        labels = np.tile(np.array([[1, 0, 1], [0, 1, 0]], np.float32), (4, 1))
        perfect = _StubModel(labels.copy())
        chance = _StubModel(np.full_like(labels, 0.5))
        result = evaluate_accuracy_gate(perfect, chance, self._gate_dataset(), 8, 0.005, 4)

        assert result["baseline_macro_roc_auc"] == pytest.approx(1.0)
        assert result["pruned_macro_roc_auc"] == pytest.approx(0.5)
        assert result["roc_auc_drop"] == pytest.approx(0.5)
        assert result["passed"] is False

    def test_gate_passes_a_regression_inside_the_tolerance(self):
        labels = np.tile(np.array([[1, 0, 1], [0, 1, 0]], np.float32), (4, 1))
        perfect = _StubModel(labels.copy())
        nudged = _StubModel(labels * 0.9 + 0.05)
        result = evaluate_accuracy_gate(perfect, nudged, self._gate_dataset(), 8, 0.005, 4)

        assert result["roc_auc_drop"] == pytest.approx(0.0)
        assert result["passed"] is True

    def test_gate_reports_the_configured_tolerance(self):
        model = _tiny_model()
        result = evaluate_accuracy_gate(model, model, self._gate_dataset(), 8, 0.02, 4)
        assert result["max_roc_auc_drop"] == pytest.approx(0.02)

    def test_gate_rejects_an_empty_dataset(self):
        empty = tf.data.Dataset.from_tensor_slices(
            (np.zeros((0, 8, 8, 4), np.float32), np.zeros((0, 3), np.float32))
        ).batch(4)
        with pytest.raises(ValueError, match="no validation samples"):
            evaluate_accuracy_gate(_tiny_model(), _tiny_model(), empty, 8, 0.005, 4)


# ---------------------------------------------------------------------------
# End-to-end behaviour
# ---------------------------------------------------------------------------


class TestPrunedTrainingRun:
    """A short distilled run must end at the target sparsity and still fit."""

    def test_masked_model_trains_and_keeps_its_sparsity(self):
        from birdnet_stm32.training.distillation import DistilledModel

        model = _tiny_model()
        teacher = _tiny_model()
        teacher.set_weights(model.get_weights())
        student, wrappers = build_pruning_model(model, select_prunable_layers(model))
        distilled = DistilledModel(student, teacher)
        distilled.compile(optimizer="adam", loss="binary_crossentropy")
        scheduler = GradualPruningScheduler(wrappers, 0, 8, final_sparsity=0.5, frequency=2)

        rng = np.random.default_rng(7)
        inputs = rng.standard_normal((16, 8, 8, 4)).astype(np.float32)
        labels = (rng.random((16, 3)) > 0.5).astype(np.float32)
        for _ in range(10):
            metrics = distilled.train_on_batch(inputs, labels, return_dict=True)
            scheduler.on_train_batch_end(0)
        assert np.isfinite(metrics["loss"])
        assert scheduler.current_sparsity == pytest.approx(0.5, abs=0.01)

        export = _tiny_model(seed=2)
        apply_masks_to_export(model, export, scheduler.masks())
        report = sparsity_report(export, scheduler.masks())
        assert report["prunable_sparsity"] == pytest.approx(0.5, abs=0.01)
        # The training graph itself stays dense so masked weights can revive.
        assert np.count_nonzero(model.get_layer("expand").kernel.numpy()) == wrappers["expand"].mask.numpy().size
