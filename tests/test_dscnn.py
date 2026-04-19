"""Unit tests for DS-CNN model architecture."""

import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow required for model tests")

from birdnet_stm32.models.blocks import _make_divisible
from birdnet_stm32.models.dscnn import build_dscnn_model, ds_conv_block


class TestMakeDivisible:
    """Tests for _make_divisible utility."""

    def test_already_divisible(self):
        assert _make_divisible(32, 8) == 32

    def test_rounds_up(self):
        assert _make_divisible(30, 8) == 32

    def test_rounds_to_nearest(self):
        """28 rounds to 32 (rounds to nearest with bias toward up)."""
        assert _make_divisible(28, 8) == 32

    def test_minimum_value(self):
        """Should never return less than the divisor."""
        assert _make_divisible(1, 8) == 8
        assert _make_divisible(0, 8) == 8

    def test_float_input(self):
        assert _make_divisible(33.5, 8) == 32


class TestDsConvBlock:
    """Tests for ds_conv_block."""

    def test_output_shape(self):
        """Output should have the specified channel count."""
        x = tf.random.uniform((1, 16, 16, 32))
        y = ds_conv_block(x, out_ch=64, name="test_ds")
        assert y.shape == (1, 16, 16, 64)

    def test_stride(self):
        """Stride should halve spatial dimensions."""
        x = tf.random.uniform((1, 16, 16, 32))
        y = ds_conv_block(x, out_ch=64, stride_f=2, stride_t=2, name="test_stride")
        assert y.shape == (1, 8, 8, 64)

    def test_residual_connection(self):
        """When stride=1 and channels match, residual should be added."""
        x = tf.random.uniform((1, 8, 8, 32))
        y = ds_conv_block(x, out_ch=32, stride_f=1, stride_t=1, name="test_res")
        # Output should differ from a block without residual
        assert y.shape == (1, 8, 8, 32)


class TestBuildDscnnModel:
    """Tests for build_dscnn_model."""

    def test_precomputed_frontend(self):
        """Precomputed frontend should build and produce correct output shape."""
        model = build_dscnn_model(
            num_mels=64,
            spec_width=256,
            sample_rate=22050,
            chunk_duration=3,
            embeddings_size=128,
            num_classes=10,
            audio_frontend="librosa",
            alpha=0.25,
            depth_multiplier=1,
            mag_scale="none",
        )
        assert model.output_shape == (None, 10)

    def test_hybrid_frontend(self):
        """Hybrid frontend should build and produce correct output shape."""
        model = build_dscnn_model(
            num_mels=64,
            spec_width=256,
            sample_rate=22050,
            chunk_duration=3,
            embeddings_size=128,
            num_classes=10,
            audio_frontend="hybrid",
            alpha=0.25,
            depth_multiplier=1,
            fft_length=512,
            mag_scale="none",
        )
        assert model.output_shape == (None, 10)

    def test_raw_frontend(self):
        """Raw frontend should build with small input."""
        model = build_dscnn_model(
            num_mels=64,
            spec_width=256,
            sample_rate=16000,
            chunk_duration=2,
            embeddings_size=128,
            num_classes=5,
            audio_frontend="raw",
            alpha=0.25,
            depth_multiplier=1,
        )
        assert model.output_shape == (None, 5)

    def test_raw_frontend_exceeds_n6_limit(self):
        """Raw frontend with too-large input should raise ValueError."""
        with pytest.raises(ValueError, match="STM32N6 constraint"):
            build_dscnn_model(
                num_mels=64,
                spec_width=256,
                sample_rate=22050,
                chunk_duration=3,
                embeddings_size=128,
                num_classes=10,
                audio_frontend="raw",
            )

    def test_channel_alignment(self):
        """All conv layers should have channels as multiples of 8."""
        model = build_dscnn_model(
            num_mels=64,
            spec_width=256,
            sample_rate=22050,
            chunk_duration=3,
            embeddings_size=256,
            num_classes=10,
            audio_frontend="librosa",
            alpha=1.0,
            depth_multiplier=1,
        )
        for layer in model.layers:
            if hasattr(layer, "filters") and layer.filters is not None:
                assert layer.filters % 8 == 0, f"{layer.name} has {layer.filters} filters (not aligned to 8)"

    def test_alpha_scaling(self):
        """Larger alpha should produce more parameters."""
        small = build_dscnn_model(
            num_mels=64,
            spec_width=256,
            sample_rate=22050,
            chunk_duration=3,
            embeddings_size=128,
            num_classes=10,
            audio_frontend="librosa",
            alpha=0.25,
            depth_multiplier=1,
        )
        large = build_dscnn_model(
            num_mels=64,
            spec_width=256,
            sample_rate=22050,
            chunk_duration=3,
            embeddings_size=128,
            num_classes=10,
            audio_frontend="librosa",
            alpha=1.0,
            depth_multiplier=1,
        )
        assert large.count_params() > small.count_params()

    def test_depth_multiplier_scaling(self):
        """Larger depth_multiplier should produce more parameters."""
        shallow = build_dscnn_model(
            num_mels=64,
            spec_width=256,
            sample_rate=22050,
            chunk_duration=3,
            embeddings_size=128,
            num_classes=10,
            audio_frontend="librosa",
            alpha=0.25,
            depth_multiplier=1,
        )
        deep = build_dscnn_model(
            num_mels=64,
            spec_width=256,
            sample_rate=22050,
            chunk_duration=3,
            embeddings_size=128,
            num_classes=10,
            audio_frontend="librosa",
            alpha=0.25,
            depth_multiplier=2,
        )
        assert deep.count_params() > shallow.count_params()

    def test_sigmoid_activation(self):
        """Model with sigmoid activation should also build."""
        model = build_dscnn_model(
            num_mels=64,
            spec_width=256,
            sample_rate=22050,
            chunk_duration=3,
            embeddings_size=128,
            num_classes=10,
            audio_frontend="librosa",
            alpha=0.25,
            depth_multiplier=1,
            class_activation="sigmoid",
        )
        assert model.output_shape == (None, 10)

    def test_invalid_frontend(self):
        """Invalid frontend name should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid audio frontend"):
            build_dscnn_model(
                num_mels=64,
                spec_width=256,
                sample_rate=22050,
                chunk_duration=3,
                embeddings_size=128,
                num_classes=10,
                audio_frontend="invalid",
            )
