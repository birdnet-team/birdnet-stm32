"""Unit tests for AudioFrontendLayer."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow required for frontend tests")

from birdnet_stm32.models.frontend import (
    AudioFrontendLayer,
    hybrid_fft_bins,
    normalize_frontend_name,
    raw_filterbank_geometry,
)


@pytest.fixture
def frontend_params():
    """Common frontend parameters."""
    return dict(
        mel_bins=64,
        spec_width=256,
        sample_rate=22050,
        chunk_duration=3,
        fft_length=512,
        mag_scale="none",
    )


class TestPrecomputedMode:
    """Tests for precomputed (pass-through) frontend."""

    def test_output_shape(self, frontend_params):
        """Output shape should match (B, mel_bins, spec_width, 1)."""
        layer = AudioFrontendLayer(mode="precomputed", **frontend_params)
        x = tf.random.uniform((2, 64, 300, 1))
        y = layer(x)
        assert y.shape == (2, 64, 256, 1)

    def test_passthrough_values(self, frontend_params):
        """Without mag_scale, values should pass through with only slicing."""
        layer = AudioFrontendLayer(mode="precomputed", **frontend_params)
        x = tf.ones((1, 64, 256, 1))
        y = layer(x)
        np.testing.assert_allclose(y.numpy(), 1.0, atol=1e-5)


class TestHybridMode:
    """Tests for hybrid (STFT + mel mixer) frontend."""

    def test_output_shape(self, frontend_params):
        """Output should be (B, mel_bins, spec_width, 1)."""
        fft_bins = hybrid_fft_bins(frontend_params["fft_length"])
        layer = AudioFrontendLayer(mode="hybrid", **frontend_params)
        x = tf.random.uniform((2, fft_bins, 256, 1))
        y = layer(x)
        assert y.shape[0] == 2
        assert y.shape[1] == frontend_params["mel_bins"]
        assert y.shape[2] == frontend_params["spec_width"]
        assert y.shape[3] == 1

    def test_mel_mixer_initialized(self, frontend_params):
        """Mel mixer kernel should be initialized from librosa basis."""
        layer = AudioFrontendLayer(mode="hybrid", init_mel=True, **frontend_params)
        fft_bins = hybrid_fft_bins(frontend_params["fft_length"])
        x = tf.random.uniform((1, fft_bins, 256, 1))
        _ = layer(x)  # build
        weights = layer.get_weights()
        # Should have non-zero weights (mel basis)
        assert any(np.any(w != 0) for w in weights)


class TestRawMode:
    """Tests for raw (waveform) frontend."""

    def test_output_shape(self, frontend_params):
        """Raw frontend should produce (B, mel_bins, spec_width, 1)."""
        params = {**frontend_params, "sample_rate": 16000, "chunk_duration": 2}
        T = params["sample_rate"] * params["chunk_duration"]
        layer = AudioFrontendLayer(mode="raw", **params)
        x = tf.random.uniform((2, T, 1))
        y = layer(x)
        assert y.shape[0] == 2
        assert y.shape[-1] == 1

    def test_pwl_output_is_nonnegative_and_finite(self, frontend_params):
        """Raw frontend with PWL scaling should produce valid magnitudes."""
        params = {**frontend_params, "sample_rate": 16000, "chunk_duration": 2, "mag_scale": "pwl"}
        T = params["sample_rate"] * params["chunk_duration"]
        layer = AudioFrontendLayer(mode="raw", **params)
        x = tf.random.uniform((2, T, 1), minval=-1.0, maxval=1.0)
        y = layer(x)
        assert bool(tf.reduce_all(tf.math.is_finite(y)).numpy())
        assert float(tf.reduce_min(y).numpy()) >= 0.0


class TestMagScaling:
    """Tests for magnitude scaling modes."""

    def test_pwl_output_shape(self, frontend_params):
        """PWL scaling should preserve shape."""
        params = {**frontend_params, "mag_scale": "pwl"}
        layer = AudioFrontendLayer(mode="precomputed", **params)
        x = tf.random.uniform((1, 64, 256, 1))
        y = layer(x)
        assert y.shape == (1, 64, 256, 1)

    def test_pcen_output_shape(self, frontend_params):
        """PCEN scaling should preserve shape."""
        params = {**frontend_params, "mag_scale": "pcen"}
        layer = AudioFrontendLayer(mode="precomputed", **params)
        x = tf.random.uniform((1, 64, 256, 1))
        y = layer(x)
        assert y.shape == (1, 64, 256, 1)

    def test_none_output_shape(self, frontend_params):
        """No scaling should preserve shape."""
        layer = AudioFrontendLayer(mode="precomputed", **frontend_params)
        x = tf.random.uniform((1, 64, 256, 1))
        y = layer(x)
        assert y.shape == (1, 64, 256, 1)


class TestSerializationRoundtrip:
    """Test that the layer config can be roundtripped."""

    def test_get_config(self, frontend_params):
        """get_config should return a valid config dict."""
        layer = AudioFrontendLayer(mode="precomputed", **frontend_params)
        config = layer.get_config()
        assert config["mode"] == "precomputed"
        assert config["mel_bins"] == 64
        assert config["spec_width"] == 256


class TestNormalizeFrontendName:
    """Tests for normalize_frontend_name."""

    def test_canonical_names_pass_through(self):
        """Canonical names should be returned as-is."""
        assert normalize_frontend_name("librosa") == "librosa"
        assert normalize_frontend_name("hybrid") == "hybrid"
        assert normalize_frontend_name("raw") == "raw"

    def test_removed_aliases_raise(self):
        """The old 'precomputed'/'tf' aliases are no longer accepted."""
        for legacy in ("precomputed", "tf"):
            with pytest.raises(ValueError, match="Invalid audio frontend"):
                normalize_frontend_name(legacy)

    def test_invalid_name_raises(self):
        """Invalid frontend name should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid audio frontend"):
            normalize_frontend_name("invalid")


class TestFrontendTrainability:
    """The is_trainable flag must reach every frontend sub-layer."""

    def _hybrid(self, is_trainable):
        return AudioFrontendLayer(
            mode="hybrid",
            mel_bins=16,
            spec_width=32,
            sample_rate=24000,
            chunk_duration=3,
            fft_length=64,
            mag_scale="pwl",
            is_trainable=is_trainable,
        )

    def test_hybrid_mel_mixer_follows_flag(self):
        """The mel mixer is frozen by default and trainable when asked.

        It used to be hard-wired to frozen, so --frontend_trainable silently
        did nothing for the hybrid frontend.
        """
        assert self._hybrid(False).mel_mixer.trainable is False
        assert self._hybrid(True).mel_mixer.trainable is True

    def test_trainable_hybrid_exposes_mixer_weights(self):
        """A trainable frontend reports the mixer kernel as a trainable weight."""
        layer = self._hybrid(True)
        layer.build(tf.TensorShape([None, hybrid_fft_bins(64), 32, 1]))
        assert any(w.path.endswith("kernel") for w in layer.mel_mixer.trainable_weights)

    def test_frozen_hybrid_has_no_mixer_weights(self):
        """A frozen frontend contributes no trainable mixer weights."""
        layer = self._hybrid(False)
        layer.build(tf.TensorShape([None, hybrid_fft_bins(64), 32, 1]))
        assert layer.mel_mixer.trainable_weights == []


class TestRawFilterbankGeometry:
    """The raw filterbank must cover the signal and stay NPU-mappable."""

    CONFIGS = [(24000, 2.5, 256), (24000, 2, 256), (24000, 3, 256), (48000, 2.5, 256), (16000, 2, 128)]

    @pytest.mark.parametrize(("sr", "cd", "width"), CONFIGS)
    def test_every_sample_is_read(self, sr, cd, width):
        """window >= hop, so no input sample falls between two frames.

        The previous filterbank used a fixed 16-tap kernel against a hop of a
        few hundred samples, so it never saw ~94% of the waveform.
        """
        g = raw_filterbank_geometry(int(sr * cd), width)
        assert g.window >= g.hop

    @pytest.mark.parametrize(("sr", "cd", "width"), CONFIGS)
    def test_stride_two_and_aligned_channels(self, sr, cd, width):
        """Folded stride is 2 and the fold is a full vector group.

        Measured on stedgeai: stride 2 keeps the convolution on the NPU while
        stride 3 pushes it to the Cortex-M55.
        """
        g = raw_filterbank_geometry(int(sr * cd), width)
        assert g.stride == 2
        assert g.fold % 8 == 0

    @pytest.mark.parametrize(("sr", "cd", "width"), CONFIGS)
    def test_emits_enough_frames(self, sr, cd, width):
        """The bank must produce at least spec_width frames without padding."""
        g = raw_filterbank_geometry(int(sr * cd), width)
        assert g.frames >= width
        assert g.crop % g.fold == 0

    def test_geometry_is_self_consistent(self):
        """window, fold, kernel and hop describe the same convolution."""
        g = raw_filterbank_geometry(60000, 256)
        assert g.window == g.kernel * g.fold
        assert g.hop == g.stride * g.fold

    def test_impossible_config_raises(self):
        """An unsatisfiable frame count fails loudly rather than silently."""
        with pytest.raises(ValueError, match="Cannot fit"):
            raw_filterbank_geometry(4000, 4096)


class TestRawFilterbankBehaviour:
    """Signal-level properties of the rewritten raw frontend."""

    def _layer(self, sr=24000, cd=2.5, width=256, mel=64):
        layer = AudioFrontendLayer(
            mode="raw",
            mel_bins=mel,
            spec_width=width,
            sample_rate=sr,
            chunk_duration=cd,
            fft_length=512,
            mag_scale="none",
        )
        layer.build(tf.TensorShape([None, int(sr * cd), 1]))
        return layer

    def _modulus(self, layer, wav):
        g = layer.geom
        y = tf.reshape(tf.constant(wav[None, : g.crop, None], tf.float32), [-1, 1, g.crop // g.fold, g.fold])
        a, b = tf.abs(layer.fb_re(y)), tf.abs(layer.fb_im(y))
        return (tf.maximum(a, b) + 0.4 * tf.minimum(a, b)).numpy()[0, 0]

    def test_responds_to_every_region_of_the_input(self):
        """Perturbing any sample changes the output — nothing is skipped."""
        layer = self._layer()
        g = layer.geom
        rng = np.random.default_rng(0)
        wav = (rng.standard_normal(g.crop) * 0.1).astype(np.float32)
        base = self._modulus(layer, wav)

        for offset in (0, g.hop // 3, g.hop // 2, g.hop - 1):
            probe = wav.copy()
            probe[[i * g.hop + offset for i in range(1, 100)]] += 1.0
            assert np.abs(self._modulus(layer, probe) - base).max() > 1e-3, f"offset {offset} ignored"

    def test_tuned_to_the_right_frequencies(self):
        """A pure tone excites the band whose centre frequency matches it."""
        layer = self._layer()
        sr, n = 24000, layer.geom.crop
        t = np.arange(n) / sr
        peaks = []
        for freq in (500.0, 2000.0, 6000.0):
            tone = np.sin(2 * np.pi * freq * t).astype(np.float32)
            peaks.append(int(self._modulus(layer, tone).mean(axis=0).argmax()))
        # Higher tones must peak in strictly higher mel bands.
        assert peaks[0] < peaks[1] < peaks[2], peaks

    def test_no_global_reduction_over_the_batch(self):
        """Each item is scaled independently of the rest of the batch.

        The old per-sample max normalization also made the graph depend on a
        global reduction, which the N6 ran in software.
        """
        layer = self._layer()
        rng = np.random.default_rng(1)
        a = (rng.standard_normal(60000) * 0.1).astype(np.float32)
        b = (rng.standard_normal(60000) * 0.1).astype(np.float32)

        alone = layer(tf.constant(a[None, :, None]), training=False).numpy()
        # Pair it with a much louder signal; `alone` must be unaffected.
        together = layer(tf.constant(np.stack([a, b * 50.0])[..., None]), training=False).numpy()[0]
        np.testing.assert_allclose(alone[0], together, rtol=1e-4, atol=1e-5)

    def test_smoothing_kernel_is_a_normalized_lowpass(self):
        """The per-band temporal smoother starts as a unit-sum Hann window."""
        layer = self._layer()
        taps = layer.band_smooth.get_weights()[0][0, :, 0, 0]
        assert taps.sum() == pytest.approx(1.0, rel=1e-5)
        assert taps.argmax() == len(taps) // 2
