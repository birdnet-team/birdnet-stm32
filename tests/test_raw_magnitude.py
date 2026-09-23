"""Tests for the raw frontend's interchangeable magnitude stages.

Raw's INT8 loss tracks the number of 8-bit activation grids its frontend puts
between the filterbank and the backbone, not the arithmetic of any one of them
(dev/ssw_magpie_rt_model.md, Phase C1). These options shorten that chain: 11
elementwise ops for ``alpha_max``, 7 for ``l1``, 1 for ``halfwave``.
"""

import numpy as np
import pytest

from birdnet_stm32.training.config import ModelConfig

tf = pytest.importorskip("tensorflow", reason="TensorFlow required")

from birdnet_stm32.models.frontend import VALID_RAW_MAGNITUDES, AudioFrontendLayer  # noqa: E402


def _frontend(raw_magnitude):
    return AudioFrontendLayer(
        mode="raw",
        mel_bins=16,
        spec_width=32,
        sample_rate=8000,
        chunk_duration=1,
        mag_scale="none",
        raw_magnitude=raw_magnitude,
    )


def _waveform(n=2, samples=8000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, samples, 1)).astype(np.float32) * 0.3


class TestConfigField:
    def test_defaults_to_the_shipped_stage(self):
        assert ModelConfig().raw_magnitude == "alpha_max"
        assert ModelConfig.from_dict({"num_classes": 0}).raw_magnitude == "alpha_max"

    def test_accepts_every_frontend_option(self):
        for name in VALID_RAW_MAGNITUDES:
            assert ModelConfig(audio_frontend="raw", raw_magnitude=name).raw_magnitude == name

    def test_rejects_an_unknown_stage(self):
        with pytest.raises(ValueError, match="raw_magnitude"):
            ModelConfig(audio_frontend="raw", raw_magnitude="rms")

    def test_rejects_a_non_default_stage_on_a_spectrogram_frontend(self):
        """Silently ignoring it would make the config lie about the model."""
        with pytest.raises(ValueError, match="raw frontend"):
            ModelConfig(audio_frontend="hybrid", raw_magnitude="l1")

    def test_round_trips_through_json(self, tmp_path):
        path = tmp_path / "cfg.json"
        ModelConfig(audio_frontend="raw", raw_magnitude="halfwave").save(path)
        assert ModelConfig.load(path).raw_magnitude == "halfwave"


class TestEnvelope:
    def test_every_stage_is_non_negative_and_shaped_like_a_spectrogram(self):
        x = _waveform()
        for name in VALID_RAW_MAGNITUDES:
            y = _frontend(name)(x, training=False).numpy()
            assert y.shape == (2, 16, 32, 1), name
            assert y.min() >= 0.0, name

    def test_l1_tracks_alpha_max(self):
        """Both approximate the same modulus, so they must stay correlated.

        On white noise through an untrained bank, which is the worst case for
        L1's carrier ripple; on the trained B3 filterbank over real recordings
        the per-band correlation with the true modulus is 0.998.
        """
        x = _waveform()
        a = _frontend("alpha_max")(x, training=False).numpy().ravel()
        b = _frontend("l1")(x, training=False).numpy().ravel()
        assert np.corrcoef(a, b)[0, 1] > 0.97

    def test_every_stage_is_homogeneous_so_equalization_stays_valid(self):
        """Equalize rescales the filterbank; the envelope must scale with it."""
        x = _waveform()
        for name in VALID_RAW_MAGNITUDES:
            fe = _frontend(name)
            one = fe(x, training=False).numpy()
            two = fe(2.0 * x, training=False).numpy()
            np.testing.assert_allclose(two, 2.0 * one, rtol=1e-4, atol=1e-5)

    def test_the_stage_survives_a_config_round_trip(self):
        fe = _frontend("halfwave")
        assert fe.get_config()["raw_magnitude"] == "halfwave"
        assert AudioFrontendLayer.from_config(fe.get_config()).raw_magnitude == "halfwave"

    def test_an_unknown_stage_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="raw_magnitude"):
            _frontend("rms")


class TestOpCount:
    """What the options are for: fewer INT8 activation grids in the frontend."""

    def _grids(self, raw_magnitude):
        seen = []

        class Hook:
            def kernel(self, layer, inputs):
                return layer(inputs)

            def activation(self, name, inputs):
                seen.append(name)
                return inputs

        fe = _frontend(raw_magnitude)
        fe.set_quantization_hook(Hook())
        fe(_waveform(n=1), training=False)
        return [n for n in seen if "fb_re" not in n and "fb_im" not in n]

    def test_each_option_shortens_the_chain(self):
        counts = {name: len(self._grids(name)) for name in VALID_RAW_MAGNITUDES}
        assert counts["halfwave"] < counts["l1"] < counts["alpha_max"]

    def test_all_stages_name_the_tensor_equalization_watches(self):
        """`equalize` measures band spread at `<frontend>_magnitude`."""
        for name in VALID_RAW_MAGNITUDES:
            assert "audio_frontend_magnitude" in self._grids(name), name


class TestOverlap:
    """The analysis window, and how many partial convolutions it becomes."""

    def test_the_default_geometry_is_unchanged(self):
        """Every shipped model must keep the graph it was validated on."""
        from birdnet_stm32.models.frontend import raw_filterbank_geometry, raw_filterbank_split

        for samples, width in [(60000, 256), (72000, 256), (60000, 384), (48000, 256)]:
            geom = raw_filterbank_geometry(samples, width)
            assert raw_filterbank_split(geom) == 4, (samples, width)

    def test_each_partial_convolution_accumulates_one_fold_of_taps(self):
        """112 taps at the release geometry: the count measured on the NPU."""
        from birdnet_stm32.models.frontend import raw_filterbank_geometry, raw_filterbank_split

        for overlap in (1, 2):
            geom = raw_filterbank_geometry(60000, 256, overlap=overlap)
            split = raw_filterbank_split(geom)
            assert geom.window // split == geom.fold, overlap
            assert geom.fold % split == 0, overlap

    def test_an_overlap_that_does_not_divide_the_fold_is_refused(self):
        """Better a build error than a filterbank with unequal groups."""
        with pytest.raises(ValueError, match="filterbank split"):
            AudioFrontendLayer(
                mode="raw",
                mel_bins=16,
                spec_width=256,
                sample_rate=24000,
                chunk_duration=2.5,
                mag_scale="none",
                raw_overlap=3,
            )

    def test_a_shorter_window_emits_fewer_convolutions(self):
        fe = AudioFrontendLayer(
            mode="raw",
            mel_bins=16,
            spec_width=32,
            sample_rate=8000,
            chunk_duration=1,
            mag_scale="none",
            raw_overlap=1,
        )
        assert len(fe.fb_re) == 2 and len(fe.fb_im) == 2
        assert fe(_waveform(n=1), training=False).shape == (1, 16, 32, 1)

    def test_it_survives_a_config_round_trip(self):
        fe = AudioFrontendLayer(
            mode="raw",
            mel_bins=16,
            spec_width=32,
            sample_rate=8000,
            chunk_duration=1,
            mag_scale="none",
            raw_overlap=1,
        )
        clone = AudioFrontendLayer.from_config(fe.get_config())
        assert clone.raw_overlap == 1
        assert clone.geom == fe.geom

    def test_config_rejects_a_non_default_overlap_on_a_spectrogram_frontend(self):
        with pytest.raises(ValueError, match="raw filterbank"):
            ModelConfig(audio_frontend="hybrid", raw_overlap=1)


class TestFusedBank:
    """One bank of 2*mel_bins filters in place of two banks of mel_bins."""

    def _layer(self, raw_bank, **kwargs):
        return AudioFrontendLayer(
            mode="raw",
            mel_bins=16,
            spec_width=32,
            sample_rate=8000,
            chunk_duration=1,
            mag_scale="none",
            raw_bank=raw_bank,
            **kwargs,
        )

    def test_it_computes_exactly_what_the_pair_computes(self):
        """Same filters, same sums, one graph: the output must be bit-equal."""
        x = _waveform()
        pair = self._layer("pair")(x, training=False).numpy()
        fused = self._layer("fused")(x, training=False).numpy()
        np.testing.assert_array_equal(pair, fused)

    def test_it_halves_the_convolutions(self):
        assert len(self._layer("fused").filterbank_convs()) * 2 == len(self._layer("pair").filterbank_convs())

    def test_per_band_equalization_reaches_both_components(self):
        """`equalize` scales bands through this method; a fused kernel holds
        the two components side by side and both halves must move."""
        x = _waveform()
        gains = np.linspace(0.5, 2.0, 16).astype(np.float32)
        out = []
        for raw_bank in ("pair", "fused"):
            fe = self._layer(raw_bank)
            fe(x, training=False)
            fe.scale_filterbank_bands(gains)
            out.append(fe(x, training=False).numpy())
        np.testing.assert_array_equal(out[0], out[1])

    def test_a_wrong_gain_shape_is_refused(self):
        with pytest.raises(ValueError, match="one gain per band"):
            self._layer("fused").scale_filterbank_bands(np.ones(3, dtype=np.float32))

    def test_it_keeps_the_names_equalization_and_qat_watch(self):
        seen = []

        class Hook:
            def kernel(self, layer, inputs):
                return layer(inputs)

            def activation(self, name, inputs):
                seen.append(name)
                return inputs

        fe = self._layer("fused")
        fe.set_quantization_hook(Hook())
        fe(_waveform(n=1), training=False)
        assert "audio_frontend_fb_re" in seen and "audio_frontend_fb_im" in seen

    def test_it_survives_a_config_round_trip(self):
        fe = self._layer("fused")
        assert AudioFrontendLayer.from_config(fe.get_config()).raw_bank == "fused"

    def test_config_rejects_it_on_a_spectrogram_frontend(self):
        with pytest.raises(ValueError, match="raw filterbank"):
            ModelConfig(audio_frontend="hybrid", raw_bank="fused")

    def test_an_unknown_layout_is_refused(self):
        with pytest.raises(ValueError, match="raw_bank"):
            self._layer("complex")
