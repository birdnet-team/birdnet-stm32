/* SPDX-License-Identifier: Apache-2.0
 * Real-valued STFT on the Cortex-M55, through CMSIS-DSP.
 *
 * Produces a linear magnitude spectrogram for the BirdNET hybrid frontend.
 * Output layout: [fft_bins, spec_width, 1] in row-major (freq, time, channel).
 *
 * This must compute what the host computes -- get_spectrogram_from_audio() in
 * birdnet_stm32/audio/spectrogram.py, i.e. librosa.stft(center=True,
 * pad_mode="constant", window="hann") followed by a per-sample min-max
 * normalization -- or the model sees a different input on the device than it
 * was trained and evaluated on. tests/test_firmware_stft.py compiles this file
 * natively and checks it against the host; reference/ specifies it step by step.
 *
 * The FFT and the magnitude are CMSIS-DSP's arm_rfft_fast_f32 and
 * arm_cmplx_mag_f32, vectorized for Helium when built with ARM_MATH_HELIUM
 * (the firmware build does). Measured on the STM32N6570-DK at 400 MHz, the STFT
 * of a 2.5 s window into 384 frames takes 33 ms, against 69 ms for the scalar
 * radix-2 FFT it replaced; the board's scores are unchanged.
 */

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("O3")
#endif

#include "audio_stft.h"
#include "arm_math.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* The output is frequency-major, so one frame's magnitudes land one row apart.
 * Collecting STFT_TILE frames first turns those scattered stores into runs of
 * STFT_TILE consecutive floats (one 32-byte cache line). */
#define STFT_TILE 8
#define STFT_MAX_FFT 512

/* ---- Hann window --------------------------------------------------------- */
/* Periodic, as librosa's "hann" (scipy get_window with fftbins=True). */
static void hann_window(float *win, uint32_t length)
{
    for (uint32_t i = 0; i < length; i++)
        win[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)length));
}

void stft_magnitude(const float *audio, uint32_t chunk_samples,
                    uint32_t fft_length, uint32_t hop_length,
                    uint32_t spec_width, float *out)
{
    /* The model contract drops Nyquist, matching hybrid_fft_bins() in Python. */
    const uint32_t fft_bins = fft_length / 2;

    /* Static, not on the stack: together ~12 KB, most of the firmware's 16 KB
     * stack. The firmware calls this from one thread only. */
    static float window[STFT_MAX_FFT];
    static float frame[STFT_MAX_FFT];     /* windowed input; arm_rfft_fast_f32 overwrites it */
    static float spectrum[STFT_MAX_FFT];  /* packed real FFT output */
    static float mag[STFT_TILE][STFT_MAX_FFT / 2];
    static arm_rfft_fast_instance_f32 rfft;
    static uint32_t ready_length = 0;

    if (ready_length != fft_length) {
        arm_rfft_fast_init_f32(&rfft, (uint16_t)fft_length);
        hann_window(window, fft_length);
        ready_length = fft_length;
    }

    for (uint32_t t0 = 0; t0 < spec_width; t0 += STFT_TILE) {
        uint32_t n = spec_width - t0 < STFT_TILE ? spec_width - t0 : STFT_TILE;
        for (uint32_t j = 0; j < n; j++) {
            /* Centered frames, as librosa.stft(center=True, pad_mode="constant"):
             * frame t is centred on sample t * hop, with zeros outside the chunk.
             * Frames that start at t * hop instead are shifted by half a window
             * and agree with the host at cos 0.32, measured. */
            int32_t start = (int32_t)((t0 + j) * hop_length) - (int32_t)(fft_length / 2);
            if (start >= 0 && start + (int32_t)fft_length <= (int32_t)chunk_samples) {
                for (uint32_t i = 0; i < fft_length; i++)
                    frame[i] = audio[start + (int32_t)i] * window[i];
            } else {
                for (uint32_t i = 0; i < fft_length; i++) {
                    int32_t idx = start + (int32_t)i;
                    float sample = (idx >= 0 && idx < (int32_t)chunk_samples) ? audio[idx] : 0.0f;
                    frame[i] = sample * window[i];
                }
            }

            /* Packed output: [0] = DC real, [1] = Nyquist real,
             * [2k], [2k+1] = real, imag of bin k for k = 1 .. N/2 - 1. */
            arm_rfft_fast_f32(&rfft, frame, spectrum, 0);
            mag[j][0] = fabsf(spectrum[0]);
            /* Bins 1 .. fft_length/2 - 1. Nyquist is intentionally omitted. */
            arm_cmplx_mag_f32(spectrum + 2, mag[j] + 1, fft_bins - 1);
        }
        for (uint32_t f = 0; f < fft_bins; f++)
            for (uint32_t j = 0; j < n; j++)
                out[f * spec_width + t0 + j] = mag[j][f];
    }
}

void spec_compress(float *spec, uint32_t count, int mode)
{
    if (mode == SPEC_COMPRESS_SQRT) {
        for (uint32_t i = 0; i < count; i++)
            spec[i] = sqrtf(spec[i]);
    } else if (mode == SPEC_COMPRESS_LOG) {
        float peak = 0.0f;
        for (uint32_t i = 0; i < count; i++)
            if (spec[i] > peak) peak = spec[i];
        float floor_value = (peak > 1e-10f ? peak : 1e-10f) * powf(10.0f, -SPEC_LOG_FLOOR_DB / 20.0f);
        for (uint32_t i = 0; i < count; i++)
            spec[i] = logf(spec[i] > floor_value ? spec[i] : floor_value);
    }
}

void spec_minmax_normalize(float *spec, uint32_t count)
{
    /* (S - min) / (max - min + 1e-10), as normalize() on the host. */
    float lo = spec[0], hi = spec[0];
    for (uint32_t i = 1; i < count; i++) {
        if (spec[i] < lo) lo = spec[i];
        if (spec[i] > hi) hi = spec[i];
    }
    float scale = 1.0f / (hi - lo + 1e-10f);
    for (uint32_t i = 0; i < count; i++)
        spec[i] = (spec[i] - lo) * scale;
}
