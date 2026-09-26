/* SPDX-License-Identifier: MIT
 * BirdNET Tiny N6 — reference audio frontend in C. See birdnet_frontend.h.
 *
 * The FFT and the complex magnitude are CMSIS-DSP's arm_rfft_fast_f32 and
 * arm_cmplx_mag_f32 (firmware/Drivers/CMSIS-DSP). On a Cortex-M55 build with
 * ARM_MATH_HELIUM they use the vector extension; on a host, define
 * __GNUC_PYTHON__ and CMSIS-DSP compiles its portable C path. Any FFT that
 * returns the same unscaled spectrum can replace them.
 */

#include "birdnet_frontend.h"

#include <math.h>

#include "arm_math.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

uint32_t bn_window_start(uint32_t num_samples, uint32_t size, uint32_t step, uint32_t index, uint32_t *start)
{
    if (num_samples <= size) {
        if (index == 0 && start) *start = 0;
        return 1;
    }
    /* Regular starts 0, step, 2*step, ... while the window fits ... */
    uint32_t regular = (num_samples - size) / step + 1;
    /* ... then one more, right-aligned, if the regular ones stop short. */
    uint32_t last_end = (regular - 1) * step + size;
    uint32_t count = regular + (last_end < num_samples ? 1 : 0);
    if (index < count && start) *start = index < regular ? index * step : num_samples - size;
    return count;
}

void bn_peak_normalize(float *x, uint32_t n)
{
    float peak = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > peak) peak = a;
    }
    float scale = 1.0f / (peak + 1e-6f);
    for (uint32_t i = 0; i < n; i++) x[i] *= scale;
}

int bn_stft_magnitude(const float *window, uint32_t n, uint32_t fft_length, uint32_t spec_width, float *out)
{
    static float hann[BN_MAX_FFT];
    static float frame[BN_MAX_FFT];    /* arm_rfft_fast_f32 overwrites its input */
    static float spectrum[BN_MAX_FFT]; /* packed: [0] DC, [1] Nyquist, then (re, im) of bins 1..N/2-1 */
    static float mag[BN_MAX_FFT / 2];
    static arm_rfft_fast_instance_f32 rfft;
    static uint32_t ready = 0;

    if (fft_length < 32 || fft_length > BN_MAX_FFT || (fft_length & (fft_length - 1)) || spec_width == 0)
        return -1;
    if (ready != fft_length) {
        if (arm_rfft_fast_init_f32(&rfft, (uint16_t)fft_length) != ARM_MATH_SUCCESS) return -1;
        /* Step 3.3: periodic Hann — divide by N, not N - 1. */
        for (uint32_t i = 0; i < fft_length; i++)
            hann[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)fft_length));
        ready = fft_length;
    }

    const uint32_t bins = fft_length / 2;
    const uint32_t hop = n / spec_width; /* step 3.1 */
    for (uint32_t t = 0; t < spec_width; t++) {
        /* Step 3.2: frame t is centred on sample t*hop; zeros outside the window. */
        int32_t start = (int32_t)(t * hop) - (int32_t)(fft_length / 2);
        for (uint32_t i = 0; i < fft_length; i++) {
            int32_t idx = start + (int32_t)i;
            float sample = (idx >= 0 && idx < (int32_t)n) ? window[idx] : 0.0f;
            frame[i] = sample * hann[i];
        }
        /* Step 3.4: unscaled real FFT, magnitude of bins 0 .. N/2 - 1. */
        arm_rfft_fast_f32(&rfft, frame, spectrum, 0);
        mag[0] = fabsf(spectrum[0]);
        arm_cmplx_mag_f32(spectrum + 2, mag + 1, bins - 1);
        for (uint32_t f = 0; f < bins; f++) out[f * spec_width + t] = mag[f];
    }
    return 0;
}

void bn_compress(float *spec, uint32_t count, enum bn_compression mode)
{
    if (mode == BN_COMPRESS_SQRT) {
        for (uint32_t i = 0; i < count; i++) spec[i] = sqrtf(spec[i]);
    } else if (mode == BN_COMPRESS_LOG) {
        float peak = 0.0f;
        for (uint32_t i = 0; i < count; i++)
            if (spec[i] > peak) peak = spec[i];
        float floor_value = (peak > 1e-10f ? peak : 1e-10f) * powf(10.0f, -80.0f / 20.0f);
        for (uint32_t i = 0; i < count; i++) spec[i] = logf(spec[i] > floor_value ? spec[i] : floor_value);
    }
}

void bn_minmax_normalize(float *spec, uint32_t count)
{
    float lo = spec[0], hi = spec[0];
    for (uint32_t i = 1; i < count; i++) {
        if (spec[i] < lo) lo = spec[i];
        if (spec[i] > hi) hi = spec[i];
    }
    float scale = 1.0f / (hi - lo + 1e-10f);
    for (uint32_t i = 0; i < count; i++) spec[i] = (spec[i] - lo) * scale;
}
