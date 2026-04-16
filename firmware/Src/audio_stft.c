/* SPDX-License-Identifier: Apache-2.0
 * Real-valued STFT using CMSIS-DSP.
 *
 * Produces a linear magnitude spectrogram for the BirdNET hybrid frontend.
 * Output layout: [fft_bins, spec_width, 1] in row-major (freq, time, channel).
 */

#include "audio_stft.h"
#include "arm_math.h"           /* CMSIS-DSP */
#include <string.h>
#include <math.h>

/* ---- Hann window --------------------------------------------------------- */
static void hann_window(float *win, uint32_t length)
{
    for (uint32_t i = 0; i < length; i++)
        win[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)(length - 1)));
}

void stft_magnitude(const float *audio, uint32_t chunk_samples,
                    uint32_t fft_length, uint32_t hop_length,
                    uint32_t spec_width, float *out)
{
    const uint32_t fft_bins = fft_length / 2 + 1;

    /* Allocate working buffers on the stack (512-point FFT = manageable).
     * For larger FFT sizes, move these to a static or heap buffer. */
    float window[512];          /* fft_length <= 512 */
    float frame[512];           /* windowed frame    */
    float fft_out[512 + 2];    /* complex output from rfft: fft_length+2 floats */

    /* Build Hann window once */
    hann_window(window, fft_length);

    /* CMSIS-DSP real-FFT instance */
    arm_rfft_fast_instance_f32 rfft;
    arm_rfft_fast_init_f32(&rfft, fft_length);

    /* Zero the output */
    memset(out, 0, fft_bins * spec_width * sizeof(float));

    for (uint32_t t = 0; t < spec_width; t++) {
        uint32_t start = t * hop_length;

        /* Copy and window the frame */
        for (uint32_t i = 0; i < fft_length; i++) {
            uint32_t idx = start + i;
            float sample = (idx < chunk_samples) ? audio[idx] : 0.0f;
            frame[i] = sample * window[i];
        }

        /* In-place real FFT.  Output is interleaved complex:
         *   fft_out[0] = DC real, fft_out[1] = Nyquist real,
         *   fft_out[2..] = interleaved (real, imag) for bins 1..(N/2-1) */
        arm_rfft_fast_f32(&rfft, frame, fft_out, 0 /* forward */);

        /* Compute magnitude for each frequency bin and store column-wise.
         * We write in frequency-major order: out[f * spec_width + t]. */

        /* Bin 0 (DC): magnitude is |fft_out[0]| */
        out[0 * spec_width + t] = fabsf(fft_out[0]);

        /* Bins 1 .. fft_length/2 - 1 */
        for (uint32_t f = 1; f < fft_bins - 1; f++) {
            float re = fft_out[2 * f];
            float im = fft_out[2 * f + 1];
            float mag;
            arm_sqrt_f32(re * re + im * im, &mag);
            out[f * spec_width + t] = mag;
        }

        /* Bin fft_length/2 (Nyquist): magnitude is |fft_out[1]| */
        out[(fft_bins - 1) * spec_width + t] = fabsf(fft_out[1]);
    }
}
