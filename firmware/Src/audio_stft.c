/* SPDX-License-Identifier: Apache-2.0
 * Real-valued STFT using a plain-C 512-point FFT.
 *
 * Produces a linear magnitude spectrogram for the BirdNET hybrid frontend.
 * Output layout: [fft_bins, spec_width, 1] in row-major (freq, time, channel).
 *
 * This must compute what the host computes -- get_spectrogram_from_audio() in
 * birdnet_stm32/audio/spectrogram.py, i.e. librosa.stft(center=True,
 * pad_mode="constant", window="hann") followed by a per-sample min-max
 * normalization -- or the model sees a different input on the device than it
 * was trained and evaluated on. tests/test_firmware_stft.py compiles this file
 * natively and checks it against the host.
 */

#include "audio_stft.h"
#include "fft.h"
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

    /* Working buffers on the stack (512-point FFT = 2 KB, manageable). */
    float window[512];          /* fft_length <= 512 */
    float fft_buf[512];         /* input/output for fft_512_real */

    /* Build Hann window once */
    hann_window(window, fft_length);

    /* Zero the output */
    memset(out, 0, fft_bins * spec_width * sizeof(float));

    for (uint32_t t = 0; t < spec_width; t++) {
        /* Centered frames, as librosa.stft(center=True, pad_mode="constant"):
         * frame t is centred on sample t * hop, with zeros outside the chunk.
         * Frames that start at t * hop instead are shifted by half a window
         * and agree with the host at cos 0.32, measured. */
        int32_t start = (int32_t)(t * hop_length) - (int32_t)(fft_length / 2);

        /* Copy and window the frame */
        for (uint32_t i = 0; i < fft_length; i++) {
            int32_t idx = start + (int32_t)i;
            float sample = (idx >= 0 && idx < (int32_t)chunk_samples) ? audio[idx] : 0.0f;
            fft_buf[i] = sample * window[i];
        }

        /* In-place real FFT.  Output is packed:
         *   fft_buf[0] = DC real, fft_buf[1] = Nyquist real,
         *   fft_buf[2..] = interleaved (real, imag) for bins 1..(N/2-1) */
        fft_512_real(fft_buf);

        /* Compute magnitude for each frequency bin and store column-wise.
         * We write in frequency-major order: out[f * spec_width + t]. */

        /* Bin 0 (DC): magnitude is |fft_buf[0]| */
        out[0 * spec_width + t] = fabsf(fft_buf[0]);

        /* Bins 1 .. fft_length/2 - 1. Nyquist is intentionally omitted. */
        for (uint32_t f = 1; f < fft_bins; f++) {
            float re = fft_buf[2 * f];
            float im = fft_buf[2 * f + 1];
            out[f * spec_width + t] = sqrtf(re * re + im * im);
        }
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
