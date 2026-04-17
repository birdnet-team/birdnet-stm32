/* SPDX-License-Identifier: Apache-2.0
 * BirdNET-STM32 — Mel filterbank for precomputed frontend.
 *
 * Builds a Slaney-normalized triangular mel weight matrix and applies it to
 * a linear STFT magnitude spectrogram.  Matches librosa's mel filterbank
 * (norm="slaney", htk=False) used in the Python training pipeline.
 */

#include "audio_mel.h"
#include <math.h>
#include <string.h>

/* Maximum supported dimensions.  Keep small to fit in static RAM. */
#define MAX_FFT_BINS   257
#define MAX_MEL_BINS   128

/* Mel weight matrix: mel_weights[m * MAX_FFT_BINS + f] = weight for
 * mel band m at FFT bin f.  Sparse but we store dense for simplicity. */
static float mel_weights[MAX_MEL_BINS * MAX_FFT_BINS];
static uint32_t mel_n_mels = 0;
static uint32_t mel_n_fft_bins = 0;

/* Hz ↔ mel conversions (Slaney / O'Shaughnessy, same as librosa default) */
static float hz_to_mel(float hz)
{
    /* Slaney's formula: linear below 1000 Hz, log above */
    const float f_sp = 200.0f / 3.0f;  /* 66.667 Hz */
    if (hz < 1000.0f)
        return hz / f_sp;
    const float min_log_hz = 1000.0f;
    const float min_log_mel = min_log_hz / f_sp;  /* 15.0 */
    const float logstep = logf(6.4f) / 27.0f;     /* log(6400/1000) / 27 */
    return min_log_mel + logf(hz / min_log_hz) / logstep;
}

static float mel_to_hz(float mel)
{
    const float f_sp = 200.0f / 3.0f;
    if (mel < 15.0f)
        return mel * f_sp;
    const float min_log_hz = 1000.0f;
    const float min_log_mel = 15.0f;
    const float logstep = logf(6.4f) / 27.0f;
    return min_log_hz * expf(logstep * (mel - min_log_mel));
}

void mel_init(uint32_t fft_bins, uint32_t num_mels,
              uint32_t sample_rate, float fmin, float fmax)
{
    if (fft_bins > MAX_FFT_BINS) fft_bins = MAX_FFT_BINS;
    if (num_mels > MAX_MEL_BINS) num_mels = MAX_MEL_BINS;

    mel_n_fft_bins = fft_bins;
    mel_n_mels = num_mels;

    memset(mel_weights, 0, sizeof(mel_weights));

    /* Compute mel center frequencies for num_mels + 2 points */
    float mel_min = hz_to_mel(fmin);
    float mel_max = hz_to_mel(fmax);

    /* num_mels + 2 equally spaced points in mel space */
    uint32_t n_points = num_mels + 2;
    float mel_points[MAX_MEL_BINS + 2];
    float hz_points[MAX_MEL_BINS + 2];
    float fft_freqs_step = (float)sample_rate / (float)((fft_bins - 1) * 2);

    for (uint32_t i = 0; i < n_points; i++) {
        mel_points[i] = mel_min + (mel_max - mel_min) * (float)i / (float)(n_points - 1);
        hz_points[i] = mel_to_hz(mel_points[i]);
    }

    /* Build triangular filters with Slaney normalization */
    for (uint32_t m = 0; m < num_mels; m++) {
        float left   = hz_points[m];
        float center = hz_points[m + 1];
        float right  = hz_points[m + 2];

        /* Slaney normalization: 2 / (right - left) */
        float enorm = 2.0f / (right - left + 1e-10f);

        for (uint32_t f = 0; f < fft_bins; f++) {
            float freq = (float)f * fft_freqs_step;
            float w = 0.0f;

            if (freq >= left && freq <= center) {
                w = (freq - left) / (center - left + 1e-10f);
            } else if (freq > center && freq <= right) {
                w = (right - freq) / (right - center + 1e-10f);
            }

            mel_weights[m * MAX_FFT_BINS + f] = w * enorm;
        }
    }
}

void mel_filterbank(const float *stft_mag, uint32_t fft_bins,
                    uint32_t spec_width, uint32_t num_mels,
                    float *mel_out)
{
    /* mel_out[m, t] = sum_f( mel_weights[m, f] * stft_mag[f, t] )
     * stft_mag layout: [fft_bins, spec_width] row-major
     * mel_out layout:  [num_mels, spec_width] row-major */

    if (fft_bins > mel_n_fft_bins) fft_bins = mel_n_fft_bins;
    if (num_mels > mel_n_mels) num_mels = mel_n_mels;

    memset(mel_out, 0, num_mels * spec_width * sizeof(float));

    for (uint32_t m = 0; m < num_mels; m++) {
        const float *w_row = &mel_weights[m * MAX_FFT_BINS];
        float *out_row = &mel_out[m * spec_width];

        for (uint32_t f = 0; f < fft_bins; f++) {
            float w = w_row[f];
            if (w == 0.0f) continue;

            const float *mag_row = &stft_mag[f * spec_width];
            for (uint32_t t = 0; t < spec_width; t++) {
                out_row[t] += w * mag_row[t];
            }
        }
    }
}
