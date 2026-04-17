/* SPDX-License-Identifier: Apache-2.0
 * BirdNET-STM32 — Mel filterbank for precomputed frontend.
 *
 * Multiplies a linear STFT magnitude spectrogram by a triangular mel weight
 * matrix to produce a mel spectrogram.
 */

#ifndef AUDIO_MEL_H
#define AUDIO_MEL_H

#include <stdint.h>

/**
 * Initialize the mel filterbank weight matrix.
 *
 * Builds a Slaney-normalized triangular mel filterbank with the given
 * parameters.  Must be called once before mel_filterbank().
 *
 * @param fft_bins    Number of FFT frequency bins (fft_length / 2 + 1).
 * @param num_mels    Number of mel bands.
 * @param sample_rate Audio sample rate in Hz.
 * @param fmin        Lowest mel filter center frequency (Hz).
 * @param fmax        Highest mel filter center frequency (Hz).
 */
void mel_init(uint32_t fft_bins, uint32_t num_mels,
              uint32_t sample_rate, float fmin, float fmax);

/**
 * Apply the mel filterbank to a linear magnitude spectrogram.
 *
 * @param stft_mag  Linear magnitude spectrogram [fft_bins, spec_width], row-major.
 * @param fft_bins  Number of FFT bins (rows in stft_mag).
 * @param spec_width Number of time frames (columns in stft_mag).
 * @param num_mels  Number of mel bands (rows in mel_out).
 * @param mel_out   Output mel spectrogram [num_mels, spec_width], row-major.
 */
void mel_filterbank(const float *stft_mag, uint32_t fft_bins,
                    uint32_t spec_width, uint32_t num_mels,
                    float *mel_out);

#endif /* AUDIO_MEL_H */
