/* SPDX-License-Identifier: Apache-2.0
 * Real-valued STFT using a plain-C 512-point FFT — produces a linear
 * magnitude spectrogram compatible with the BirdNET hybrid frontend.
 */

#ifndef AUDIO_STFT_H
#define AUDIO_STFT_H

#include <stdint.h>

/**
 * Compute the linear magnitude STFT of an audio chunk.
 *
 * Produces a spectrogram of shape [fft_bins, spec_width] stored in
 * row-major order (frequency-major: row = frequency bin, col = time frame).
 *
 * Uses a custom radix-2 FFT and a Hann window.
 *
 * @param audio       Input: mono float32 samples, length >= chunk_samples.
 * @param chunk_samples  Number of input samples (e.g. sample_rate * duration).
 * @param fft_length  FFT window size (must be 512).
 * @param hop_length  Hop between successive frames (e.g. 258).
 * @param spec_width  Number of STFT frames to produce.
 * @param out         Output buffer: [fft_bins x spec_width] floats,
 *                    where fft_bins = fft_length / 2 + 1.
 *                    Must be allocated by caller.
 */
void stft_magnitude(const float *audio, uint32_t chunk_samples,
                    uint32_t fft_length, uint32_t hop_length,
                    uint32_t spec_width, float *out);

#endif /* AUDIO_STFT_H */
