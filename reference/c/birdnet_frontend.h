/* SPDX-License-Identifier: MIT
 * BirdNET Tiny N6 — reference audio frontend in C.
 *
 * Everything a device has to compute before the model runs, for both
 * frontends, as plain C99 plus CMSIS-DSP for the FFT. It mirrors
 * reference/birdnet_tiny_reference.py step for step; the numbers in the step
 * comments refer to reference/README.md.
 *
 *   raw:     window -> peak_normalize()                      -> model input [samples]
 *   hybrid:  window -> stft_magnitude() -> compress()
 *                   -> minmax_normalize()                    -> model input [bins x frames]
 *
 * All sizes come from the bundle's *_model_config.json:
 *   sample_rate, chunk_duration -> window length in samples
 *   fft_length, spec_width      -> STFT size and frame count (hybrid)
 *   input_compression           -> COMPRESS_* (hybrid)
 */
#ifndef BIRDNET_FRONTEND_H
#define BIRDNET_FRONTEND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BN_MAX_FFT 512 /* largest supported fft_length (a power of two, >= 32) */

enum bn_compression { BN_COMPRESS_NONE = 0, BN_COMPRESS_SQRT = 1, BN_COMPRESS_LOG = 2 };

/* Step 2: start sample of window `index` for a recording of `num_samples`.
 * Windows are `size` samples long and step by `step`; the last window is
 * right-aligned to the end of the recording. Returns the number of windows,
 * and writes the start into *start when index < that number. A recording
 * shorter than one window gives one window starting at 0 (zero-pad it). */
uint32_t bn_window_start(uint32_t num_samples, uint32_t size, uint32_t step, uint32_t index, uint32_t *start);

/* Step 3, raw: x /= (max|x| + 1e-6), in place. */
void bn_peak_normalize(float *x, uint32_t n);

/* Step 3.1-3.4, hybrid: linear magnitude STFT of one window.
 *   hop = n // spec_width; frames centred on sample t*hop with zeros outside
 *   the window; periodic Hann; bins 0 .. fft_length/2 - 1 (Nyquist dropped).
 * `out` holds (fft_length/2) * spec_width floats, frequency-major:
 *   out[bin * spec_width + frame]. Returns 0, or -1 for an unsupported size. */
int bn_stft_magnitude(const float *window, uint32_t n, uint32_t fft_length, uint32_t spec_width, float *out);

/* Step 3.5, hybrid: sqrt, or natural log floored 80 dB below the peak. */
void bn_compress(float *spec, uint32_t count, enum bn_compression mode);

/* Step 3.6, hybrid: (S - min) / (max - min + 1e-10), in place. */
void bn_minmax_normalize(float *spec, uint32_t count);

#ifdef __cplusplus
}
#endif

#endif /* BIRDNET_FRONTEND_H */
