/* SPDX-License-Identifier: Apache-2.0
 * Real-valued STFT through CMSIS-DSP (Helium on the Cortex-M55) — produces a
 * linear magnitude spectrogram compatible with the BirdNET hybrid frontend.
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
 * Uses CMSIS-DSP's real FFT and a periodic Hann window, with frames centred on
 * t * hop_length and zero-padded at the edges -- librosa.stft(center=True,
 * pad_mode="constant"), which is what the host feeds the model.
 *
 * @param audio       Input: mono float32 samples, length >= chunk_samples.
 * @param chunk_samples  Number of input samples (e.g. sample_rate * duration).
 * @param fft_length  FFT window size: a power of two from 32 to 512.
 * @param hop_length  Hop between successive frames (e.g. 258).
 * @param spec_width  Number of STFT frames to produce.
 * @param out         Output buffer: [fft_bins x spec_width] floats,
 *                    where fft_bins = fft_length / 2 (Nyquist omitted).
 *                    Must be allocated by caller.
 */
void stft_magnitude(const float *audio, uint32_t chunk_samples,
                    uint32_t fft_length, uint32_t hop_length,
                    uint32_t spec_width, float *out);

/**
 * Min-max normalize a spectrogram in place to [0, 1]: (S - min) / (max - min + 1e-10).
 *
 * The host normalizes every spectrogram this way before the model sees it
 * (normalize() in birdnet_stm32/audio/spectrogram.py), so the firmware must too.
 *
 * @param spec   Spectrogram buffer.
 * @param count  Number of values (rows x columns).
 */
void spec_minmax_normalize(float *spec, uint32_t count);

/* Input compression modes, as input_compression in the model config. */
#define SPEC_COMPRESS_NONE 0
#define SPEC_COMPRESS_SQRT 1
#define SPEC_COMPRESS_LOG  2
/* Dynamic range kept by SPEC_COMPRESS_LOG below each chunk's peak (LOG_FLOOR_DB on the host). */
#define SPEC_LOG_FLOOR_DB  80.0f

/**
 * Compress a magnitude spectrogram in place before spec_minmax_normalize().
 *
 * Must match compress() in birdnet_stm32/audio/stft.py: sqrt, or the natural
 * log with a floor SPEC_LOG_FLOOR_DB below the peak. It runs in float on the M55,
 * so the model's first INT8 tensor already holds compressed values.
 *
 * @param spec   Spectrogram buffer.
 * @param count  Number of values (rows x columns).
 * @param mode   SPEC_COMPRESS_NONE, SPEC_COMPRESS_SQRT or SPEC_COMPRESS_LOG.
 */
void spec_compress(float *spec, uint32_t count, int mode);

#endif /* AUDIO_STFT_H */
