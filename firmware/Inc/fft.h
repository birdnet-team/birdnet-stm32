/* SPDX-License-Identifier: Apache-2.0
 * Minimal 512-point real FFT for BirdNET-STM32 board-test firmware.
 *
 * Implements a radix-2 decimation-in-time FFT with real-input optimization.
 * Output format matches CMSIS-DSP arm_rfft_fast_f32:
 *   out[0] = DC component (real)
 *   out[1] = Nyquist component (real)
 *   out[2..N-1] = interleaved (real, imag) for bins 1..(N/2-1)
 *
 * Only supports N = 512 (compile-time constant for twiddle table).
 */

#ifndef FFT_H
#define FFT_H

#include <stdint.h>

/**
 * Compute a 512-point real FFT in-place.
 *
 * @param buf  Input: 512 real float32 samples.
 *             Output: packed complex spectrum (512 floats):
 *               [0] = DC real
 *               [1] = Nyquist real
 *               [2k], [2k+1] = real, imag of bin k  (k = 1..255)
 */
void fft_512_real(float *buf);

#endif /* FFT_H */
