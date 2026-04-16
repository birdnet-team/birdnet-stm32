/* SPDX-License-Identifier: Apache-2.0
 * Minimal 512-point real FFT for BirdNET-STM32 board-test firmware.
 *
 * Uses the "real FFT via N/2 complex FFT" trick:
 *   1. Treat the 512 real samples as 256 complex pairs.
 *   2. Compute a 256-point complex FFT in-place.
 *   3. Unpack into the 512-point real spectrum.
 *
 * Twiddle factors are precomputed at init time (once).
 */

#include "fft.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* N = 512 real points, N/2 = 256 complex points for the inner FFT. */
#define NREAL  512
#define NCPLX  (NREAL / 2)   /* 256 */

/* Precomputed twiddle factors for the 256-point complex FFT.
 * twiddle_re[k] = cos(2*PI*k / 256), twiddle_im[k] = -sin(2*PI*k / 256)
 * Only need N/2 = 128 values (for butterfly stages). */
static float tw_re[NCPLX / 2];
static float tw_im[NCPLX / 2];
static int   tw_initialized = 0;

/* Bit-reversal permutation table for N = 256 */
static uint16_t bitrev[NCPLX];

static void init_twiddles(void)
{
    if (tw_initialized) return;

    /* Twiddle factors for complex FFT of size NCPLX = 256 */
    for (int k = 0; k < NCPLX / 2; k++) {
        float angle = 2.0f * M_PI * (float)k / (float)NCPLX;
        tw_re[k] =  cosf(angle);
        tw_im[k] = -sinf(angle);
    }

    /* Bit-reversal table for NCPLX = 256 = 2^8 */
    int bits = 8;  /* log2(256) */
    for (int i = 0; i < NCPLX; i++) {
        uint16_t rev = 0;
        uint16_t val = (uint16_t)i;
        for (int b = 0; b < bits; b++) {
            rev = (uint16_t)((rev << 1) | (val & 1));
            val >>= 1;
        }
        bitrev[i] = rev;
    }

    tw_initialized = 1;
}

/**
 * In-place 256-point complex FFT (radix-2 DIT).
 * data[2*i] = real part, data[2*i+1] = imag part.
 */
static void cfft_256(float *data)
{
    /* Bit-reversal permutation */
    for (int i = 0; i < NCPLX; i++) {
        int j = bitrev[i];
        if (j > i) {
            /* Swap complex elements i and j */
            float tr = data[2 * i];
            float ti = data[2 * i + 1];
            data[2 * i]     = data[2 * j];
            data[2 * i + 1] = data[2 * j + 1];
            data[2 * j]     = tr;
            data[2 * j + 1] = ti;
        }
    }

    /* Butterfly stages */
    for (int stage_size = 2; stage_size <= NCPLX; stage_size <<= 1) {
        int half = stage_size / 2;
        int tw_step = NCPLX / stage_size;  /* twiddle index stride */

        for (int group = 0; group < NCPLX; group += stage_size) {
            for (int k = 0; k < half; k++) {
                int tw_idx = k * tw_step;
                float wr = tw_re[tw_idx];
                float wi = tw_im[tw_idx];

                int i1 = 2 * (group + k);
                int i2 = 2 * (group + k + half);

                /* Complex multiply: W * data[i2] */
                float tr = wr * data[i2] - wi * data[i2 + 1];
                float ti = wr * data[i2 + 1] + wi * data[i2];

                /* Butterfly */
                data[i2]     = data[i1] - tr;
                data[i2 + 1] = data[i1 + 1] - ti;
                data[i1]     += tr;
                data[i1 + 1] += ti;
            }
        }
    }
}

void fft_512_real(float *buf)
{
    init_twiddles();

    /* Step 1: The 512 real values in buf are treated as 256 complex values.
     * buf[2*k] = x[2*k] (real part), buf[2*k+1] = x[2*k+1] (imag part).
     * Run the 256-point complex FFT in-place. */
    cfft_256(buf);

    /* Step 2: Unpack the 256-point complex FFT result into the 512-point
     * real FFT result using the "split radix" post-processing.
     *
     * Let X[k] = result of 256-pt complex FFT.
     * The 512-pt real FFT bins F[k] for k = 0..256 are:
     *   F[k] = 0.5 * (X[k] + X*[N/2-k]) - 0.5j * W(k,N) * (X[k] - X*[N/2-k])
     * where W(k,N) = exp(-j*2*PI*k/N) and X*[k] is conjugate.
     *
     * We store the result in CMSIS-DSP format:
     *   buf[0] = F[0].real  (DC)
     *   buf[1] = F[N/2].real (Nyquist)
     *   buf[2*k], buf[2*k+1] = F[k].real, F[k].imag  for k = 1..N/2-1
     */

    /* Save X[0] before we overwrite it */
    float x0r = buf[0];
    float x0i = buf[1];

    /* F[0] = (X[0].real + X[0].imag, 0) — DC is purely real from a real-input FFT
     * F[N/2] = (X[0].real - X[0].imag, 0) — Nyquist is purely real */
    buf[0] = x0r + x0i;  /* DC */
    buf[1] = x0r - x0i;  /* Nyquist */

    /* Unpack bins k = 1 .. NCPLX-1
     * We process k and (NCPLX - k) together. */
    for (int k = 1; k < NCPLX / 2; k++) {
        int kc = NCPLX - k;  /* conjugate index */

        float xkr  = buf[2 * k];
        float xki  = buf[2 * k + 1];
        float xkcr = buf[2 * kc];
        float xkci = buf[2 * kc + 1];

        /* Twiddle factor W(k, NREAL) = exp(-j * 2*PI*k / 512) */
        float angle = 2.0f * M_PI * (float)k / (float)NREAL;
        float wr = cosf(angle);
        float wi = -sinf(angle);

        /* Even part: Ae = 0.5 * (X[k] + X*[NCPLX-k]) */
        float ae_r = 0.5f * (xkr + xkcr);
        float ae_i = 0.5f * (xki - xkci);

        /* Odd part: Ao = 0.5 * (X[k] - X*[NCPLX-k]) */
        float ao_r = 0.5f * (xkr - xkcr);
        float ao_i = 0.5f * (xki + xkci);

        /* W * Ao (complex multiply) */
        /* But we need -j * W * Ao:
         * -j * (wr + j*wi) * (ao_r + j*ao_i)
         * = -j * ((wr*ao_r - wi*ao_i) + j*(wr*ao_i + wi*ao_r))
         * = (wr*ao_i + wi*ao_r) - j*(wr*ao_r - wi*ao_i) */
        float wao_r =   wr * ao_i + wi * ao_r;
        float wao_i = -(wr * ao_r - wi * ao_i);

        /* F[k] = Ae + (-j * W * Ao) */
        buf[2 * k]     = ae_r + wao_r;
        buf[2 * k + 1] = ae_i + wao_i;

        /* F[NCPLX-k] uses conjugate symmetry of the twiddle:
         * W(NCPLX-k, NREAL) = exp(-j*2*PI*(NCPLX-k)/NREAL)
         * For real input: F[NREAL-k] = F*[k], so F[NCPLX-k] is just the
         * conjugate pair with the mirrored twiddle. */

        /* Twiddle for kc: angle_kc = 2*PI*kc/NREAL, which is PI - angle_k */
        float wr2 = -wr;     /* cos(PI - angle) = -cos(angle) */
        float wi2 =  wi;     /* -sin(PI - angle) = sin(angle) = -wi... wait:
                                 -sin(PI - a) = -sin(a) = wi (since wi = -sin(a)) */

        /* Even part for kc: Ae_kc = 0.5 * (X[kc] + X*[k]) */
        float ae2_r = 0.5f * (xkcr + xkr);
        float ae2_i = 0.5f * (xkci - xki);

        /* Odd part for kc: Ao_kc = 0.5 * (X[kc] - X*[k]) */
        float ao2_r = 0.5f * (xkcr - xkr);
        float ao2_i = 0.5f * (xkci + xki);

        /* -j * W_kc * Ao_kc */
        float wao2_r =   wr2 * ao2_i + wi2 * ao2_r;
        float wao2_i = -(wr2 * ao2_r - wi2 * ao2_i);

        buf[2 * kc]     = ae2_r + wao2_r;
        buf[2 * kc + 1] = ae2_i + wao2_i;
    }

    /* Handle k = NCPLX/2 = 128 (its conjugate is also 128)
     * X[128] is self-conjugate in the split. */
    {
        int k = NCPLX / 2;
        float xkr = buf[2 * k];
        float xki = buf[2 * k + 1];

        /* For k = N/4 of the real FFT: W(N/4, N) = exp(-j*PI/2) = -j
         * F[N/4] = Ae - j*(-j)*Ao = Ae - Ao
         * But Ae = (X[k] + X*[k])/2 = (xkr, 0)  [self-conjugate]
         * Wait, X[k] and X[NCPLX-k] = X[128] are the same point, so:
         * Ae = 0.5*(X[k] + X*[k]) = (xkr, 0)
         * Ao = 0.5*(X[k] - X*[k]) = (0, xki)
         * W(128, 512) = exp(-j*2*PI*128/512) = exp(-j*PI/2) = -j
         * -j * W * Ao = -j * (-j) * (0 + j*xki) = -1 * j*xki = -j*xki = (0, -xki)
         * Hmm, let me just directly compute:  */
        float angle = 2.0f * M_PI * (float)k / (float)NREAL;
        float wr = cosf(angle);    /* cos(PI/2) = 0 */
        float wi_val = -sinf(angle);  /* -sin(PI/2) = -1 */

        float ae_r = xkr;   /* 0.5 * (xkr + xkr) */
        float ae_i = 0.0f;  /* 0.5 * (xki - xki) */

        float ao_r = 0.0f;  /* 0.5 * (xkr - xkr) */
        float ao_i = xki;   /* 0.5 * (xki + xki) */

        float wao_r =   wr * ao_i + wi_val * ao_r;   /* 0*xki + (-1)*0 = 0 */
        float wao_i = -(wr * ao_r - wi_val * ao_i);  /* -(0*0 - (-1)*xki) = -xki */

        buf[2 * k]     = ae_r + wao_r;      /* xkr */
        buf[2 * k + 1] = ae_i + wao_i;      /* -xki */
    }
}
