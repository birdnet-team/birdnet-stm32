/* SPDX-License-Identifier: Apache-2.0
 * BirdNET-STM32 — SD card batch inference application
 *
 * Audio and inference parameters, must match model_config.json.
 * These values are patched by the Python board_test.py orchestrator
 * at deploy time based on the model configuration.
 *
 * Target: STM32N6570-DK  (Cortex-M55 + NPU)
 */

#ifndef APP_CONFIG_H
#define APP_CONFIG_H

/* --- Audio parameters (must match model_config.json) ---------------------- */
#define APP_SAMPLE_RATE       22050
#define APP_CHUNK_DURATION    3          /* seconds                            */
#define APP_CHUNK_SAMPLES     (APP_SAMPLE_RATE * APP_CHUNK_DURATION)  /* 66150 */
#define APP_FFT_LENGTH        512
#define APP_FFT_BINS          (APP_FFT_LENGTH / 2 + 1)               /* 257   */
#define APP_HOP_LENGTH        258
#define APP_SPEC_WIDTH        256
#define APP_NUM_MELS          64         /* mel bins (precomputed frontend)    */
#define APP_NUM_CLASSES       1000       /* patched at deploy time             */

/* --- Audio frontend mode -------------------------------------------------- */
/* 0 = hybrid:      STFT on M55 → feed [fft_bins, spec_width] to NPU         */
/* 1 = raw:         feed raw waveform [chunk_samples, 1] to NPU (no STFT)    */
/* 2 = precomputed: STFT + mel filterbank on M55 → feed [num_mels, spec_width] */
#define APP_FRONTEND_HYBRID       0
#define APP_FRONTEND_RAW          1
#define APP_FRONTEND_PRECOMPUTED  2
#define APP_AUDIO_FRONTEND        APP_FRONTEND_HYBRID  /* patched at deploy   */

/* --- SD card paths -------------------------------------------------------- */
#define APP_AUDIO_DIR         "audio"
#define APP_RESULTS_FILE      "results.txt"

/* --- UART (debug / results echo) ------------------------------------------ */
#define APP_UART_BAUDRATE     115200

/* --- Inference ------------------------------------------------------------- */
#define APP_TOP_K             5          /* top-K results to print per file    */
#define APP_SCORE_THRESHOLD   0.1f       /* minimum score to include in output */

#endif /* APP_CONFIG_H */
