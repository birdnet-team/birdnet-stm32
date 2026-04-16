/* SPDX-License-Identifier: Apache-2.0
 * BirdNET-STM32 — SD card batch inference application
 *
 * Reads WAV files from SD:/audio/, computes STFT, runs NPU inference,
 * writes results to SD:/results.txt and echoes over UART.
 *
 * Target: STM32N6570-DK  (Cortex-M55 + NPU)
 * Dependencies: STM32CubeN6 HAL, BSP, FatFs, CMSIS-DSP, LL_ATON runtime
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
#define APP_NUM_CLASSES       10         /* update to match your model         */

/* --- SD card paths -------------------------------------------------------- */
#define APP_AUDIO_DIR         "audio"
#define APP_RESULTS_FILE      "results.txt"

/* --- UART (debug / results echo) ------------------------------------------ */
#define APP_UART_BAUDRATE     115200

/* --- Inference ------------------------------------------------------------- */
#define APP_TOP_K             5          /* top-K results to print per file    */
#define APP_SCORE_THRESHOLD   0.1f       /* minimum score to include in output */

#endif /* APP_CONFIG_H */
