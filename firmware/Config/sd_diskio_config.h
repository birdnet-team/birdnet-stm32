/* SPDX-License-Identifier: Apache-2.0
 * SD diskio configuration for BirdNET-STM32 board-test firmware.
 *
 * Adapts the ST FatFs SD driver to use SDMMC2 on the STM32N6570-DK
 * via the BSP SD driver (hsd_sdmmc[0]).
 */

#ifndef SD_DISKIO_CONFIG_H
#define SD_DISKIO_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stm32n6xx_hal.h"

/* Timeout for HAL_SD_ReadBlocks / HAL_SD_WriteBlocks (ms) */
#define SD_TIMEOUT  (30 * 1000)

/* SD init is done by BSP_SD_Init() in our main.c, not by the diskio layer */
#define ENABLE_SD_INIT  0

/* No DMA cache maintenance needed — we use polling mode */
#define ENABLE_SD_DMA_CACHE_MAINTENANCE  0

/* The BSP SD driver exposes the handle as hsd_sdmmc[0].
 * The sd_diskio.c expects a single handle named sdmmc_handle. */
extern SD_HandleTypeDef hsd_sdmmc[];
#define sdmmc_handle  hsd_sdmmc[0]

#ifdef __cplusplus
}
#endif

#endif /* SD_DISKIO_CONFIG_H */
