/* SPDX-License-Identifier: Apache-2.0
 * SD card handler — BSP SD init, FatFs mount, directory scan, results writer.
 */

#ifndef SD_HANDLER_H
#define SD_HANDLER_H

#include <stdint.h>
#include <stdbool.h>
#include "ff.h"

/** Maximum path length for SD card files. */
#define SD_MAX_PATH 256

/** Maximum number of audio files to enumerate. */
#define SD_MAX_FILES 512

/** File list populated by sd_scan_audio_dir(). */
typedef struct {
    char paths[SD_MAX_FILES][SD_MAX_PATH];
    uint32_t count;
} SdFileList;

/**
 * Mount the SD card and initialise FatFs.
 * @return true on success.
 */
bool sd_mount(void);

/**
 * Unmount the SD card.
 */
void sd_unmount(void);

/**
 * Scan a directory for .wav files (non-recursive).
 *
 * @param dir_path  Path on the SD card (e.g. "audio").
 * @param list      Output: populated file list.
 * @return Number of .wav files found.
 */
uint32_t sd_scan_audio_dir(const char *dir_path, SdFileList *list);

/**
 * Append one result line to the results file.
 * Format: "filename\tscore_0\tscore_1\t...\n"
 *
 * @param path         Results file path (e.g. "results.txt").
 * @param filename     Audio file name (basename).
 * @param scores       Float array of class scores.
 * @param num_classes  Number of classes.
 * @return true on success.
 */
bool sd_append_result(const char *path, const char *filename,
                      const float *scores, uint32_t num_classes);

/**
 * Write the results file header.
 *
 * @param path         Results file path.
 * @param labels       Array of class label strings.
 * @param num_classes  Number of classes.
 * @return true on success.
 */
bool sd_write_header(const char *path, const char * const *labels,
                     uint32_t num_classes);

#endif /* SD_HANDLER_H */
