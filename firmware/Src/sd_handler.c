/* SPDX-License-Identifier: Apache-2.0
 * SD card handler — FatFs mount, directory scan, results writer.
 *
 * Uses FatFs with the STM32 SDMMC BSP driver.
 */

#include "sd_handler.h"
#include <string.h>
#include <stdio.h>

/* FatFs objects */
static FATFS sd_fs;

/* ---- Public API ---------------------------------------------------------- */

bool sd_mount(void)
{
    FRESULT res = f_mount(&sd_fs, "", 1 /* mount now */);
    return (res == FR_OK);
}

void sd_unmount(void)
{
    f_mount(NULL, "", 0);
}

uint32_t sd_scan_audio_dir(const char *dir_path, SdFileList *list)
{
    DIR dir;
    FILINFO fno;
    list->count = 0;

    if (f_opendir(&dir, dir_path) != FR_OK)
        return 0;

    while (f_readdir(&dir, &fno) == FR_OK && fno.fname[0] != '\0') {
        if (fno.fattrib & AM_DIR)
            continue;                            /* skip subdirectories */

        /* Check for .wav extension (case-insensitive) */
        const char *ext = strrchr(fno.fname, '.');
        if (!ext)
            continue;
        if (strcasecmp(ext, ".wav") != 0)
            continue;

        if (list->count >= SD_MAX_FILES)
            break;

        snprintf(list->paths[list->count], SD_MAX_PATH,
                 "%s/%s", dir_path, fno.fname);
        list->count++;
    }

    f_closedir(&dir);
    return list->count;
}

bool sd_write_header(const char *path, const char * const *labels,
                     uint32_t num_classes)
{
    FIL fp;
    if (f_open(&fp, path, FA_CREATE_ALWAYS | FA_WRITE) != FR_OK)
        return false;

    /* Write TSV header: filename <TAB> class_0 <TAB> class_1 ... */
    f_printf(&fp, "filename");
    for (uint32_t i = 0; i < num_classes; i++)
        f_printf(&fp, "\t%s", labels[i]);
    f_printf(&fp, "\n");

    f_close(&fp);
    return true;
}

bool sd_append_result(const char *path, const char *filename,
                      const float *scores, uint32_t num_classes)
{
    FIL fp;
    if (f_open(&fp, path, FA_OPEN_APPEND | FA_WRITE) != FR_OK)
        return false;

    f_printf(&fp, "%s", filename);
    for (uint32_t i = 0; i < num_classes; i++) {
        /* f_printf doesn't support %f; use integer + decimal workaround */
        int whole = (int)(scores[i] * 10000.0f);
        int sign  = (whole < 0) ? -1 : 1;
        whole = whole * sign;
        f_printf(&fp, "\t%s%d.%04d",
                 (sign < 0) ? "-" : "",
                 whole / 10000, whole % 10000);
    }
    f_printf(&fp, "\n");

    f_close(&fp);
    return true;
}
