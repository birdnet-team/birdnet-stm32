/*---------------------------------------------------------------------------/
/  FatFs configuration for BirdNET-STM32 board-test firmware
/  Minimal read-only config to keep code size small.
/---------------------------------------------------------------------------*/

#define FFCONF_DEF  80286  /* FatFs revision ID */

/*---------------------------------------------------------------------------/
/ Function Configurations
/---------------------------------------------------------------------------*/

#define FF_FS_READONLY  0
/* We need write support to create results.txt on the SD card. */

#define FF_FS_MINIMIZE  0
/* Keep full API — we use f_opendir/f_readdir for scanning audio/. */

#define FF_USE_FIND     0
#define FF_USE_MKFS     0
#define FF_USE_FASTSEEK 0
#define FF_USE_EXPAND   0
#define FF_USE_CHMOD    0
#define FF_USE_LABEL    0
#define FF_USE_FORWARD  0

#define FF_USE_STRFUNC  1
#define FF_PRINT_LLI    0
#define FF_PRINT_FLOAT  0
#define FF_STRF_ENCODE  3
/* f_gets() is used for reading labels.txt.
 * f_printf() is used for writing results.txt. */

/*---------------------------------------------------------------------------/
/ Locale and Namespace Configurations
/---------------------------------------------------------------------------*/

#define FF_CODE_PAGE  437
/* U.S. code page — species names are ASCII/Latin. */

#define FF_USE_LFN    0
#define FF_MAX_LFN    255
/* LFN on stack (mode 1). Species folder names can be long. */

#define FF_LFN_UNICODE    0
#define FF_LFN_BUF        255
#define FF_SFN_BUF        12
#define FF_FS_RPATH        0

/*---------------------------------------------------------------------------/
/ Drive/Volume Configurations
/---------------------------------------------------------------------------*/

#define FF_VOLUMES    1
#define FF_STR_VOLUME_ID  0
#define FF_VOLUME_STRS    "SD"
#define FF_MULTI_PARTITION  0
#define FF_MIN_SS     512
#define FF_MAX_SS     512
/* SD cards always use 512-byte sectors. */

#define FF_LBA64      0
#define FF_MIN_GPT    0x10000000
#define FF_USE_TRIM   0

/*---------------------------------------------------------------------------/
/ System Configurations
/---------------------------------------------------------------------------*/

#define FF_FS_TINY    0
#define FF_FS_EXFAT   0
#define FF_FS_NORTC   1
#define FF_NORTC_MON  1
#define FF_NORTC_MDAY 1
#define FF_NORTC_YEAR 2025
/* No RTC on this firmware — fixed timestamp is fine. */

#define FF_FS_NOFSINFO  0
#define FF_FS_LOCK  0
#define FF_FS_REENTRANT 0
