/* SPDX-License-Identifier: Apache-2.0
 * WAV file reader — parse RIFF/WAVE headers and read PCM16 samples.
 */

#include "wav_reader.h"
#include <string.h>

/* ---- RIFF chunk IDs ------------------------------------------------------ */
#define RIFF_ID  0x46464952U  /* "RIFF" little-endian */
#define WAVE_ID  0x45564157U  /* "WAVE"               */
#define FMT_ID   0x20746D66U  /* "fmt "               */
#define DATA_ID  0x61746164U  /* "data"               */

/* PCM format tag */
#define WAVE_FORMAT_PCM  1

bool wav_parse_header(FIL *fp, WavInfo *info)
{
    UINT br;
    uint32_t buf32[3];

    /* Read RIFF header: "RIFF" + size + "WAVE" */
    if (f_read(fp, buf32, 12, &br) != FR_OK || br != 12)
        return false;
    if (buf32[0] != RIFF_ID || buf32[2] != WAVE_ID)
        return false;

    /* Walk sub-chunks until we find "fmt " and "data" */
    bool got_fmt = false;
    bool got_data = false;

    while (!(got_fmt && got_data)) {
        uint32_t chunk_id, chunk_size;
        if (f_read(fp, &chunk_id, 4, &br) != FR_OK || br != 4)
            break;
        if (f_read(fp, &chunk_size, 4, &br) != FR_OK || br != 4)
            break;

        if (chunk_id == FMT_ID) {
            uint8_t fmt_buf[40];
            UINT to_read = chunk_size < sizeof(fmt_buf) ? chunk_size : sizeof(fmt_buf);
            if (f_read(fp, fmt_buf, to_read, &br) != FR_OK || br != to_read)
                return false;

            uint16_t audio_format;
            memcpy(&audio_format, &fmt_buf[0], 2);
            if (audio_format != WAVE_FORMAT_PCM)
                return false;

            memcpy(&info->num_channels,   &fmt_buf[2],  2);
            memcpy(&info->sample_rate,    &fmt_buf[4],  4);
            memcpy(&info->bits_per_sample, &fmt_buf[14], 2);
            got_fmt = true;

            /* Skip any extra fmt bytes */
            if (to_read < chunk_size)
                f_lseek(fp, f_tell(fp) + (chunk_size - to_read));
        }
        else if (chunk_id == DATA_ID) {
            info->data_size   = chunk_size;
            info->data_offset = (uint32_t)f_tell(fp);
            info->num_samples = chunk_size / (info->bits_per_sample / 8) / info->num_channels;
            got_data = true;
            /* Don't seek past data — caller will read from here */
        }
        else {
            /* Skip unknown chunk (pad to even boundary) */
            uint32_t skip = chunk_size + (chunk_size & 1);
            f_lseek(fp, f_tell(fp) + skip);
        }
    }

    return got_fmt && got_data;
}

uint32_t wav_read_chunk_f32(FIL *fp, const WavInfo *info,
                            uint32_t sample_offset, uint32_t num_samples,
                            float *out)
{
    /* Seek to the sample position */
    uint32_t bytes_per_sample = info->bits_per_sample / 8;
    uint32_t frame_size = bytes_per_sample * info->num_channels;
    uint32_t file_offset = info->data_offset + sample_offset * frame_size;
    f_lseek(fp, file_offset);

    /* Determine how many samples are available */
    uint32_t available = 0;
    if (sample_offset < info->num_samples)
        available = info->num_samples - sample_offset;
    uint32_t to_read = (available < num_samples) ? available : num_samples;

    /* Read in small blocks (stack-friendly) */
    #define READ_BLK  256
    int16_t pcm_buf[READ_BLK];
    uint32_t written = 0;

    for (uint32_t i = 0; i < to_read; i += READ_BLK) {
        uint32_t blk = (to_read - i < READ_BLK) ? (to_read - i) : READ_BLK;
        UINT br;

        if (info->num_channels == 1 && bytes_per_sample == 2) {
            /* Fast path: mono 16-bit */
            if (f_read(fp, pcm_buf, blk * 2, &br) != FR_OK)
                break;
            uint32_t samples_read = br / 2;
            for (uint32_t j = 0; j < samples_read; j++)
                out[written + j] = (float)pcm_buf[j] / 32768.0f;
            written += samples_read;
        } else {
            /* General path: read frame-by-frame, take channel 0 */
            for (uint32_t j = 0; j < blk; j++) {
                uint8_t frame[8]; /* max 4 bytes/sample * 2 channels */
                if (f_read(fp, frame, frame_size, &br) != FR_OK || br != frame_size)
                    goto done;
                int16_t sample;
                memcpy(&sample, frame, 2);
                out[written++] = (float)sample / 32768.0f;
            }
        }
    }

done:
    /* Zero-pad the remainder */
    for (uint32_t i = written; i < num_samples; i++)
        out[i] = 0.0f;

    return written;
    #undef READ_BLK
}
