/* SPDX-License-Identifier: Apache-2.0
 * WAV file reader — parse RIFF/WAVE headers and read PCM16 samples.
 */

#ifndef WAV_READER_H
#define WAV_READER_H

#include <stdint.h>
#include <stdbool.h>
#include "ff.h"   /* FatFs */

typedef struct {
    uint16_t num_channels;
    uint32_t sample_rate;
    uint16_t bits_per_sample;
    uint32_t data_size;        /* bytes of PCM data                         */
    uint32_t num_samples;      /* total samples (per channel)               */
    uint32_t data_offset;      /* byte offset of PCM data start in file     */
} WavInfo;

/**
 * Parse a WAV file header.  Seeks to the start of the 'data' chunk.
 *
 * @param fp     Open FatFs file handle (must be readable).
 * @param info   Output: parsed header fields.
 * @return true on success, false if the file is not a valid PCM WAV.
 */
bool wav_parse_header(FIL *fp, WavInfo *info);

/**
 * Read a chunk of PCM samples from an open WAV file and convert to float32.
 *
 * Reads mono (channel 0) samples starting at `sample_offset`.  If the file
 * is shorter than `num_samples`, the remainder is zero-padded.
 *
 * @param fp             Open FatFs file positioned past the header.
 * @param info           Parsed WAV header.
 * @param sample_offset  First sample index to read (per-channel).
 * @param num_samples    Number of float32 samples to produce.
 * @param out            Output buffer (num_samples floats, must be allocated).
 * @return Number of samples actually read from the file (before zero-pad).
 */
uint32_t wav_read_chunk_f32(FIL *fp, const WavInfo *info,
                            uint32_t sample_offset, uint32_t num_samples,
                            float *out);

#endif /* WAV_READER_H */
