/* SPDX-License-Identifier: MIT
 * Run the reference C frontend on a WAV file and print what it computed.
 *
 *   frontend_cli <audio.wav> raw    <chunk_duration> [inputs.f32]
 *   frontend_cli <audio.wav> hybrid <chunk_duration> <fft_length> <spec_width> <none|sqrt|log> [inputs.f32]
 *
 * The WAV must be mono PCM16 at the model's sample rate (resampling is step 1
 * and belongs to the caller). For every window it prints one JSON line with
 * the same summaries as reference/vectors/<bundle>.json -- shape, sum, min,
 * max and the first eight values of each stage -- so the output can be
 * compared with the recorded vectors directly. With an output path, the model
 * inputs are also written there as consecutive float32 tensors.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "birdnet_frontend.h"

static float *read_wav_pcm16(const char *path, uint32_t *num_samples, uint32_t *sample_rate)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    unsigned char h[12];
    if (fread(h, 1, 12, f) != 12 || memcmp(h, "RIFF", 4) || memcmp(h + 8, "WAVE", 4)) goto fail;
    uint16_t channels = 0, bits = 0;
    for (;;) {
        unsigned char c[8];
        if (fread(c, 1, 8, f) != 8) goto fail;
        uint32_t size = c[4] | c[5] << 8 | c[6] << 16 | (uint32_t)c[7] << 24;
        if (!memcmp(c, "fmt ", 4)) {
            unsigned char fmt[16];
            if (size < 16 || fread(fmt, 1, 16, f) != 16) goto fail;
            channels = fmt[2] | fmt[3] << 8;
            *sample_rate = fmt[4] | fmt[5] << 8 | fmt[6] << 16 | (uint32_t)fmt[7] << 24;
            bits = fmt[14] | fmt[15] << 8;
            fseek(f, (long)(size - 16 + (size & 1)), SEEK_CUR);
        } else if (!memcmp(c, "data", 4)) {
            if (channels != 1 || bits != 16) goto fail;
            *num_samples = size / 2;
            int16_t *pcm = malloc(size);
            float *audio = malloc(*num_samples * sizeof(float));
            if (!pcm || !audio || fread(pcm, 2, *num_samples, f) != *num_samples) goto fail;
            for (uint32_t i = 0; i < *num_samples; i++) audio[i] = pcm[i] / 32768.0f; /* step 1 */
            free(pcm);
            fclose(f);
            return audio;
        } else {
            fseek(f, (long)(size + (size & 1)), SEEK_CUR);
        }
    }
fail:
    fclose(f);
    return NULL;
}

static void print_summary(const char *key, const float *x, uint32_t rows, uint32_t cols, int last)
{
    double sum = 0.0;
    float lo = x[0], hi = x[0];
    for (uint32_t i = 0; i < rows * cols; i++) {
        sum += x[i];
        if (x[i] < lo) lo = x[i];
        if (x[i] > hi) hi = x[i];
    }
    printf("\"%s\": {\"shape\": [%u%s", key, rows, cols > 1 ? ", " : "");
    if (cols > 1) printf("%u", cols);
    printf("], \"sum\": %.6f, \"min\": %.7f, \"max\": %.7f, \"head\": [", sum, lo, hi);
    for (int i = 0; i < 8; i++) printf("%s%.7f", i ? ", " : "", x[i]);
    printf("]}%s", last ? "" : ", ");
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s audio.wav raw <chunk_s> [out.f32]\n"
                        "       %s audio.wav hybrid <chunk_s> <fft_length> <spec_width> <none|sqrt|log> [out.f32]\n",
                argv[0], argv[0]);
        return 2;
    }
    int hybrid = !strcmp(argv[2], "hybrid");
    if (hybrid && argc < 7) return 2;
    uint32_t n_audio = 0, rate = 0;
    float *audio = read_wav_pcm16(argv[1], &n_audio, &rate);
    if (!audio) {
        fprintf(stderr, "cannot read %s as mono PCM16 WAV\n", argv[1]);
        return 1;
    }
    const uint32_t size = (uint32_t)(atof(argv[3]) * rate + 0.5); /* step 2 */
    const uint32_t step = size / 2;
    uint32_t fft = hybrid ? (uint32_t)atoi(argv[4]) : 0, width = hybrid ? (uint32_t)atoi(argv[5]) : 0;
    enum bn_compression mode = BN_COMPRESS_NONE;
    if (hybrid && !strcmp(argv[6], "sqrt")) mode = BN_COMPRESS_SQRT;
    if (hybrid && !strcmp(argv[6], "log")) mode = BN_COMPRESS_LOG;
    const char *out_path = argc > (hybrid ? 7 : 4) ? argv[hybrid ? 7 : 4] : NULL;
    FILE *out = out_path ? fopen(out_path, "wb") : NULL;

    float *window = calloc(size, sizeof(float));
    float *spec = hybrid ? malloc((fft / 2) * width * sizeof(float)) : NULL;
    uint32_t count = bn_window_start(n_audio, size, step, 0, NULL);
    for (uint32_t w = 0; w < count; w++) {
        uint32_t start = 0;
        bn_window_start(n_audio, size, step, w, &start);
        for (uint32_t i = 0; i < size; i++) window[i] = start + i < n_audio ? audio[start + i] : 0.0f;
        printf("{\"start_sample\": %u, ", start);
        print_summary("window", window, size, 1, 0);
        if (hybrid) {
            if (bn_stft_magnitude(window, size, fft, width, spec)) return 1;
            print_summary("stft_magnitude", spec, fft / 2, width, 0);
            bn_compress(spec, (fft / 2) * width, mode);
            print_summary("compressed", spec, fft / 2, width, 0);
            bn_minmax_normalize(spec, (fft / 2) * width);
            print_summary("model_input", spec, fft / 2, width, 1);
            if (out) fwrite(spec, sizeof(float), (fft / 2) * width, out);
        } else {
            bn_peak_normalize(window, size);
            print_summary("model_input", window, size, 1, 1);
            if (out) fwrite(window, sizeof(float), size, out);
        }
        printf("}\n");
    }
    if (out) fclose(out);
    free(audio);
    free(window);
    free(spec);
    return 0;
}
