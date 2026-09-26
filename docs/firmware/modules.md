# Source Modules

Detailed reference for the firmware application modules.

## `main.c` — Orchestrator

**Location:** `firmware/Src/main.c`

The entry point. Initializes the board, mounts the SD card, loops over WAV
files, applies frontend-specific preprocessing, and coordinates NPU inference
and UART output.

### Key Sections

**Board init** — a 12-step sequence that must execute in order. See
[Building & Flashing — Init Sequence](building.md#init-sequence) for the full
table.

**UART redirect** — provides `__io_putchar()` which routes `printf()` output to
USART1 via HAL. The weak symbol in `syscalls.c` (from NPU_Validation) calls
this.

```c
int __io_putchar(int ch)
{
    HAL_UART_Transmit(&UartHandle, (uint8_t *)&ch, 1, HAL_MAX_DELAY);
    return ch;
}
```

**NPU inference** — `run_inference()` handles the CPU ↔ NPU data transfer:

1. Query input/output buffer addresses from LL_ATON.
2. `memcpy` data into the NPU input buffer. Depending on `APP_AUDIO_FRONTEND`:
   - **Hybrid:** `spec_buf` (STFT spectrogram)
   - **Raw:** `audio_buf` after peak normalization
   - **Precomputed:** `mel_buf` (STFT + Mel filterbank)
3. `SCB_CleanDCache_by_Addr()` — flush CPU cache so the NPU sees fresh data.
4. `LL_ATON_RT_Main()` — run inference (blocking).
5. `SCB_InvalidateDCache_by_Addr()` — invalidate cache so the CPU sees NPU
   output.
6. `memcpy` scores out of the NPU output buffer.

**Top-K selection** — simple O(K×N) partial sort. Prints predictions with score
≥ `APP_SCORE_THRESHOLD`.

**Benchmark timing** — `HAL_GetTick()` (1 ms resolution) wraps each stage:
read, STFT, NPU. Per-file `[BENCH]` lines and an aggregate summary at the end.

### Functions

| Function | Signature | Purpose |
|---|---|---|
| `main()` | `int main(void)` | Entry point: init + processing loop |
| `run_inference()` | `bool run_inference(const float *, float *)` | Copy frontend data to NPU, run, copy scores back |
| `print_top_k()` | `void print_top_k(const char *, const float *, int)` | Print top-K over UART |
| `__io_putchar()` | `int __io_putchar(int)` | UART printf redirect |
| `aiValidationInit()` | `static void aiValidationInit(void)` | GDB breakpoint stub (**do not remove**) |
| `Error_Handler()` | `void Error_Handler(void)` | HAL error hook (infinite loop) |
| `assert_failed()` | `void assert_failed(uint8_t *, uint32_t)` | HAL assert hook (infinite loop) |

---

## `wav_reader.c` — WAV File Parser

**Location:** `firmware/Src/wav_reader.c`

Parses standard RIFF/WAVE files with PCM encoding.

### How It Works

1. Reads 12-byte RIFF header, validates `"RIFF"` and `"WAVE"` magic.
2. Walks sub-chunks looking for `"fmt "` and `"data"`:
   - `fmt ` — extracts channels, sample rate, bits per sample.
   - `data` — records the file offset and byte count.
   - Unknown chunks are skipped by reading their size and seeking past them.
3. Validates: PCM format (tag 1), 16-bit, correct sample rate.
4. Returns a `WavInfo` struct with all parsed metadata.

### `wav_read_chunk_f32()`

Reads a chunk of PCM16 audio from the file and converts to float32:

- Reads `num_samples × num_channels × 2` bytes via FatFs `f_read()`.
- Converts each `int16_t` sample to `float32` by dividing by 32768.
- For stereo: extracts channel 0 only (every other sample).
- Zero-pads if the file is shorter than `num_samples`.

### Data Structures

```c
typedef struct {
    uint16_t num_channels;      // 1 = mono, 2 = stereo
    uint32_t sample_rate;       // e.g. 24000
    uint16_t bits_per_sample;   // must be 16
    uint32_t data_size;         // PCM data size in bytes
    uint32_t num_samples;       // total samples (per channel)
    uint32_t data_offset;       // file offset to start of PCM data
} WavInfo;
```

### Limitations

- **16-bit PCM only** — no float32 WAV, A-law, mu-law, or ADPCM.
- **No resampling** — sample rate must match `APP_SAMPLE_RATE`.
- **First chunk only** — reads from the start of the file, not seeking to
  arbitrary positions.

---

## `audio_stft.c` — STFT Engine

**Location:** `firmware/Src/audio_stft.c`

Computes a magnitude STFT with CMSIS-DSP that reproduces the host's hybrid
input — `librosa.stft(center=True, pad_mode="constant", window="hann")` — and
normalizes it like the host. `tests/test_firmware_stft.py` compiles this file
natively and checks it against `get_spectrogram_from_audio()`.

### `stft_magnitude()`

```c
void stft_magnitude(const float *audio, uint32_t num_samples,
                    uint32_t fft_length, uint32_t hop_length,
                    uint32_t spec_width, float *out);
```

**Algorithm:**

1. Pre-compute a periodic Hann window of `fft_length` samples.
2. For each of `spec_width` time frames:
   - Extract `fft_length` samples centred on `frame × hop_length`, taking
     zeros outside the chunk (librosa's `center=True, pad_mode="constant"`).
   - Multiply by the Hann window.
   - Real FFT with CMSIS-DSP's `arm_rfft_fast_f32` (Helium-vectorized).
   - Magnitude for bins 0–255 with `arm_cmplx_mag_f32`; Nyquist is omitted,
     matching the model input.
   - Store in output as `out[freq_bin * spec_width + frame]`
     (frequency-major), eight frames at a time so each store run fills a
     cache line.

At the default 400 MHz, a 384-frame STFT of a 2.5 s chunk takes 33 ms. The
FFT size is any power of two from 32 to 512; the working buffers are static and
sized for 512.

### `spec_minmax_normalize()`

```c
void spec_minmax_normalize(float *spec, uint32_t count);
```

Maps a finished spectrogram to [0, 1] in place as
`(S - min) / (max - min + 1e-10)`, which is what the host does to every
spectrogram before the model sees it. `main.c` calls it after the STFT on the
hybrid path and after the mel projection on the precomputed path.

**Output layout:** `[fft_bins, spec_width]` — frequency-major (each row is one
frequency bin across all time frames). This matches the expected layout
`[B, fft_bins, spec_width, 1]` for the hybrid frontend and feeds the mel stage
for the librosa frontend.

!!! note "Why frequency-major?"
    The TFLite model's first layer expects input shaped `[B, filters, T, 1]`.
    By storing frequency-major on the firmware side, the `memcpy` to the NPU
    input buffer preserves the correct layout without a transpose.

### Stack Usage

Working buffers are stack-allocated:
- Hann window: 512 × 4 = 2,048 bytes
- FFT work buffer: 512 × 4 = 2,048 bytes
- **Total: ~4 KB stack** per call

---

## `audio_mel.c` — Mel Filterbank

**Location:** `firmware/Src/audio_mel.c`

Computes a triangular mel-frequency filterbank. It is compiled and used for
the `librosa` frontend (`APP_FRONTEND_PRECOMPUTED` in the C enum).

### `mel_filterbank()`
```c
void mel_filterbank(const float *stft_mag, uint32_t fft_bins,
                    uint32_t spec_width, uint32_t num_mels,
                    float *mel_out);
```
**Algorithm:**
1. `mel_init()` builds a dense Slaney-normalized triangular weight matrix once
   during startup.
2. `mel_filterbank()` applies it to the frequency-major
   `[APP_FFT_BINS, spec_width]` spectrogram.
3. It produces an `[APP_NUM_MELS, spec_width]` output array.

This reproduces librosa's Slaney-normalized mel weight matrices natively on the Cortex-M55 CPU.

---

## CMSIS-DSP (vendored)

**Location:** `firmware/Drivers/CMSIS-DSP/`

An unmodified subset of [CMSIS-DSP](https://github.com/ARM-software/CMSIS-DSP)
`v1.16.2` (Apache-2.0): the headers and the ten sources `arm_rfft_fast_f32` and
`arm_cmplx_mag_f32` need. The build defines `ARM_MATH_HELIUM`, so the FFT uses
the Cortex-M55's vector extension, and GCC needs `-flax-vector-conversions` for
that code. The native tests compile the same sources on the host with
`__GNUC_PYTHON__`, CMSIS-DSP's portable path.

The plain-C radix-2 FFT it replaced (up to 1.5) took 69 ms for the same STFT;
the board's scores are unchanged.

---

## `sd_handler.c` — SD Card + FatFs

**Location:** `firmware/Src/sd_handler.c`

Manages the SD card via BSP_SD (SDMMC2) and FatFs filesystem.

### Functions

| Function | Purpose |
|---|---|
| `sd_mount()` | Init BSP SD, link FatFs diskio driver, mount filesystem |
| `sd_unmount()` | Unmount filesystem, de-init SD |
| `sd_scan_audio_dir(dir, list)` | Enumerate up to 512 `.wav` files in `dir` |
| `sd_write_header(path, classes, n)` | Write TSV header row to results file |
| `sd_append_result(path, name, scores, n)` | Append one TSV row with all class scores |

The current `main.c` uses the mount and scan functions only. Result writer
helpers remain available for experiments, but the board-test workflow reports
predictions over UART.

### `sd_scan_audio_dir()`

Uses FatFs `f_opendir` / `f_readdir` to enumerate files:

- Non-recursive (flat directory only).
- Checks extension: `.wav` or `.WAV`.
- Stores full paths (`/audio/filename.wav`) in an `SdFileList` struct.
- Limited to `SD_MAX_FILES` (512) entries.

### `sd_append_result()`

Writes scores as integer-formatted values because FatFs's `f_printf` doesn't
support `%f`. Scores are multiplied by 10,000 and written as integers (e.g.,
`9230` for 0.923).

### Data Structures

```c
#define SD_MAX_PATH  256
#define SD_MAX_FILES 512

typedef struct {
    char paths[SD_MAX_FILES][SD_MAX_PATH];
    uint32_t count;
} SdFileList;
```

!!! warning "Memory usage"
    `SdFileList` is ~128 KB. It's declared `static` in `main.c` to keep it off
    the stack.

---

## Board Support Files (NPU_Validation)

These files come from ST's NPU_Validation example project and are **not
modified** by the board-test workflow:

| File | Purpose |
|---|---|
| `misc_toolbox.c/h` | UART config, NPU config, RISAF config; `UartHandle` global |
| `mcu_cache.c/h` | CPU cache enable/clean/invalidate helpers |
| `npu_cache.c/h` | NPU-specific cache management |
| `system_clock_config.c` | Clock tree setup (HSI overdrive, PLL config) |
| `stm32n6xx_it.c` | Interrupt handlers (SysTick, HardFault, etc.) |
| `syscalls.c` | Newlib stubs (`_write`, `_read`, `_sbrk`) for printf/malloc |
| `sysmem.c` | Heap region definition |
| `startup_stm32n657xx.s` | Vector table, reset handler, initial stack pointer |

These provide the foundation that our application code builds on. They handle
clock trees, voltage rails, cache management, and the NPU register interface.
