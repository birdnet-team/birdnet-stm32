# Source Modules

Detailed reference for every firmware C source file. The firmware is intentionally
compact — ~700 lines of application code across 5 files, plus a 230-line FFT.

## `main.c` — Orchestrator

**Location:** `firmware/Src/main.c` (~310 lines)

The entry point. Initializes the board, mounts the SD card, loops over WAV
files, and coordinates the STFT → NPU → output pipeline.

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
2. `memcpy` spectrogram into the NPU input buffer.
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
| `run_inference()` | `bool run_inference(const float *, float *)` | Copy spec to NPU, run, copy scores back |
| `print_top_k()` | `void print_top_k(const char *, const float *, int)` | Print top-K over UART |
| `__io_putchar()` | `int __io_putchar(int)` | UART printf redirect |
| `aiValidationInit()` | `static void aiValidationInit(void)` | GDB breakpoint stub (**do not remove**) |
| `Error_Handler()` | `void Error_Handler(void)` | HAL error hook (infinite loop) |
| `assert_failed()` | `void assert_failed(uint8_t *, uint32_t)` | HAL assert hook (infinite loop) |

---

## `wav_reader.c` — WAV File Parser

**Location:** `firmware/Src/wav_reader.c` (~130 lines)

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

**Location:** `firmware/Src/audio_stft.c` (~68 lines)

Computes a Hann-windowed magnitude STFT using `fft.c`.

### `stft_magnitude()`

```c
void stft_magnitude(const float *audio, uint32_t num_samples,
                    uint32_t fft_length, uint32_t hop_length,
                    uint32_t spec_width, float *out);
```

**Algorithm:**

1. Pre-compute a Hann window of `fft_length` samples (done once, cached).
2. For each of `spec_width` time frames:
   - Extract `fft_length` samples starting at `frame × hop_length`.
   - Multiply by the Hann window.
   - Call `fft_512_real()` for the FFT.
   - Compute magnitude: `sqrt(re² + im²)` for each of 257 bins.
   - Store in output as `out[freq_bin * spec_width + frame]`
     (frequency-major).
3. Zero-fill if the audio is shorter than expected.

**Output layout:** `[fft_bins, spec_width]` — frequency-major (each row is one
frequency bin across all time frames). This matches the hybrid frontend's
expected input tensor layout `[B, fft_bins, spec_width, 1]`.

!!! note "Why frequency-major?"
    The TFLite model's first layer expects input shaped `[B, fft_bins, T, 1]`.
    By storing frequency-major on the firmware side, the `memcpy` to the NPU
    input buffer preserves the correct layout without a transpose.

### Stack Usage

Working buffers are stack-allocated:
- Hann window: 512 × 4 = 2,048 bytes
- FFT work buffer: 512 × 4 = 2,048 bytes
- **Total: ~4 KB stack** per call

---

## `fft.c` — 512-Point Real FFT

**Location:** `firmware/Src/fft.c` (~230 lines)

A self-contained radix-2 Decimation-in-Time (DIT) FFT with **zero external
dependencies**.

### Why Not CMSIS-DSP?

The CMSIS-DSP `arm_rfft_fast_f32` requires linking `libarm_cortexM55l_math.a`,
which adds ~200 KB to the binary and complicates the Makefile. Our custom
512-point FFT is 230 lines of plain C, has no dependencies, and runs in ~0.05 ms
per frame at 800 MHz. For a 256-frame spectrogram the entire STFT takes
~25–35 ms — negligible compared to the 3 s audio chunk.

### Algorithm

1. **Input packing**: treat 512 real samples as 256 complex pairs
   (`x[2k] + j·x[2k+1]`).
2. **Bit-reversal permutation**: reorder the 256 complex values for in-place
   butterfly computation.
3. **Butterfly stages**: 8 stages of radix-2 DIT butterflies with precomputed
   twiddle factors (`cos` + `j·sin`).
4. **Split-radix unpack**: decompose the 256-point complex FFT result into 257
   real-FFT bins (DC through Nyquist) using even/odd twiddle factors.

### Output Format

CMSIS-DSP compatible layout:

| Index | Content |
|---|---|
| `buf[0]` | DC component (real) |
| `buf[1]` | Nyquist component (real) |
| `buf[2k]`, `buf[2k+1]` | Real and imaginary parts of bin _k_ (k = 1..255) |

### Performance

| Metric | Value |
|---|---|
| Per-frame time | ~0.05 ms @ 800 MHz |
| 256 frames | ~13 ms (FFT only, excludes windowing and magnitude) |
| Table init | One-time, first call only |
| Static memory | ~4 KB (twiddle + bit-reversal tables) |

### Limitations

The FFT size is **hardcoded at 512** (256 complex points). The twiddle tables,
bit-reversal tables, and buffer sizes are all compile-time constants. To support
other FFT sizes, the code would need generalization or conditional compilation.

---

## `sd_handler.c` — SD Card + FatFs

**Location:** `firmware/Src/sd_handler.c` (~120 lines)

Manages the SD card via BSP_SD (SDMMC2) and FatFs filesystem.

### Functions

| Function | Purpose |
|---|---|
| `sd_mount()` | Init BSP SD, link FatFs diskio driver, mount filesystem |
| `sd_unmount()` | Unmount filesystem, de-init SD |
| `sd_scan_audio_dir(dir, list)` | Enumerate up to 512 `.wav` files in `dir` |
| `sd_write_header(path, classes, n)` | Write TSV header row to results file |
| `sd_append_result(path, name, scores, n)` | Append one TSV row with all class scores |

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
