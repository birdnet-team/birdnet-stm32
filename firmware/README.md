# Firmware: BirdNET-STM32 On-Device Inference

Standalone bare-metal application for the **STM32N6570-DK** development board.
Reads WAV files from an SD card, computes a Short-Time Fourier Transform (STFT)
on the Cortex-M55 CPU, runs neural network inference on the dedicated NPU, and
reports bird species detections over UART and to the SD card.

> **Design principle:** The firmware is a self-contained integration test and
> demo. Everything runs on the board — no host preprocessing, no streaming, no
> RTOS. This makes it easy to validate the full pipeline (audio → spectrogram →
> NPU → classification) in isolation.

---

## Table of Contents

1. [Hardware Overview](#hardware-overview)
2. [Architecture](#architecture)
3. [Processing Pipeline](#processing-pipeline)
4. [Module Reference](#module-reference)
5. [Memory Layout](#memory-layout)
6. [Build System](#build-system)
7. [Configuration](#configuration)
8. [SD Card Layout](#sd-card-layout)
9. [UART Protocol](#uart-protocol)
10. [Benchmark Timing](#benchmark-timing)
11. [Pitfalls and Debugging Notes](#pitfalls-and-debugging-notes)
12. [Third-Party Components](#third-party-components)

---

## Hardware Overview

### STM32N6570-DK Board

The [STM32N6570-DK](https://www.st.com/en/evaluation-tools/stm32n6570-dk.html)
is ST's first discovery kit with a hardware Neural Processing Unit (NPU). Key
specs:

| Feature | Detail |
|---|---|
| **MCU** | STM32N657X0H3QU — Arm Cortex-M55 @ 600 MHz (800 MHz with overdrive) |
| **NPU** | ST Neural-ART accelerator, 1.2 TOPS (INT8) |
| **Internal SRAM** | 4.2 MB total (cpuRAM1/2/3, npuRAM1–6, flexRAM) |
| **External RAM** | 256 Mbit octal HyperRAM (XSPI port 1) |
| **External Flash** | 1 Gbit octal NOR (XSPI port 2) — model weights go here |
| **SD card** | microSD via SDMMC2, 4-bit bus, up to 208 MHz |
| **Debug** | ST-LINK V3 (SWD + VCP UART on USB) |
| **UART** | USART1 via ST-LINK VCP at 921,600 baud |

### Cortex-M55 CPU

The M55 is an Armv8.1-M core with:
- **Helium (MVE)** — SIMD extensions for DSP (not used in our plain-C FFT, but
  available for future optimization).
- **Dual DCache/ICache** — coherency with NPU requires explicit cache
  management via `SCB_CleanDCache_by_Addr()` / `SCB_InvalidateDCache_by_Addr()`.
- **TrustZone** — the N6 boots in secure mode; the NPU_Validation project
  handles the secure-to-nonsecure transition. Our firmware runs in privileged
  secure mode.

### Neural-ART NPU

The NPU is a hardware accelerator for INT8 convolutional neural networks:
- Supports Conv2D, DepthwiseConv2D, Dense, Pool, Add, ReLU, Softmax, and
  more (see `stedgeai analyze` for the full operator list).
- Operates on its own SRAM banks (npuRAM1–6) with DMA-like data movement.
- The CPU communicates with the NPU via the **LL_ATON** runtime API.
- Weights are stored in external NOR flash (memory-mapped via XSPI) and
  streamed to the NPU during inference.
- Activations live in npuRAM (internal SRAM), not external memory.

### Memory Map (Simplified)

```
0x2400_0000 .. 0x2440_0000   cpuRAM1 (256 KB)  — stack, small globals
0x2440_0000 .. 0x2480_0000   cpuRAM2 (256 KB)  — audio_buf, spec_buf
0x2480_0000 .. 0x24C0_0000   cpuRAM3 (256 KB)  — heap, FatFs buffers
0x3400_0000 .. 0x3460_0000   npuRAM1–6 (total ~1.5 MB) — NPU I/O + activations
0x7000_0000 .. 0x7200_0000   External HyperRAM (32 MB, memory-mapped)
0x7200_0000 .. 0x7A00_0000   External NOR flash (128 MB, memory-mapped)
```

The linker script (`STM32N657xx.ld`) places `.text` and `.rodata` in internal
flash/SRAM, and model weights are flashed to external NOR by the `n6_loader`
tool at deployment time.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.c (orchestrator)                     │
│                                                                  │
│  Board init ─► SD mount ─► Scan WAVs ─► Per-file loop:          │
│    ┌──────────┐   ┌───────────┐   ┌──────────────┐              │
│    │wav_reader│──►│audio_stft │──►│ NPU runtime  │──► UART/SD   │
│    │  (.c/.h) │   │  (.c/.h)  │   │ (LL_ATON)    │              │
│    └──────────┘   └───────────┘   └──────────────┘              │
│         ▲              ▲                  ▲                       │
│    sd_handler     fft.c (512pt)    network.c (generated)         │
│    FatFs+BSP_SD   plain-C radix-2  stedgeai output               │
└─────────────────────────────────────────────────────────────────┘
```

### Module Responsibilities

| Module | File(s) | Purpose |
|---|---|---|
| **Orchestrator** | `main.c` | Init board, loop over files, coordinate pipeline |
| **WAV reader** | `wav_reader.c/h` | Parse RIFF headers, read PCM16 → float32 |
| **STFT** | `audio_stft.c/h` | Hann-windowed STFT via `fft.c` |
| **FFT** | `fft.c/h` | 512-point real FFT (radix-2 DIT, plain C) |
| **SD handler** | `sd_handler.c/h` | BSP_SD init, FatFs mount, dir scan, results write |
| **NPU runtime** | `network.c/h` + LL_ATON | Generated by stedgeai; manages NPU execution |
| **Board support** | `misc_toolbox`, `mcu_cache`, `npu_cache`, `system_clock_config` | Clock trees, caches, UART, NPU config (from NPU_Validation) |

---

## Processing Pipeline

For each WAV file on the SD card, the firmware executes:

### Step 1: WAV Read (`wav_reader.c`)

```
SD card → SDMMC2 → FatFs → f_read() → PCM16 buffer → float32 conversion
```

- Parses the RIFF/WAVE header to extract sample rate, bit depth, channels.
- Validates sample rate matches `APP_SAMPLE_RATE` (model training rate).
- Reads the first `APP_CHUNK_SAMPLES` samples (e.g., 63,945 for 22050 Hz × 2.9 s).
- Converts 16-bit signed integers to float32 in `[-1.0, 1.0]` range.
- Mono extraction: for stereo files, takes channel 0 only.
- Zero-pads if the file is shorter than one chunk.

### Step 2: STFT (`audio_stft.c` + `fft.c`)

```
float32 audio → Hann window → 512-pt FFT → magnitude → [257 × 256] spectrogram
```

- **Window**: Hann window of length `APP_FFT_LENGTH` (512 samples).
- **Hop**: `APP_HOP_LENGTH` samples between frames (typically 281 for 256
  frames from a 72,000-sample chunk).
- **FFT**: Custom plain-C 512-point real FFT using the "N/2 complex FFT"
  trick:
  1. Treat 512 real samples as 256 complex pairs.
  2. Run a 256-point radix-2 DIT complex FFT (in-place, bit-reversal +
     butterfly stages).
  3. Unpack into 257 real-FFT bins (DC through Nyquist) using split-radix
     post-processing.
  4. Twiddle factors are precomputed once on first call.
- **Magnitude**: `|bin| = sqrt(re² + im²)` for each of the 257 frequency bins.
- **Output layout**: `[fft_bins, spec_width]` = `[257, 256]`, frequency-major
  (row = one frequency bin across all time frames). This matches the hybrid
  frontend's expected input tensor layout `[B, fft_bins, spec_width, 1]`.

**Why a custom FFT instead of CMSIS-DSP?** The CMSIS-DSP `arm_rfft_fast_f32`
requires linking `libarm_cortexM55l_math.a`, which adds ~200 KB to the binary
and complicates the Makefile. Our custom 512-point FFT is ~230 lines of C, has
zero dependencies, and runs in ~0.1 ms per frame at 600 MHz. For a 256-frame
spectrogram, the entire STFT takes ~25–35 ms — negligible compared to the 3 s
audio chunk.

### Step 3: NPU Inference (`network.c` + LL_ATON)

```
spec_buf → memcpy to NPU input → SCB_CleanDCache → LL_ATON_RT_Main()
         → SCB_InvalidateDCache → memcpy from NPU output → scores[NUM_CLASSES]
```

- **Input tensor**: `[1, 257, 256, 1]` float32 spectrogram. The model's
  `AudioFrontendLayer` (hybrid mode) is baked into the TFLite graph: it applies
  the mel filter bank (1×1 Conv2D), ReLU, per-sample normalization, and
  piecewise-linear (PWL) magnitude scaling — all on the NPU.
- **Output tensor**: `[1, NUM_CLASSES]` float32 class probabilities (softmax).
- **Cache coherency**: The CPU must clean the DCache before the NPU reads the
  input (`SCB_CleanDCache_by_Addr`) and invalidate after the NPU writes the
  output (`SCB_InvalidateDCache_by_Addr`). Missing these calls causes stale
  data and garbage predictions.
- **LL_ATON API**:
  - `LL_ATON_Input_Buffers_Info_Default()` — query input buffer address/shape.
  - `LL_ATON_Output_Buffers_Info_Default()` — query output buffer address/shape.
  - `LL_ATON_RT_Main(&NN_Instance_Default)` — run inference (blocking).
  - The `network.c` file (generated by `stedgeai generate`) contains the model
    graph description and weight pointers.

### Step 4: Results Output

- **UART**: Top-K class predictions printed over USART1 (921,600 baud) via
  `printf()`. The host Python script (`board_test.py`) captures and parses
  this output.
- **SD card** (optional): TSV results file written via `sd_append_result()`.
- **Benchmark**: Per-file timing (SD read, STFT, NPU) and aggregate averages
  printed as `[BENCH]` lines.

---

## Module Reference

### `main.c` — Application Entry Point

**Board initialization sequence** (mirrors the NPU_Validation reference):

1. `set_vector_table_addr()` — point VTOR to the correct vector table address
   (needed because n6_loader places code at a non-default address in SRAM).
2. `HAL_Init()` — initialize the HAL tick, NVIC priority grouping.
3. `SystemClock_Config_ResetClocks()` — reset all clock domains to a known
   state before reconfiguring.
4. `system_init_post()` — post-reset cleanup (clear pending interrupts, etc.).
5. `SCB_EnableICache()` / `SCB_EnableDCache()` — enable CPU caches.
6. **Clock configuration** — selected at compile time by `USE_OVERDRIVE`:
   - `USE_OVERDRIVE=0` (default): `SystemClock_Config_HSI_no_overdrive()` —
     CPU @ 600 MHz, NPU @ 800 MHz. No VDD core upscaling needed.
   - `USE_OVERDRIVE=1`: `upscale_vddcore_level()` + `SystemClock_Config_HSI_overdrive()`
     — CPU @ 800 MHz, NPU @ 1 GHz. Requires higher VDD core voltage.
7. `fuse_vddio()` — configure IO voltage rails for external memory interfaces.
8. `UART_Config()` — USART1 at 921,600 baud (ST-LINK VCP).
9. `BSP_XSPI_RAM_Init()` + `BSP_XSPI_NOR_Init()` — initialize and
    memory-map external HyperRAM and NOR flash. The NOR flash contains the NPU
    model weights; memory-mapping makes them accessible as regular read-only
    memory at `0x7200_0000`.
10. `NPU_Config()` — enable NPU clocks and configure the Neural-ART register
    interface (RISAF, security attributes).
11. `aiValidationInit()` — empty stub used by `n6_loader.py` as a GDB
    breakpoint during the flash process. The script sets a hardware breakpoint
    here, resumes execution, then detaches GDB once the board reaches this
    point. **Do not remove this function.**

### `wav_reader.c` — WAV File Parser

Parses standard RIFF/WAVE files with PCM encoding:
- Walks sub-chunks (`fmt `, `data`, and skips unknown chunks).
- Supports mono and stereo (extracts channel 0).
- Supports 16-bit PCM only (the most common WAV format).
- Does **not** support compressed formats (A-law, mu-law, ADPCM, float32 WAV).
- Returns the file position at the start of PCM data for subsequent reads.

### `audio_stft.c` — STFT Engine

Computes a Hann-windowed magnitude STFT:
- Stack-allocated working buffers (512 floats for window + 512 for FFT = 4 KB).
- Output is frequency-major: `out[f * spec_width + t]` — this matches the
  column-major layout expected by the TFLite model's hybrid frontend.
- Zero-pads the last frame if the audio chunk is shorter than expected.

### `fft.c` — 512-Point Real FFT

A self-contained radix-2 Decimation-in-Time (DIT) FFT:

**Algorithm:**
1. Interpret 512 real samples as 256 complex pairs.
2. Compute a 256-point complex FFT via bit-reversal permutation + log₂(256) = 8
   butterfly stages.
3. Unpack the complex result into 257 real-FFT bins using the "split radix"
   even/odd decomposition with twiddle factors.

**Output format** (CMSIS-DSP compatible):
- `buf[0]` = DC component (real).
- `buf[1]` = Nyquist component (real).
- `buf[2k], buf[2k+1]` = real and imaginary parts of bin k, for k = 1..255.

**Performance**: ~0.05 ms per frame at 800 MHz (estimated from cycle counts).
The twiddle table and bit-reversal table are computed once and cached in static
arrays.

### `sd_handler.c` — SD Card + FatFs

- Initializes the BSP SD driver (SDMMC2 peripheral, GPIO pins, DMA).
- Links the `sd_diskio` driver to FatFs and mounts the filesystem.
- `sd_scan_audio_dir()` — enumerates up to 512 `.wav` files in a flat
  directory (non-recursive). Files are stored as full paths in an `SdFileList`
  struct (~128 KB).
- `sd_append_result()` — writes one TSV row per file with all class scores.
  Uses integer formatting (`f_printf` doesn't support `%f`).

### Board Support (from NPU_Validation)

These files come from ST's NPU_Validation example project and are **not
modified** by the board-test workflow:

| File | Purpose |
|---|---|
| `misc_toolbox.c/h` | UART, NPU, RISAF configuration; `UartHandle` global |
| `mcu_cache.c/h` | CPU cache enable/clean/invalidate helpers |
| `npu_cache.c/h` | NPU-specific cache management |
| `system_clock_config.c` | Clock tree setup (overdrive and non-overdrive PLL configs) |
| `stm32n6xx_it.c` | Interrupt handlers (SysTick, HardFault, etc.) |
| `syscalls.c` | Newlib stubs (`_write`, `_read`, `_sbrk`) for printf/malloc |
| `sysmem.c` | Heap region definition |
| `startup_stm32n657xx.s` | Vector table, reset handler, stack setup |

---

## Memory Layout

### Static Buffers

| Buffer | Size | Type | Alignment |
|---|---|---|---|
| `audio_buf` | `APP_CHUNK_SAMPLES` × 4 bytes | float32 | 32-byte (DCache line) |
| `spec_buf` | `APP_FFT_BINS × APP_SPEC_WIDTH` × 4 bytes | float32 | 32-byte |
| `scores` | `APP_NUM_CLASSES` × 4 bytes | float32 | stack |
| `file_list` | `SD_MAX_FILES × SD_MAX_PATH` bytes | char | .bss |

**Example sizes** (22050 Hz, 2.9 s, raw frontend, 10 classes):
- `audio_buf`: 63,945 × 4 = 250 KB
- `spec_buf`: 257 × 256 × 4 = 263 KB (allocated but unused with raw frontend)
- `file_list`: 512 × 256 = 128 KB
- Total application static RAM: ~640 KB

### NPU Memory (Managed by LL_ATON)

The NPU runtime allocates its own input/output buffers and activation scratch
space in `npuRAM` banks. The exact sizes depend on the model topology and are
reported by `stedgeai analyze`. Typical values for the DS-CNN:

| Region | Size | Location |
|---|---|---|
| NPU input | ~263 KB | npuRAM (auto-placed) |
| NPU output | 40 bytes (10 classes × 4 B) | npuRAM |
| NPU activations | ~320 KB | npuRAM |
| NPU weights | ~200–300 KB | External NOR flash (read-only) |

---

## Build System

The firmware does **not** have a standalone build system. It is designed to be
overlaid onto ST's NPU_Validation project, which provides:
- Complete HAL and BSP driver set for the STM32N6570-DK.
- LL_ATON NPU runtime libraries.
- Linker script, startup code, and Makefile.
- GDB-based flash loader (`n6_loader.py`).

The firmware also has a **standalone Makefile** for compilation checks and
configuration. Use `make configure` to generate `app_config.h` and
`app_labels.h` from a model config:

```bash
# Generate firmware headers from model config (default: ../checkpoints/best_model_*)
make configure

# Or specify a different model:
make configure MODEL_CONFIG=../checkpoints/raw_model_model_config.json \
               LABELS_FILE=../checkpoints/raw_model_labels.txt
```

This runs `gen_app_config.py`, which reads `*_model_config.json` and
`*_labels.txt` to produce `Inc/app_config.h` and `Inc/app_labels.h` with the
correct sample rate, chunk duration, chunk samples, frontend mode, and class
labels.

### How `board-test` Builds the Firmware

The Python `board_test.py` orchestrator automates the entire process:

1. **`stedgeai generate`** — compiles the TFLite model into `network.c` +
   weight blobs for the NPU.
2. **Patch NPU_Validation** — copies our `firmware/` sources into the
   NPU_Validation project tree:
   - `firmware/Src/*.c` → `Core/Src/`
   - `firmware/Inc/*.h` → `Core/Inc/`
   - `firmware/Drivers/HAL_SD/` → `Drivers/.../Src/` and `.../Inc/`
   - `firmware/Drivers/stm32n6570_discovery_sd.*` → `Drivers/BSP/STM32N6570-DK/`
   - `firmware/Drivers/FatFs/` → `FatFs/` (new directory)
   - Auto-generates `app_config.h` values from `model_config.json`.
   - Auto-generates `app_labels.h` from the labels file.
   - Patches the Makefile to compile our added sources.
   - Enables `HAL_SD_MODULE_ENABLED` in `stm32n6xx_hal_conf.h`.
3. **`n6_loader.py`** — builds the firmware (ARM GCC), flashes external
   NOR (model weights), and loads + runs the binary via GDB.
4. **Cleanup** — restores all patched files from backups.

### Why Overlay Instead of Standalone?

The NPU_Validation project contains ~50 source files, pre-built libraries
(LL_ATON runtime), and complex linker scripts specific to the N6's multi-bank
SRAM layout. Reproducing this as a standalone project would be fragile and hard
to maintain across X-CUBE-AI versions. The overlay approach:
- Uses ST's tested build infrastructure.
- Stays compatible across X-CUBE-AI updates.
- Adds only the files we need (6 .c + 12 .h + FatFs).
- Cleans up after itself (backup/restore).

---

### `gen_app_config.py` — Header Generator

Single source of truth for generating firmware headers from model configuration.
Used by both `make configure` (standalone) and `board_test.py` (automated).

**Generated files:**

- `Inc/app_config.h` — all audio parameters, frontend mode, board support
  defines (`USE_OVERDRIVE`, `USE_UART_BAUDRATE`, `USE_USB_PACKET_SIZE`, etc.)
- `Inc/app_labels.h` — class name string array

**Usage:**

```bash
# Auto-detect labels file from model config path:
python gen_app_config.py ../checkpoints/best_model_model_config.json

# Explicit labels file:
python gen_app_config.py ../checkpoints/best_model_model_config.json \
                         ../checkpoints/best_model_labels.txt

# Custom output directory:
python gen_app_config.py model_config.json -o /path/to/output/
```

The script:

- Reads `sample_rate`, `chunk_duration`, `fft_length`, `spec_width`,
  `num_mels`, and `audio_frontend` from the model config JSON.
- Computes `APP_CHUNK_SAMPLES = int(sample_rate × chunk_duration)` as a literal
  integer, avoiding truncation issues with fractional chunk durations (e.g.,
  2.9 s → 63945 samples at 22050 Hz, not `22050 * 3 = 66150`).
- Includes all NPU_Validation board support defines wrapped in `#ifndef` guards
  so they can be overridden from the compiler command line.
- Auto-detects the labels file from the model config path
  (`best_model_model_config.json` → `best_model_labels.txt`).

### `Makefile` — Standalone Build & Configuration

The firmware Makefile provides:

- **`make configure`** — runs `gen_app_config.py` to generate headers:
  ```bash
  make configure
  make configure MODEL_CONFIG=../checkpoints/raw_model_model_config.json
  ```
- **`make all`** — standalone compilation check (links against BSP stubs; the
  full build happens inside the NPU_Validation tree via `n6_loader.py`).
- **`make clean`** — remove build artifacts.

## Configuration

### `app_config.h` — Firmware Parameters

Auto-generated by `gen_app_config.py` (or `make configure`) from
`model_config.json`. All audio and inference parameters are `#define`
constants that **must match** the values in the model config:

| Define | Example | Description |
|---|---|---|
| `APP_SAMPLE_RATE` | 24000 | Audio sample rate in Hz. Set to the model's training rate. |
| `APP_CHUNK_DURATION` | 2.9 | Chunk length in seconds (can be fractional). |
| `APP_CHUNK_SAMPLES` | 63945 | Total samples per chunk (`int(sr * duration)`). Computed as a literal integer to avoid truncation from integer-only C macro arithmetic. |
| `APP_FFT_LENGTH` | 512 | FFT window size. Must be 512 (hardcoded in `fft.c`). |
| `APP_FFT_BINS` | 257 | Frequency bins = FFT_LENGTH / 2 + 1 (computed). |
| `APP_HOP_LENGTH` | 172 | STFT hop in samples. Controls spec_width. |
| `APP_SPEC_WIDTH` | 256 | Number of STFT time frames (must match model). |
| `APP_NUM_CLASSES` | 10 | Number of output classes in the model. |
| `APP_TOP_K` | 5 | Top-K predictions to print per file. |
| `APP_SCORE_THRESHOLD` | 0.01 | Minimum score to include in output. |

### `USE_OVERDRIVE` — Clock Speed Selection

Set in the Makefile via `-DUSE_OVERDRIVE=0` (default) or `-DUSE_OVERDRIVE=1`.
Controls which clock configuration is used at startup:

| `USE_OVERDRIVE` | CPU Clock | NPU Clock | NPU RAM Clock | VDD Core |
|---|---|---|---|---|
| `0` (default) | 600 MHz | 800 MHz | 800 MHz | Nominal |
| `1` | 800 MHz | 1 GHz | 900 MHz | Upscaled via SMPS/I2C |

Overdrive mode provides maximum throughput but draws more power and requires
VDD core upscaling (`upscale_vddcore_level()`). The default non-overdrive
configuration is sufficient for real-time inference and is more conservative.

### `app_labels.h` — Class Label Array

Auto-generated C header with the class names as a `const char *` array. Used
for human-readable output over UART. Example:

```c
#define APP_NUM_CLASSES_ACTUAL 10
static const char * const APP_LABELS[] = {
    "Common Chaffinch",
    "Common Chiffchaff",
    /* ... */
};
```

---

## SD Card Layout

```
SD card (FAT32) root/
├── audio/                    ← Put test .wav files here
│   ├── recording_001.wav
│   ├── recording_002.wav
│   └── ...
└── results.txt               ← Written by firmware (optional)
```

**WAV file requirements:**
- Format: PCM, 16-bit signed integer.
- Sample rate: must match `APP_SAMPLE_RATE` (rejected on mismatch).
- Channels: mono preferred; stereo OK (channel 0 extracted).
- Duration: at least `APP_CHUNK_DURATION` seconds (shorter files zero-padded).
- Maximum files: 512 (limited by `SD_MAX_FILES`).

**SD card hardware notes:**
- The STM32N6570-DK uses **SDMMC2** (not SDMMC1) for the microSD slot.
- 4-bit bus mode, up to 208 MHz clock.
- The BSP driver handles GPIO configuration, clock gating, and DMA.
- Use a quality SD card (Class 10 / UHS-I recommended) — slow cards can cause
  FatFs timeout errors.

---

## UART Protocol

The firmware outputs structured text over USART1 at **921,600 baud** (8N1).
The Python host script parses these lines using regex:

### Per-File Output

```
[1/8] recording_001.wav
  [WAV] 24000 Hz, 16-bit, 1 ch, 72000 samples
  [BENCH] read=12ms stft=28ms npu=4ms total=44ms
  recording_001.wav:
    [1] Common Chiffchaff: 72.3%
    [2] Eurasian Blue Tit: 15.1%
```

### Summary (End of Run)

```
=== DONE ===
Processed: 8 / 8 files (0 errors)
Benchmark: read=96ms stft=224ms npu=32ms total=352ms (avg read=12ms stft=28ms npu=4ms total=44ms)
```

### Regex Patterns (Host-Side)

| Pattern | Matches |
|---|---|
| `^\[(\d+)/(\d+)\]\s+(.+)$` | File header: index, total, filename |
| `^\s+\[(\d+)\]\s+(.+?):\s+([\d.]+)%$` | Detection: rank, label, score% |
| `^\s+\[BENCH\]\s+read=(\d+)ms\s+stft=(\d+)ms\s+npu=(\d+)ms\s+total=(\d+)ms$` | Per-file timing |
| `^Benchmark:.*avg read=(\d+)ms ...` | Aggregate timing |
| `^Processed:\s+(\d+)\s*/\s*(\d+)` | Summary: processed/total |

---

## Benchmark Timing

Each stage is timed using `HAL_GetTick()` (1 ms resolution from SysTick):

| Stage | Typical Time | Notes |
|---|---|---|
| **SD read** | 10–20 ms | Depends on SD card speed and file size. 3 s @ 24 kHz = 144 KB. |
| **STFT** | 25–35 ms | 256 frames × 512-pt FFT. CPU @ 600 MHz (default). |
| **NPU inference** | 3–5 ms | Full DS-CNN including mel filter, PWL, and classifier. |
| **Total per file** | ~40–60 ms | Approximately 50–75× faster than real-time for 3 s chunks. |

The bottleneck is the STFT on the CPU. The NPU inference itself is extremely
fast (~3 ms), which makes sense — the NPU is a dedicated matrix engine running
INT8 convolutions.

**Optimization opportunities** (not yet implemented):
- Use CMSIS-DSP or Helium (MVE) intrinsics for the FFT — could halve STFT time.
- Pipeline SD reads with STFT computation (double-buffering).
- Process multiple chunks per file for long recordings.

---

## Pitfalls and Debugging Notes

### Cache Coherency

The #1 source of subtle bugs on the STM32N6. The CPU DCache and the NPU see
different views of SRAM:

- **Before NPU reads CPU data**: call `SCB_CleanDCache_by_Addr()` to flush
  dirty cache lines to SRAM.
- **After NPU writes results**: call `SCB_InvalidateDCache_by_Addr()` to
  discard stale cache lines.
- Buffers must be 32-byte aligned (cache line size) for these calls to work
  correctly.
- Missing cache ops → garbage input/output → wrong predictions or crashes.

### XSPI Initialization Order

The external memory init **must** happen before `NPU_Config()` because the NPU
weights are in external NOR flash. If the NOR is not memory-mapped when the NPU
tries to read weights, you get a bus fault.

The init sequence is:
1. `BSP_XSPI_RAM_Init(0)` + `BSP_XSPI_RAM_EnableMemoryMappedMode(0)`
2. Disable automatic prefetch: `MODIFY_REG(XSPI1->CR, XSPI_CR_NOPREF, ...)`
3. `BSP_XSPI_NOR_Init(0, &Flash)` + `BSP_XSPI_NOR_EnableMemoryMappedMode(0)`
4. Then `NPU_Config()`, `RISAF_Config()`, etc.

### TrustZone and RISAF

The STM32N6 boots in secure mode. The RISAF (Region Isolation and Security
Access Filter) must be configured to allow the NPU to access its SRAM banks.
`RISAF_Config()` (from `misc_toolbox.c`) handles this. If you see "NPU network
init failed", check that RISAF is configured before calling
`LL_ATON_EC_Network_Init_Default()`.

### GDB Breakpoint and `aiValidationInit()`

The `n6_loader.py` script uses GDB to:
1. Load the firmware binary into SRAM.
2. Set a breakpoint at `aiValidationInit()`.
3. Run the firmware, which initializes clocks and external memories.
4. When the breakpoint hits, GDB flashes external NOR with model weights.
5. GDB continues execution — the firmware proceeds with SD card processing.

If you rename or remove `aiValidationInit()`, the GDB flash process will fail.
The function body is just `__asm volatile("nop")` — it exists solely as a
breakpoint target.

### `assert_failed()` — Hard Fault Safety Net

The HAL calls `assert_failed()` when an `assert_param()` check fails. Our
`main.c` provides a stub that halts. If you see the board freeze, connect a
debugger and check whether execution stopped in `assert_failed` or
`Error_Handler`.

### SD Card Gotchas

- **8.3 filenames**: FatFs with `_USE_LFN=0` (default in the NPU_Validation
  config) only supports short 8.3 filenames. Long filenames are truncated to
  `~1` names (e.g., `BIODCA~1.WAV`). If you need long filenames, enable
  `_USE_LFN=1` in `ffconf.h`.
- **Card detection**: The BSP SD driver uses polling, not GPIO card-detect.
  Insert the SD card before powering the board.
- **FAT32 only**: exFAT requires a license. Use FAT32 for SD cards ≤ 32 GB.

### Sample Rate Mismatch

The firmware rejects WAV files whose sample rate doesn't match
`APP_SAMPLE_RATE`. This is intentional — resampling on the M55 is possible but
adds complexity. Train your model at the same sample rate as your test audio.

### FFT Size Fixed at 512

The `fft.c` implementation is hardcoded for N = 512 (256 complex points). The
twiddle tables, bit-reversal tables, and stack buffers are all sized for this.
To support other FFT sizes, the code would need generalization or multiple
compiled variants.

---

## Third-Party Components

| Component | License | Source |
|---|---|---|
| FatFs R0.15 | BSD-1-Clause | http://elm-chan.org/fsw/ff/ |
| STM32N6xx HAL | BSD-3-Clause | STM32CubeN6 |
| BSP STM32N6570-DK | BSD-3-Clause | STM32CubeN6 |
| LL_ATON NPU Runtime | ST Proprietary | X-CUBE-AI 10.2.0 |
| sd_diskio driver | ST Proprietary | STM32CubeN6 FatFs middleware |
