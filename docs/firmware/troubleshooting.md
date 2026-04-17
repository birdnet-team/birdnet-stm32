# Troubleshooting

Common pitfalls, debugging hints, and known issues when working with the
STM32N6570-DK firmware.

## Cache Coherency

**The #1 source of subtle bugs on the STM32N6.**

The CPU DCache and the NPU see different views of SRAM. When you write data
from the CPU that the NPU needs to read (or vice versa), you must explicitly
manage the cache:

| Direction | Required Call | What It Does |
|---|---|---|
| CPU → NPU | `SCB_CleanDCache_by_Addr(ptr, size)` | Flushes dirty cache lines to SRAM so the NPU sees fresh data |
| NPU → CPU | `SCB_InvalidateDCache_by_Addr(ptr, size)` | Discards stale cache lines so the CPU reads the NPU's output |

**Requirements:**

- Buffers must be **32-byte aligned** (one DCache line). Use
  `__attribute__((aligned(32)))`.
- The `size` parameter must be a multiple of 32 bytes (round up).

**Symptoms of missing cache ops:**

- NPU reads stale input → wrong predictions (often all zeros or identical
  scores for every file).
- CPU reads stale output → garbage scores, NaN values.
- Intermittent failures (cache sometimes coincidentally coherent).

```c
// Correct pattern — from run_inference() in main.c
memcpy(input_ptr, spect, input_bytes);
SCB_CleanDCache_by_Addr((uint32_t *)input_ptr, (int32_t)input_bytes);  // (1)

LL_ATON_RT_Main(&NN_Instance_Default);

SCB_InvalidateDCache_by_Addr((uint32_t *)output_ptr, (int32_t)output_bytes);  // (2)
memcpy(output, output_ptr, output_bytes);
```

1. Flush CPU writes to SRAM before NPU reads.
2. Discard CPU cache after NPU writes.

## XSPI Initialization Order

External memory init **must** happen before `NPU_Config()`.

The NPU weights are stored in external NOR flash (memory-mapped at
`0x7200_0000`). If the NOR is not initialized and memory-mapped when the NPU
tries to read weights during `LL_ATON_EC_Network_Init_Default()`, you get a
**bus fault** (HardFault).

**Correct order:**

```
BSP_XSPI_RAM_Init(0)
BSP_XSPI_RAM_EnableMemoryMappedMode(0)
    ↓
BSP_XSPI_NOR_Init(0, &Flash)
BSP_XSPI_NOR_EnableMemoryMappedMode(0)
    ↓
NPU_Config()
RISAF_Config()
```

**Symptom:** HardFault immediately after `LL_ATON_EC_Network_Init_Default()`,
or the board appears to hang right after the `[INIT] Configuring NPU...` line.

## TrustZone and RISAF

The STM32N6 boots in secure mode. The RISAF (Region Isolation and Security
Access Filter) must be configured to allow the NPU to access its SRAM banks.

`RISAF_Config()` from `misc_toolbox.c` handles this. It opens the npuRAM
regions for NPU access.

**Symptom:** If RISAF is not configured before `LL_ATON_EC_Network_Init_Default()`:

- `[ERROR] NPU network init failed` on UART
- Or a silent HardFault (no UART output at all if the fault happens before
  UART is fully initialized)

## GDB Breakpoint: `aiValidationInit()`

The `n6_loader.py` flashing process depends on this function:

1. GDB loads the `.elf` into SRAM.
2. GDB sets a hardware breakpoint at `aiValidationInit()`.
3. GDB starts execution.
4. Firmware runs init sequence (clocks, XSPI, NOR memory-mapping).
5. Firmware hits the breakpoint.
6. GDB flashes model weights into NOR via `STM32_Programmer_CLI`.
7. GDB continues execution.

!!! danger "Do not rename or remove this function"
    The function body is just `__asm volatile("nop")`. It exists solely as a
    breakpoint target. If it's missing, the flash process hangs indefinitely
    waiting for the breakpoint to be hit.

**Symptom:** `n6_loader.py` hangs with "Waiting for breakpoint..." or similar.

## SD Card Issues

### 8.3 Filename Limitation

FatFs with `_USE_LFN=0` (default in NPU_Validation's `ffconf.h`) only supports
8.3 filenames. Longer names are truncated to `~1` form:

| Original Name | 8.3 Name |
|---|---|
| `Biodiversity_Recording_001.wav` | `BIODIV~1.WAV` |
| `forest_dawn_chorus.wav` | `FOREST~1.WAV` |

If you need long filenames, set `_USE_LFN=1` in `ffconf.h`. This requires
extra heap memory for the LFN working buffer.

### Card Not Detected

The BSP SD driver uses **polling**, not GPIO card-detect. You must:

1. Insert the SD card **before** powering the board.
2. Use a FAT32-formatted card (not exFAT).
3. Use a quality card — Class 10 or UHS-I recommended.

**Symptom:** `[ERROR] SD card mount failed` on UART.

### Slow Card Timeouts

Very slow cards can cause FatFs read timeouts. The BSP driver has a default
timeout of ~1 second for each read operation.

**Symptom:** Random `[SKIP] Cannot open file` errors, especially on large files.

**Fix:** Use a faster card, or increase the timeout in the BSP SD driver
configuration.

## Sample Rate Mismatch

The firmware **rejects** WAV files whose sample rate doesn't match
`APP_SAMPLE_RATE`:

```
[SKIP] Sample rate 22050 != 24000
```

This is intentional. Resampling on the Cortex-M55 is possible but adds code
complexity and processing time. The correct solution is to ensure test audio
matches the model's training sample rate.

!!! tip
    The `board-test` Python command resamples files automatically when
    preparing the SD card. If you prepare the card manually, use a tool like
    `sox` or `ffmpeg`:
    ```bash
    sox input.wav -r 24000 output.wav
    ffmpeg -i input.wav -ar 24000 output.wav
    ```

## FFT Size

The `fft.c` implementation is **hardcoded for N=512** (256 complex points).
The twiddle tables, bit-reversal tables, and stack buffers are all sized for 512.

If you need a different FFT size:

- **1024-point:** Would require doubling all tables and adding a 9th butterfly
  stage. Possible but requires code changes.
- **256-point:** Would require halving tables and removing a butterfly stage.
- **Arbitrary size:** Would require a fully general FFT implementation (e.g.,
  mixed-radix or split-radix).

For now, the model and firmware are locked to 512-point FFT. This provides
257 frequency bins and works well for the 24 kHz → 64-mel hybrid frontend.

## `assert_failed()` and `Error_Handler()`

The HAL calls `assert_failed()` when an `assert_param()` check fails, and
`Error_Handler()` for general errors. Both are infinite loops in our firmware.

**Debugging approach:**

1. Connect a debugger (ST-LINK + GDB or STM32CubeIDE).
2. Set breakpoints in `assert_failed()` and `Error_Handler()`.
3. Trigger the failure.
4. Inspect the call stack and the `file` / `line` parameters in
   `assert_failed()`.

If you don't have a debugger, the symptoms are that UART output stops abruptly
(no `=== DONE ===` marker, the board appears to hang).

## Common Error Messages

| UART Output | Cause | Fix |
|---|---|---|
| `[ERROR] xSPI NOR init failed` | External NOR flash not responding | Check board hardware, power cycle |
| `[ERROR] NPU network init failed` | NPU can't load model, or RISAF not configured | Check XSPI init order, RISAF config |
| `[ERROR] SD card mount failed` | SD card not inserted, not FAT32, or hardware issue | Re-insert card, reformat as FAT32 |
| `[SKIP] Sample rate X != Y` | WAV sample rate doesn't match model | Resample audio to correct rate |
| `[SKIP] Invalid WAV format` | Not a valid RIFF/WAVE PCM file | Convert to 16-bit PCM WAV |
| `[SKIP] Cannot open file` | FatFs can't open the file | Check 8.3 filename, card speed |
| `[ERROR] Inference failed` | NPU runtime error | Check model compatibility with `stedgeai analyze` |
| No output at all | Board not running, wrong serial port, wrong baud rate | Check USB connection, use 921600 baud |

## Debugging with GDB

For deeper investigation, connect via GDB:

```bash
arm-none-eabi-gdb build/NPU_Validation.elf
(gdb) target remote :3333      # ST-LINK GDB server
(gdb) monitor reset halt
(gdb) break main
(gdb) continue
```

Useful breakpoints:

- `Error_Handler` — catches HAL errors
- `assert_failed` — catches HAL assertion failures
- `HardFault_Handler` — catches memory access violations
- `run_inference` — inspect spectrogram and scores

## Third-Party Components

| Component | License | Source |
|---|---|---|
| FatFs R0.15 | BSD-1-Clause | [elm-chan.org](http://elm-chan.org/fsw/ff/) |
| STM32N6xx HAL | BSD-3-Clause | STM32CubeN6 |
| BSP STM32N6570-DK | BSD-3-Clause | STM32CubeN6 |
| LL_ATON NPU Runtime | ST Proprietary | X-CUBE-AI 10.2.0 |
| sd_diskio driver | ST Proprietary | STM32CubeN6 FatFs middleware |
