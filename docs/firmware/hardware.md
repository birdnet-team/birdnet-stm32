# Hardware

## STM32N6570-DK Board

The [STM32N6570-DK](https://www.st.com/en/evaluation-tools/stm32n6570-dk.html)
is ST's first discovery kit with a hardware Neural Processing Unit (NPU).

| Feature | Detail |
|---|---|
| **MCU** | STM32N657X0H3QU — Arm Cortex-M55 @ 800 MHz |
| **NPU** | ST Neural-ART accelerator, 1.2 TOPS (INT8) |
| **Internal SRAM** | 4.2 MB total (cpuRAM1/2/3, npuRAM1–6, flexRAM) |
| **External RAM** | 256 Mbit octal HyperRAM (XSPI port 1) |
| **External Flash** | 1 Gbit octal NOR (XSPI port 2) — model weights go here |
| **SD card** | microSD via SDMMC2, 4-bit bus, up to 208 MHz |
| **Debug** | ST-LINK V3 (SWD + VCP UART on USB) |
| **UART** | USART1 via ST-LINK VCP at 921,600 baud |

## Cortex-M55 CPU

The M55 is an Armv8.1-M core with:

- **Helium (MVE)** — SIMD extensions for DSP. Not used in our plain-C FFT, but
  available for future optimization (could roughly halve STFT time).
- **Dual DCache / ICache** — coherency with the NPU requires explicit cache
  management via `SCB_CleanDCache_by_Addr()` and
  `SCB_InvalidateDCache_by_Addr()`. This is the single most common source of
  bugs on this platform — see [Troubleshooting](troubleshooting.md#cache-coherency).
- **TrustZone** — the N6 boots in secure mode. The NPU_Validation project
  handles the secure-to-nonsecure transition. Our firmware runs in privileged
  secure mode throughout.

## Neural-ART NPU

The NPU is a hardware accelerator purpose-built for INT8 convolutional neural
networks:

- **Supported operators**: Conv2D, DepthwiseConv2D, Dense, Pool (avg/max), Add,
  ReLU/ReLU6, Softmax, Reshape, and more. Run `stedgeai analyze` on your model
  to check operator compatibility.
- **Memory access**: operates on its own SRAM banks (npuRAM1–6) with DMA-like
  data movement. The CPU communicates with the NPU via the **LL_ATON** runtime
  API.
- **Weights**: stored in external NOR flash, memory-mapped via XSPI, and
  streamed to the NPU during inference. This means the model size is limited by
  the NOR capacity (128 MB), not internal SRAM.
- **Activations**: live in npuRAM (internal SRAM), not external memory. The
  activation scratch size depends on the model topology and is reported by
  `stedgeai analyze`.

!!! tip "NPU compatibility is the absolute priority"
    Every model, layer, and quantization decision must be verified against the
    STM32N6 NPU operator set. Always run `stedgeai analyze` before committing
    model changes. See the [troubleshooting page](troubleshooting.md) for
    common pitfalls.

## Memory Map

```
Address Range                 Region              Typical Usage
─────────────────────────────────────────────────────────────────
0x2400_0000 .. 0x2440_0000    cpuRAM1 (256 KB)    Stack, small globals
0x2440_0000 .. 0x2480_0000    cpuRAM2 (256 KB)    audio_buf, spec_buf
0x2480_0000 .. 0x24C0_0000    cpuRAM3 (256 KB)    Heap, FatFs work area
0x3400_0000 .. 0x3460_0000    npuRAM1–6 (~1.5 MB) NPU I/O + activations
0x7000_0000 .. 0x7200_0000    HyperRAM (32 MB)    External RAM (memory-mapped)
0x7200_0000 .. 0x7A00_0000    NOR flash (128 MB)  NPU weights (memory-mapped)
```

The linker script (`STM32N657xx.ld`) places `.text` and `.rodata` in internal
SRAM, and model weights are flashed to external NOR by the `n6_loader` tool at
deployment time. The NOR memory-mapping is configured during the board init
sequence (see [Building & Flashing — Init Sequence](building.md#init-sequence)).

### Application Buffers

| Buffer | Size | Alignment | Notes |
|---|---|---|---|
| `audio_buf` | `CHUNK_SAMPLES × 4` bytes | 32 B (DCache line) | e.g. 72,000 × 4 = 288 KB @ 24 kHz × 3 s |
| `spec_buf` | `FFT_BINS × SPEC_WIDTH × 4` bytes | 32 B | e.g. 257 × 256 × 4 = 263 KB |
| `file_list` | `SD_MAX_FILES × SD_MAX_PATH` bytes | .bss | 512 × 256 = 128 KB |
| `scores` | `NUM_CLASSES × 4` bytes | stack | Tiny (40 B for 10 classes) |
| **Total** | **~680 KB** | | Fits comfortably in the 4.2 MB internal SRAM |

### NPU Memory

| Region | Typical Size | Location |
|---|---|---|
| NPU input | ~263 KB | npuRAM (auto-placed by LL_ATON) |
| NPU output | 40 bytes | npuRAM |
| NPU activations | ~320 KB | npuRAM |
| NPU weights | ~200–300 KB | External NOR flash (read-only) |

The exact sizes depend on the model and are reported by `stedgeai analyze`:

```bash
stedgeai analyze --model checkpoints/best_model_quantized.tflite --target stm32n6
```
