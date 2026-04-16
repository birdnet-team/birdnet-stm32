# Firmware: SD Card Batch Inference

Standalone application for the **STM32N6570-DK** that reads WAV files from an
SD card, runs NPU inference, and writes detection results back to the card.

## How it works

```
SD:/audio/*.wav → WAV parser → STFT (CMSIS-DSP) → NPU inference → SD:/results.txt
                                                                 → UART console
```

1. Mount SD card (FatFs + SDMMC).
2. Load class labels from `SD:/labels.txt` (optional — falls back to `class_N`).
3. Scan `SD:/audio/` for `.wav` files.
4. For each file:
   - Read 3-second chunks of PCM16 audio.
   - Compute 512-point STFT with Hann window → `[257, 256]` magnitude spectrogram.
   - Copy spectrogram to NPU input buffer → run inference → read class scores.
   - Average scores across chunks (multi-chunk pooling).
5. Write tab-separated results to `SD:/results.txt`.
6. Echo top-K detections over UART (115200 baud).

## Prerequisites

- **STM32N6570-DK** board with SD card inserted.
- **X-CUBE-AI 10.2.0** (provides LL_ATON runtime, stedgeai, n6_loader).
- **ARM GCC** toolchain (arm-none-eabi-gcc 13+).
- **STM32CubeN6** HAL/BSP package (provides SDMMC, FatFs, clock config).
- **CMSIS-DSP** (included in ARM GCC or STM32CubeN6).

## SD card layout

```
SD card root/
├── audio/              ← WAV files to process (22050 Hz, PCM16, mono or stereo)
│   ├── recording_001.wav
│   ├── recording_002.wav
│   └── ...
├── labels.txt          ← (optional) one class name per line, matching model
└── results.txt         ← written by firmware after processing
```

## Build instructions

### Option A: Integrate into NPU\_Validation project

The simplest approach — overlay our source files into the existing NPU validation
project that is already set up for the N6-DK.

```bash
# 1. Generate NPU artifacts from your model
python -m birdnet_stm32 deploy  # or just the generate step

# 2. Copy application source into the NPU_Validation project
NPU_VAL="$X_CUBE_AI_PATH/Projects/STM32N6570-DK/Applications/NPU_Validation"
cp firmware/Src/main.c       "$NPU_VAL/Core/Src/app_main.c"
cp firmware/Src/wav_reader.c "$NPU_VAL/Core/Src/"
cp firmware/Src/audio_stft.c "$NPU_VAL/Core/Src/"
cp firmware/Src/sd_handler.c "$NPU_VAL/Core/Src/"
cp firmware/Inc/*.h          "$NPU_VAL/Core/Inc/"

# 3. Add FatFs and SDMMC to the project
#    - In STM32CubeIDE: enable SDMMC1 + FatFs middleware
#    - Or manually add the FatFs source files and SDMMC BSP driver

# 4. Update the NPU_Validation Makefile/project to:
#    - Compile our new .c files
#    - Link CMSIS-DSP (libarm_cortexM55l_math.a)
#    - Replace the default main() with our app_main.c
#    - Add -I paths for our Inc/ headers

# 5. Build
cd "$NPU_VAL"
make -j$(nproc) BUILD_CONF=N6-DK

# 6. Flash via n6_loader
python -m birdnet_stm32 deploy  # flashes the built binary
```

### Option B: Standalone STM32CubeIDE project

1. Create a new STM32CubeIDE project for STM32N6570-DK.
2. Enable peripherals: SDMMC1, USART1, ICACHE, DCACHE.
3. Enable middleware: FatFs (SD card mode).
4. Add CMSIS-DSP library to the linker.
5. Import the LL_ATON runtime from X-CUBE-AI.
6. Copy `firmware/Src/*.c` and `firmware/Inc/*.h` into the project.
7. Copy the stedgeai-generated `network.c` into `X-CUBE-AI/App/`.
8. Build and flash.

## Configuration

Edit `firmware/Inc/app_config.h` to match your model:

| Define | Default | Description |
|---|---|---|
| `APP_SAMPLE_RATE` | 22050 | Audio sample rate (must match training) |
| `APP_CHUNK_DURATION` | 3 | Chunk length in seconds |
| `APP_FFT_LENGTH` | 512 | FFT window size |
| `APP_HOP_LENGTH` | 258 | STFT hop length |
| `APP_SPEC_WIDTH` | 256 | Number of STFT time frames |
| `APP_NUM_CLASSES` | 10 | Number of output classes |
| `APP_TOP_K` | 5 | Top-K results printed per file |
| `APP_SCORE_THRESHOLD` | 0.1 | Minimum score to display |

## Host-side test runner

The Python CLI can orchestrate the test and monitor serial output:

```bash
python -m birdnet_stm32 board-test \
  --model_path checkpoints/best_model_quantized.tflite \
  --labels checkpoints/best_model_labels.txt \
  --serial_port /dev/ttyACM0 \
  --timeout 300 \
  --save_output board_test_output.txt
```

## Output format

### Serial (UART)

```
=== BirdNET-STM32 SD Card Inference ===
[OK] NPU network initialised
[OK] SD card mounted
[OK] 10 classes loaded
[OK] Found 5 audio files in /audio/

[1/5] recording_001.wav
    [1] Turdus merula_Common Blackbird: 92.3%
    [2] Erithacus rubecula_European Robin: 15.1%

[2/5] recording_002.wav
    [1] Parus major_Great Tit: 87.6%

=== DONE ===
Processed: 5 / 5 files (0 errors)
Results written to /results.txt
```

### results.txt (TSV)

```
filename	class_0	class_1	...	class_9
recording_001.wav	0.0123	0.9230	...	0.0045
recording_002.wav	0.8760	0.0034	...	0.0012
```

## Memory requirements

| Buffer | Size | Location |
|---|---|---|
| Audio chunk (66150 × f32) | ~258 KB | cpuRAM2 / hyperRAM |
| Spectrogram (257 × 256 × f32) | ~263 KB | cpuRAM2 / hyperRAM |
| NPU input tensor | ~263 KB | npuRAM5 (managed by runtime) |
| NPU activations | ~321 KB | npuRAM5 |
| NPU weights | ~275 KB | octoFlash (read-only) |
| File list (512 entries) | ~128 KB | .bss |

Total application RAM: ~650 KB + NPU runtime.  The STM32N6 has several MB of
SRAM, so this fits comfortably.
