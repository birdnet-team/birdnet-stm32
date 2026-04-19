# Configuration

All firmware parameters are `#define` constants in two auto-generated headers.
The `board-test` Python command generates these from your `model_config.json`
and labels file. You can also generate them manually with `gen_app_config.py`
or `make configure`.

## Generating Headers

The recommended way to produce `app_config.h` and `app_labels.h` is through
the shared generator script:

```bash
# From the firmware/ directory:
make configure

# Or specify a different model:
make configure MODEL_CONFIG=../checkpoints/raw_model_model_config.json \
               LABELS_FILE=../checkpoints/raw_model_labels.txt

# Or call the script directly:
python gen_app_config.py ../checkpoints/best_model_model_config.json
```

The `board-test` command calls the same generator automatically.

## `app_config.h` — Audio & Inference Parameters

| Define | Example | Description |
|---|---|---|
| `APP_SAMPLE_RATE` | `24000` | Audio sample rate in Hz. **Must match** the model's training rate. WAV files with a different rate are skipped. |
| `APP_CHUNK_DURATION` | `2.9` | Chunk length in seconds. Can be fractional. |
| `APP_CHUNK_SAMPLES` | `63945` | Total samples per chunk (`int(SR × duration)`). Computed as an integer literal — avoids truncation from integer-only C macros. |
| `APP_FFT_LENGTH` | `512` | FFT window size. **Must be 512** — the FFT implementation is hardcoded for this size. |
| `APP_FFT_BINS` | `257` | Frequency bins (`FFT_LENGTH / 2 + 1`). Derived, do not set independently. |
| `APP_HOP_LENGTH` | `281` | STFT hop in samples. Controls how many time frames fit in one chunk. |
| `APP_SPEC_WIDTH` | `256` | Number of STFT time frames. **Must match** the model's expected input width. |
| `APP_NUM_MELS` | `64` | Number of mel bands for the `precomputed` frontend. |
| `APP_AUDIO_FRONTEND` | `0` | Frontend mode: `APP_FRONTEND_HYBRID` (0), `APP_FRONTEND_RAW` (1), or `APP_FRONTEND_PRECOMPUTED` (2). |
| `APP_NUM_CLASSES` | `10` | Number of output classes. **Must match** the model's output dimension. |
| `APP_TOP_K` | `5` | Number of top predictions printed per file. |
| `APP_SCORE_THRESHOLD` | `0.01` | Minimum score (0–1) to include in output. Below this, predictions are suppressed. |
| `APP_AUDIO_DIR` | `"audio"` | SD card subdirectory to scan for WAV files. |

### Board Support Defines

The generated header also includes defines required by the NPU_Validation
project's board support code (`misc_toolbox.c`, `system_clock_config.c`,
`aiPbIO.c`). These are wrapped in `#ifndef` guards so they can be overridden
from the compiler command line:

| Define | Default | Description |
|---|---|---|
| `NUCLEO_N6_CONFIG` | `0` | Board variant (0 = DK, not Nucleo). |
| `USE_MCU_DCACHE` | `1` | Enable CPU data cache. |
| `USE_MCU_ICACHE` | `1` | Enable CPU instruction cache. |
| `USE_EXTERNAL_MEMORY_DEVICES` | `1` | Enable XSPI RAM + NOR flash init. |
| `USE_UART_BAUDRATE` | `921600` | USART1 baud rate for UART output. |
| `USE_USB_PACKET_SIZE` | `512` | USB packet size (used by `aiPbIO.c`). |
| `USE_OVERDRIVE` | `0` | Clock speed selection (see below). |

### `USE_OVERDRIVE` — Clock Speed Selection

Controls which clock configuration is used at startup:

| `USE_OVERDRIVE` | CPU Clock | NPU Clock | VDD Core |
|---|---|---|---|
| `0` (default) | 600 MHz | 800 MHz | Nominal |
| `1` | 800 MHz | 1 GHz | Upscaled via SMPS/I2C |

Overdrive provides maximum throughput but draws more power and requires VDD
core upscaling. The default non-overdrive configuration is sufficient for
real-time inference (the NPU finishes a 3 s chunk in ~4 ms at 800 MHz).

!!! tip "Consistency is critical"
    If `app_config.h` values don't match the TFLite model's expectations, the
    spectrogram dimensions will be wrong and NPU inference will produce garbage
    or crash. Always use `gen_app_config.py` or `board-test` to ensure
    consistency.

### How Values Are Derived

`gen_app_config.py` reads these fields from `model_config.json`:

```json
{
    "sample_rate": 24000,
    "chunk_duration": 3,
    "fft_length": 512,
    "spec_width": 256,
    "num_mels": 64,
    "audio_frontend": "hybrid"
}
```

And computes:
```
APP_CHUNK_SAMPLES = int(sample_rate × chunk_duration)   # 63945
APP_FFT_BINS      = fft_length / 2 + 1
APP_HOP_LENGTH    = from model config (or fft_length // 2 + 2)
```

## `app_labels.h` — Class Names

Auto-generated C header mapping class indices to human-readable names:

```c
#define APP_NUM_CLASSES_ACTUAL 10
static const char * const APP_LABELS[] = {
    "Common Chaffinch",
    "Common Chiffchaff",
    "Eurasian Blue Tit",
    "Eurasian Magpie",
    "Eurasian Wren",
    "European Goldfinch",
    "European Robin",
    "Great Spotted Woodpecker",
    "Great Tit",
    "Song Thrush",
};
```

Generated from the `_labels.txt` file produced during training. The order must
match the model's output indices exactly.

## SD Card Layout

```
SD card (FAT32) root/
├── audio/                    ← Put test WAV files here
│   ├── recording_001.wav
│   ├── recording_002.wav
│   └── ...
└── results.txt               ← Written by firmware after processing
```

### WAV File Requirements

| Property | Requirement |
|---|---|
| **Format** | PCM, 16-bit signed integer |
| **Sample rate** | Must match `APP_SAMPLE_RATE` (files with wrong rate are skipped as errors) |
| **Channels** | Mono preferred; stereo OK (channel 0 extracted) |
| **Duration** | At least `APP_CHUNK_DURATION` seconds (shorter files are zero-padded) |
| **Maximum count** | 512 files (`SD_MAX_FILES` constant in `sd_handler.h`) |

!!! note "Sample rate mismatch"
    The firmware does **not** resample. If your model was trained at 22050 Hz,
    all test WAV files must also be 22050 Hz. Prepare matching files before
    copying them to the SD card.

### SD Card Hardware Notes

- The STM32N6570-DK uses **SDMMC2** (not SDMMC1) for the microSD slot.
- 4-bit bus mode, up to 208 MHz clock. Use a quality card (Class 10 / UHS-I).
- The BSP driver uses polling, not GPIO card-detect — insert the SD card
  **before** powering the board.
- **FAT32 only** — exFAT requires a license. Use FAT32 for cards ≤ 32 GB.

## Adapting for Your Model

To deploy a different model on the same firmware:

1. **Train and quantize** your model with the standard pipeline (`train.py` →
   `convert.py`). The model config JSON and labels file are produced
   automatically.

2. **Run `board-test`** — it reads the new model config and regenerates
   `app_config.h` and `app_labels.h` to match:
   ```bash
   python -m birdnet_stm32 board-test \
     --model_path checkpoints/my_new_model_quantized.tflite \
     --model_config checkpoints/my_new_model_model_config.json \
     --labels checkpoints/my_new_model_labels.txt \
     --config config.json
   ```

3. **Prepare test audio** at the correct sample rate and copy to the SD card.

Things you do **not** need to change in firmware source code:

- Number of classes (read from `app_config.h` at compile time).
- Sample rate, FFT size, spectrogram shape (all from `app_config.h`).
- Class names (from `app_labels.h`).

Things you **might** need to change:

- `SD_MAX_FILES` in `sd_handler.h` if you have more than 512 test files.
- `APP_TOP_K` or `APP_SCORE_THRESHOLD` if you want different output verbosity.
- Buffer alignment or placement if your model requires significantly more
  memory (unlikely for DS-CNN variants).
