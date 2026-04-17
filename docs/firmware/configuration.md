# Configuration

All firmware parameters are `#define` constants in two auto-generated headers.
The `board-test` Python command writes these from your `model_config.json` and
labels file. If you're building manually, you must set them yourself.

## `app_config.h` — Audio & Inference Parameters

| Define | Example | Description |
|---|---|---|
| `APP_SAMPLE_RATE` | `24000` | Audio sample rate in Hz. **Must match** the model's training rate. WAV files with a different rate are skipped. |
| `APP_CHUNK_DURATION` | `3` | Chunk length in seconds. |
| `APP_CHUNK_SAMPLES` | `72000` | Total samples per chunk (`SR × duration`). Determines `audio_buf` size. |
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

!!! tip "Consistency is critical"
    If `app_config.h` values don't match the TFLite model's expectations, the
    spectrogram dimensions will be wrong and NPU inference will produce garbage
    or crash. The automated `board-test` command ensures consistency by reading
    `model_config.json`.

### How Values Are Derived

The `board-test` orchestrator reads these fields from `model_config.json`:

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
APP_CHUNK_SAMPLES = sample_rate × chunk_duration
APP_FFT_BINS      = fft_length / 2 + 1
APP_HOP_LENGTH    = (sample_rate × chunk_duration - 1) / (spec_width - 1)
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
    The firmware does **not** resample. If your model was trained at 24 kHz, all
    test WAV files must also be 24 kHz. The `board-test` command resamples files
    automatically when copying them to the SD card — if you prepare the card
    manually, ensure the rates match.

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
