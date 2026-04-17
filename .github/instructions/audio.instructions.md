---
description: "Use when modifying audio ingestion, spectrograms, data generator pipelines, and augmentations."
applyTo: "birdnet_stm32/audio/**/*.py"
---

# Audio Processing Conventions

When working in `birdnet_stm32/audio/`, strictly observe these rules:

## Resampling
- Always `fast_resample(waveform, orig_sr, target_sr)` or specify sampling rates strictly.
- Always assume normalized [-1.0, 1.0] `float32` arrays for internal python operations.
- Avoid passing integers to downstream `tf.keras.layers`.

## Signal Processing & Shape Rules
- Keep STFT operations isolated (e.g. `get_spectrogram_from_audio`). Wait, STFT on the host is usually `librosa.stft` or `tf.signal.stft`. If building a layer, it's `get_spectrogram_from_audio`.
- Audio Frontend Modes (`librosa`, `hybrid`, `raw`):
  - `librosa`: Host-side computationally intensive Mel transform.
  - `hybrid`: Offline STFT (Cortex-M55) + learned Mel mixer on NPU.
  - `raw`: No STFT, pure waveform feeding an NPU 1D-like layout.
- The `raw` frontend supports lengths <= 65536 samples on STM32N6 (e.g., 24000Hz * 2s = 48000). Avoid going above.

## Dtypes & Scaling Constraints
- `tf.float32` must be used throughout the network build until PTQ (quantization) converts internals to `tf.int8`.
- Model I/O remains `tf.float32`. Do not force inputs to be integer values in the Python data pipeline before feeding `model.predict`.
- Keep the `pwl` (piecewise-linear) scaling path as the default since it's quantization friendly for ST edge AI deployment. Avoid `db` (log) as it produces poor INT8 ranges.