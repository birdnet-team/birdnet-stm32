# Project Guidelines

BirdNET-STM32: bird sound classification for edge deployment on STM32N6570-DK with NPU. Pipeline: train DS-CNN → quantize to INT8 TFLite → deploy via X-CUBE-AI/stedgeai.

## Build and Test

```bash
# Install
pip install -e ".[dev]"  # needs CUDA-enabled TensorFlow

# Train (outputs .keras + _model_config.json + _labels.txt)
python -m birdnet_stm32 train --data_path_train data/train --audio_frontend hybrid --mag_scale pwl \
  --alpha 1 --depth_multiplier 1 --embeddings_size 256 --batch_size 32 --max_samples 500

# Convert to quantized TFLite (outputs _quantized.tflite + _validation_data.npz)
python -m birdnet_stm32 convert --checkpoint_path checkpoints/best_model.keras \
  --model_config checkpoints/best_model_model_config.json --data_path_train data/train

# Evaluate (metrics: ROC-AUC, cmAP, F1)
python -m birdnet_stm32 evaluate --model_path checkpoints/best_model_quantized.tflite \
  --model_config checkpoints/best_model_model_config.json \
  --data_path_test data/test --pooling lme

# Deploy/Test on board (requires USB-connected STM32N6570-DK + config.json)
python -m birdnet_stm32 board-test --config config.json
```

## Architecture

- **Audio frontend** (`birdnet_stm32/models/frontend.py`): Three modes — `librosa` (precomputed mel), `hybrid` (offline STFT + learned mel mixer), `raw` (waveform → learned filterbank). `hybrid` or `raw` are the deployment options (`raw` achieves 0ms STFT overhead but uses more NPU).
- **Magnitude scaling**: `pwl` (piecewise-linear, default, quantization-friendly), `pcen`, `db` (avoid — poor quantization). Decoupled in `birdnet_stm32/models/magnitude.py`.
- **Model**: DS-CNN (depthwise-separable CNN) with 4 stages, ReLU6, global avg pool → dropout → dense. Scaled via `alpha` (channel multiplier) and `depth_multiplier` (block repeats).
- **Quantization**: Post-training quantization (PTQ) with representative dataset calibration. Float32 I/O, INT8 internals.
- **Deployment**: `stedgeai generate` → `n6_loader.py` (serial flash) → `stedgeai validate` (on-device).

## Workflow

- **Language**: All code, comments, docs, and commit messages must be in American English.
- **Documentation**: Document often — add docstrings to public functions, update docs when behavior changes.
- **Commits**: One semantic unit per commit. One-line commit messages (imperative mood, e.g., "Add lme pooling to evaluation pipeline").

## Conventions

- **Dataset layout**: `data/{train,test}/<species_name>/*.wav`. Special folder names (`noise`, `silence`, `background`, `other`) get all-zero label vectors.
- **Checkpoint outputs**: `{name}.keras`, `{name}_model_config.json`, `{name}_labels.txt`, `{name}_quantized.tflite`, `{name}_quantized_validation_data.npz`.
- **Config files**: `config.json` (gcc/CubeIDE paths), `config_n6l.json` (N6 loader mappings). These are machine-local — don't hardcode paths.
- **Eval runs**: CSV results stored in `report/eval_runs/` with naming `{run_number}_{frontend}_{mag}_{alpha}_{depth}_{embed}_{batch}_{maxsamples}.csv`.

## Pitfalls

- **N6 compatibility is the absolute priority.** Every model, layer, and quantization decision must be verified against the STM32N6 NPU operator set. Verify via `stedgeai analyze` / `stedgeai generate`.
- **Raw frontend sizes**: The N6 limits standard input arrays dynamically transferring from M55 to 65536 samples (16-bit size limit). E.g., `24kHz × 2.0s` is safe. Exceeding (like `22kHz × 3s`) requires falling back to `hybrid` / `librosa` or shorter chunks.
- **Quantization similarity**: Overly diverse representative datasets widen INT8 ranges → worse cosine similarity. Target > 0.95 cosine sim in `convert.py` output.
- **Channel alignment**: Keep channel counts as multiples of 8 for NPU vectorization.
- **On-device validation**: Requires physical USB at 921600 baud to STM32N6570-DK.
- **Board test firmware must be standalone.** The `board-test` command must deploy real firmware that does all processing on the board: read WAV from SD card → compute STFT on Cortex-M55 → run NPU inference → write results to SD card + serial. Do NOT precompute spectrograms on the host — that defeats the purpose of an integration test.
