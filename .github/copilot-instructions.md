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

# Optuna hyperparameter search (pip install -e ".[tune]")
python -m birdnet_stm32 train --data_path_train data/train --tune --n_trials 20 --epochs 30
```

## Architecture

- **Audio frontend** (`birdnet_stm32/models/frontend.py`): Five modes — `librosa` (precomputed mel), `hybrid` (linear STFT + learned mel mixer), `raw` (waveform → learned Gabor quadrature filterbank), `mfcc` (precomputed MFCC), `log_mel` (precomputed log-mel). The firmware supports `raw`, `hybrid`, and `librosa`; `mfcc` and `log_mel` remain host-preprocessed paths.
- **Magnitude scaling**: `pwl` (piecewise-linear, default, quantization-friendly), `pcen`, `db` (avoid — poor quantization). Decoupled in `birdnet_stm32/models/magnitude.py`.
- **Model**: DS-CNN (depthwise-separable CNN) with 4 stages, ReLU6, global avg pool → dropout → dense. Scaled via `alpha` (channel multiplier) and `depth_multiplier` (block repeats). Optional attention pooling (`--use_attention_pooling`).
- **Building blocks** (`birdnet_stm32/models/blocks.py`): channel alignment and optional attention pooling.
- **Model profiler** (`birdnet_stm32/models/profiler.py`): Per-layer MACs, params, activation memory, N6 compatibility check. Printed by `train` in place of `model.summary()`.
- **Quantization**: Post-training quantization (PTQ) with representative dataset calibration (stratified sampling + SNR filtering). Float32 I/O, INT8 internals. Per-channel (default) or per-tensor (`--per_tensor`). Dynamic range mode (`--quantization dynamic`). Batch validation (`--batch_validate N`). ONNX export (`--export_onnx`). JSON conversion report (`--report_json`).
- **QAT**: Quantization-aware training via shadow-weight fake-quantization (`--qat`). Freezes BN, injects INT8 noise into kernels during fine-tuning. No FakeQuant ops in saved model — N6 compatible. Implemented in `birdnet_stm32/training/qat.py`.
- **Training pipeline**: Always-multi-label sigmoid + binary crossentropy head. Linear warmup into cosine LR decay, checkpoint/early-stopping on val ROC-AUC, resume (`--resume`), gradient clipping (`--grad_clip`, default 1.0), mixed precision (`--mixed_precision`), Dirichlet multi-source mixup, smart crop for long recordings, Optuna hyperparameter tuning (`--tune`, `birdnet_stm32/training/tuner.py`), linear probing (`--linear_probe`, `birdnet_stm32/training/linear_probe.py`).
- **Data pipeline** (`birdnet_stm32/data/generator.py`): Multiprocessing pool (`--num_workers`, default 8) bypasses GIL for parallel FLAC decode + resample + spectrogram. Multi-chunk extraction (`--max_chunks_per_file`, default 3) reuses long file opens by extracting multiple salient chunks per decode, buffered in a shuffled in-memory reservoir (~135 MB) for batch diversity.
- **Deployment**: `stedgeai generate` → `n6_loader.py` (serial flash) → `stedgeai validate` (on-device).

## Workflow

- **Language**: All code, comments, docs, and commit messages must be in American English.
- **Documentation**: Document often — add docstrings to public functions, update docs when behavior changes.
- **Commits**: One semantic unit per commit. One-line commit messages (imperative mood, e.g., "Add lme pooling to evaluation pipeline").
- **Linting**: Always run `ruff check .` (and fix any errors) before committing. Zero warnings policy — all code must pass `ruff check` cleanly.
- **Formatting**: Always run `ruff format birdnet_stm32/ tests/` before committing. CI runs `ruff format --check` and rejects unformatted code.

## Conventions

- **Dataset layout**: `data/{train,test}/<species_name>/*.wav`. Special folder names (`noise`, `silence`, `background`, `other`) get all-zero label vectors.
- **Checkpoint outputs**: `{name}.keras`, `{name}_model_config.json`, `{name}_labels.txt`, `{name}_history.csv`, `{name}_curves.png`, `{name}_train_state.json`, `{name}_quantized.tflite`, `{name}_quantized_validation_data.npz`.
- **Public model names**: `BirdNET_Tiny_N6_<REGION>_<SPECIES_COUNT>_V<MAJOR.MINOR>`. The count excludes nuisance outputs. Release `.keras`, `.tflite`, and `.onnx` files share the exact basename; stage them under gitignored `release/<basename>/` only after validation passes.
- **Model config**: `ModelConfig` dataclass in `birdnet_stm32/training/config.py` — validated, JSON-serializable, backward-compatible with legacy configs.
- **Config files**: `config.json` (gcc/CubeIDE paths), `config_n6l.json` (N6 loader mappings). These are machine-local — don't hardcode paths.
- **Eval runs**: CSV results stored in `report/eval_runs/` with naming `{run_number}_{frontend}_{mag}_{alpha}_{depth}_{embed}_{batch}_{maxsamples}.csv`.

## Pitfalls

- **N6 compatibility is the absolute priority.** Every model, layer, and quantization decision must be verified against the STM32N6 NPU operator set. Verify via `stedgeai analyze` / `stedgeai generate`.
- **Raw frontend sizes**: The N6 limits standard input arrays dynamically transferring from M55 to 65536 samples (16-bit size limit). E.g., `24kHz × 2.5s` is safe. Exceeding (like `24kHz × 3s`) requires falling back to `hybrid` / `librosa` or shorter chunks.
- **NPU conv strides**: measured with `stedgeai analyze`, the N6 runs convolutions with stride > 2 in **software**. Any large-hop "learned STFT" must be expressed as a folded stride-2 convolution (see `raw_filterbank_geometry`), or the whole filterbank lands on the Cortex-M55.
- **Quantization similarity**: Overly diverse representative datasets widen INT8 ranges → worse cosine similarity. Target > 0.95 cosine sim in `convert.py` output.
- **Channel alignment**: Keep channel counts as multiples of 8 for NPU vectorization.
- **On-device validation**: Requires physical USB at 921600 baud to STM32N6570-DK.
- **Board test firmware must be standalone.** The `board-test` command deploys real firmware that reads WAV from SD, applies frontend-specific preprocessing on the board, runs NPU inference, and streams results over UART. Do not precompute test inputs on the host.
