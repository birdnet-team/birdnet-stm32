# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0] — 2026-04-20

### Added

- **Memory profiling** (`--profile_memory`): measures peak RSS and delta during inference via `resource.getrusage`.
- **Deploy CLI enhancements**: `--dry_run` (print commands without executing), `--skip_validate` (skip on-target validation), colored ANSI terminal output (auto-disabled when not a tty), auto-detect board on `/dev/ttyACM*`.
- **Config TOML migration**: `config.toml.example` now has `[deploy]`, `[build]`, and `[n6_loader]` sections; config resolver auto-generates n6_loader JSON from TOML `[n6_loader]` table.
- **Setup and download scripts**: `scripts/setup_stm32.sh` (toolchain check), `scripts/download_checkpoints.sh`, `scripts/download_data.sh` (placeholders for release assets).
- **Deploy config tests**: 6 new tests for config resolution, TOML fallback, and board detection.
- **Quantization-Aware Training (QAT)** (`--qat`): shadow-weight fake-quantization fine-tuning for Keras 3. Freezes BatchNorm, injects INT8 noise into kernel weights during training, maintains FP32 shadow weights with STE-like gradient transfer. No FakeQuant ops in saved model — full N6 NPU compatibility. Improves quantized model accuracy (cmAP +1.5pp, ROC-AUC +0.8pp on 10-class test set).
- `extra_callbacks` parameter for `train_model()` to support QAT and other custom callbacks.
- **Species-level AP report** (`--species_report`): per-species average precision with bootstrap confidence intervals (`--n_bootstrap`).
- **DET curve** (`--det_curve`, `--save_det_plot`): Detection Error Tradeoff curve (FAR vs FRR) — ASCII and matplotlib output.
- **Benchmark mode** (`--benchmark`): structured JSON report with all metrics, per-species AP, latency stats, and model config.
- **Latency measurement** (`--benchmark_latency`): per-chunk inference timing with mean/median/p95/p99 statistics.
- **HTML evaluation report** (`--report_html`): self-contained HTML with inline CSS, summary metrics table, per-species AP table, and confusion matrix heatmap (base64 matplotlib).
- Dev guide docs: implementation notes, adding-a-frontend, adding-a-model, experiment-tracking, release-process.
- Integration CI workflow (weekly + manual trigger).

## [0.7.0] — 2026-04-19

### Added

- **Optuna hyperparameter tuning** (`--tune`, `--n_trials`): searches over alpha, depth_multiplier, embeddings_size, learning_rate, dropout, batch_size, mixup_alpha, label_smoothing, optimizer, weight_decay, grad_clip, use_se, use_inverted_residual, use_attention_pooling, se_reduction, expansion_factor. Maximizes val_roc_auc with MedianPruner.
- **Per-channel / per-tensor quantization** (`--per_tensor`): per-channel (default, more accurate) or per-tensor (simpler, use if N6 rejects per-channel).
- **Dynamic range quantization** (`--quantization dynamic`): INT8 weights with runtime float activations — no calibration data needed.
- **Stratified representative dataset**: calibration sampling now draws equal samples per class with SNR filtering (near-silent chunks skipped).
- **Batch validation** (`--batch_validate N`): run Keras-vs-TFLite validation N times with different seeds, report worst-case metrics.
- **ONNX export** (`--export_onnx`): exports `.onnx` alongside `.tflite` (requires `tf2onnx`).
- **Conversion report** (`--report_json`): structured JSON with validation metrics, compression ratio, model sizes, and config.
- **Float32 I/O runtime assertion**: `convert_to_tflite()` now verifies the quantized model preserved float32 I/O after conversion.
- **`pip install -e ".[all]"`**: meta extras group pulling in dev + docs + deploy + tune dependencies.
- **ModelConfig dataclass** (`birdnet_stm32/training/config.py`): validated, JSON-serializable, backward-compatible.
- **Resumable training** (`--resume`): reloads model + optimizer state from checkpoint.
- **Gradient clipping** (`--grad_clip`): max gradient norm for optimizer.
- **Mixed precision** (`--mixed_precision`): FP16 compute, FP32 accumulation.
- **Balanced class weights** (`--class_weights balanced`): inverse-frequency weighting.
- **LR finder** (`birdnet_stm32/training/lr_finder.py`): LR range test utility.
- **Training dashboard**: CSV history (`_history.csv`) + training curves PNG (`_curves.png`).

### Changed

- Representative dataset generator now uses stratified class sampling instead of random shuffle.
- Cosine similarity function handles near-zero vectors gracefully (both-zero = perfect match for noise/background class predictions).
- Removed stale `setuptools-scm` build requirement from `pyproject.toml`.
- Removed deprecated license classifier (PEP 639 compliance).

## [0.6.0] — 2026-04-19

### Added

- **MFCC frontend** (`--audio_frontend mfcc`): mel spectrogram → power-to-dB → librosa DCT. Configurable via `--n_mfcc` (default 20).
- **Log-mel frontend** (`--audio_frontend log_mel`): mel spectrogram → log1p → normalize. Lightweight alternative to librosa precompute.
- **Squeeze-and-excite (SE) blocks** (`--use_se`): channel attention after each DS block. NPU-compatible (GAP + Dense + Sigmoid + Multiply).
- **MobileNetV2-style inverted residual blocks** (`--use_inverted_residual`): expand → DW → project with configurable `--expansion_factor`.
- **Attention pooling** (`--use_attention_pooling`): learned spatial attention replacing GlobalAveragePooling2D.
- **Label smoothing** (`--label_smoothing`): applies to CategoricalCrossentropy (single-label) or BinaryCrossentropy (multilabel/mixup).
- **Knowledge distillation** (`birdnet_stm32/training/distillation.py`): `DistillationLoss` combining hard labels with soft teacher logits (KL divergence, configurable temperature and alpha).
- **Model registry** (`birdnet_stm32/models/__init__.py`): `build_model(name, **kwargs)`, `register_model()`, `list_models()` dispatcher pattern.
- **Model profiler** (`birdnet_stm32/models/profiler.py`): per-layer MACs, params, activation memory. N6 NPU compatibility check with `N6_SUPPORTED_OPS` and `N6_WARN_OPS` sets.
- **Frontend registry** (`birdnet_stm32/models/registry.py`): `FrontendInfo` dataclass with N6 compatibility metadata. `register_frontend()`, `get_frontend_info()`, `list_frontends()`.
- **Species list utilities** (`birdnet_stm32/data/species.py`): `load_species_list()`, `save_species_list()`, `combine_species_lists()` extracted from dev scripts.
- **Beta distribution mixup** (`use_beta=True` in `apply_mixup()`): sample mixing weights from Beta(α, α) instead of uniform.
- Test suite for frontend registry and new spectrogram modes (`tests/test_frontend_registry.py`).

### Changed

- **Default sample rate**: 22050 → 24000 Hz across all CLI defaults, audio I/O, and data generators.
- **Unified DS-CNN model**: SE, inverted residual, and attention pooling options are now flags on the single `build_dscnn_model()` function (removed separate `dscnn_se` module).
- **`_make_divisible()` moved** from `dscnn.py` to `blocks.py` to avoid circular imports; `blocks.py` is now the canonical source for all building blocks.
- Deploy config (`birdnet_stm32/deploy/config.py`) now supports TOML config files alongside JSON, with env vars `STEDGEAI_PATH`, `CUBEIDE_PATH`, `ARM_TOOLCHAIN_PATH`.
- CLI deploy command accepts `--stedgeai_path`, `--model`, `--cubeide_path`, `--arm_toolchain_path`.
- Top-level `train.py`, `test.py`, `convert.py` are now thin wrappers with deprecation warnings, delegating to the package CLI.

### Fixed

- Label smoothing with mixup: now correctly uses `BinaryCrossentropy(label_smoothing=...)` when mixup is active (sigmoid output), instead of `CategoricalCrossentropy`.
- `pick_random_samples()` `pick_first` logic: when `pick_first=True` and `num_samples > 1`, first sample is always included plus random picks from remaining.
- Model profiler handles Keras 3 layers that lack `output_shape` attribute (e.g., `InputLayer`).

### Removed

- `TERMS_OF_USE.txt` (redundant with `TERMS_OF_USE.md`).
- `birdnet_stm32/models/dscnn_se.py` (merged into `dscnn.py`).
- `notes.txt` (moved to `dev/notes.md`).

## [0.5.0] — 2026-04-17

### Added

- `gen_app_config.py` — single source of truth for generating `app_config.h` and `app_labels.h` from `model_config.json` + labels file. Used by both `make configure` and `board_test.py`.
- `make configure` target in firmware Makefile — generates firmware headers from model config without running the full board-test pipeline.
- `USE_OVERDRIVE` compile-time flag — selects between non-overdrive (CPU @ 600 MHz, NPU @ 800 MHz, default) and overdrive (CPU @ 800 MHz, NPU @ 1 GHz) clock configurations.
- Fractional chunk duration support — `APP_CHUNK_SAMPLES` is now computed as `int(sample_rate × chunk_duration)` and emitted as a literal integer, avoiding truncation from integer-only C macro arithmetic (e.g., 2.9 s × 22050 Hz = 63945 samples).
- NPU_Validation board support defines (`USE_UART_BAUDRATE`, `USE_USB_PACKET_SIZE`, `USE_OVERDRIVE`, `NUCLEO_N6_CONFIG`, etc.) included in generated `app_config.h` with `#ifndef` guards.
- Raw UART output printed during board-test for easier debugging of firmware errors.
- Serial capture reads all summary lines after `=== DONE ===` marker (Processed + Benchmark).

### Changed

- `board_test.py` delegates header generation to `gen_app_config.py` (via `importlib`) instead of duplicating the logic.
- `_patch_app_config()` now fully replaces `app_config.h` instead of appending to the NPU_Validation original — eliminates dependency on the original file's `#endif` guard format.
- `main.c` clock init is now conditional on `#if USE_OVERDRIVE` with a non-overdrive fallback calling `SystemClock_Config_HSI_no_overdrive()`.

### Fixed

- Board-test compilation failures when building inside NPU_Validation tree due to missing `USE_UART_BAUDRATE` and `USE_USB_PACKET_SIZE` defines.
- Integer truncation of fractional chunk durations (e.g., 2.9 → 3) causing wrong `APP_CHUNK_SAMPLES` and model input shape mismatches.

## [0.3.0] — 2026-04-16

### Added

- `normalize_frontend_name()` with canonical names (`librosa`, `hybrid`, `raw`) and deprecation warnings for legacy aliases (`precomputed` → `librosa`, `tf` → `raw`)
- `MagnitudeScalingLayer` — standalone composable Keras layer for magnitude scaling (`pwl`, `pcen`, `db`, `none`), decoupled from `AudioFrontendLayer`
- SpecAugment augmentation (`--spec_augment`, `--freq_mask_max`, `--time_mask_max`) with frequency and time masking
- `BinaryFocalLoss` for imbalanced datasets (`--loss focal`, `--focal_gamma`)
- Configurable optimizer (`--optimizer adam|sgd|adamw`), weight decay (`--weight_decay`), and dropout rate (`--dropout`)
- Deterministic training mode (`--deterministic`, `--seed`) — sets all RNG seeds and TF deterministic ops
- Automatic cosine similarity threshold in conversion (`--min_cosine_sim`, default 0.95) — fails conversion if below threshold
- Confusion matrix output (`--confusion_matrix` for ASCII, `--save_cm_plot` for matplotlib image)
- Per-class threshold optimization via precision-recall curve (`--optimize_thresholds`)
- New test suites: `test_spec_augment`, `test_focal_loss`, `test_magnitude`, `test_optimizer`, `test_threshold_opt`

### Changed

- `AudioFrontendLayer` now delegates magnitude scaling to `MagnitudeScalingLayer`
- `validate_models()` returns a metrics dict instead of printing only
- All frontend name lookups normalized through `normalize_frontend_name()` across the codebase

## [0.2.0]

### Added

- Project scaffolding: CODE_OF_CONDUCT, CONTRIBUTING, CITATION.cff, SECURITY, CHANGELOG
- `pyproject.toml` with dev/docs dependency groups
- Pre-commit hooks (ruff, yaml, whitespace)
- `birdnet_stm32/` Python package structure
- Test framework with pytest fixtures and synthetic audio data
- `config.example.json` replacing hardcoded paths

### Changed

- Refactored flat scripts into `birdnet_stm32/` package modules
- Replaced `deploy.sh` hardcoded paths with config resolution (env vars, config file, CLI args)
- Updated `.gitignore` for new project structure

### Removed

- Hardcoded personal paths from `deploy.sh`, `config.json`, `config_n6l.json`
