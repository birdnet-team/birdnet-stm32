# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
