# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
