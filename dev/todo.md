# BirdNet-STM32 — Full Rewrite Plan

This document is the master plan for rewriting the birdnet-stm32 codebase from a research prototype into a production-quality, well-documented, testable, and deployable project.

---

## Table of Contents

1. [Project Scaffolding & Repo Hygiene](#1-project-scaffolding--repo-hygiene)
2. [MkDocs Material Documentation](#2-mkdocs-material-documentation)
3. [README Revision](#3-readme-revision)
4. [Code Refactor — Package Structure](#4-code-refactor--package-structure)
5. [Configuration & Paths](#5-configuration--paths)
6. [Audio Frontend Improvements](#6-audio-frontend-improvements)
7. [Model Architecture Improvements](#7-model-architecture-improvements)
8. [Training Pipeline](#8-training-pipeline)
9. [Quantization & Conversion](#9-quantization--conversion)
10. [Evaluation & Testing Pipeline](#10-evaluation--testing-pipeline)
11. [Setup & Deploy Scripts](#11-setup--deploy-scripts)
12. [Unit & Integration Tests](#12-unit--integration-tests)
13. [CI/CD](#13-cicd)
14. [Agent Docs & Dev Guides](#14-agent-docs--dev-guides)

---

## 1. Project Scaffolding & Repo Hygiene

### 1.1 New repo-level files

- [ ] `CODE_OF_CONDUCT.md` — Contributor Covenant v2.1
- [ ] `CONTRIBUTING.md` — contribution guide (setup, branching, PR process, code style, testing expectations)
- [ ] `CITATION.cff` — CFF for academic citation (authors, title, DOI placeholder, repo URL)
- [ ] `SECURITY.md` — vulnerability reporting policy
- [ ] `CHANGELOG.md` — keep-a-changelog format, start with `[Unreleased]`
- [ ] `.gitignore` — comprehensive Python/TF/IDE/OS ignores (verify current one, add `*.tflite`, `samples/`, `validation/st_ai_ws/`, `__pycache__/`, `.venv/`, etc.)
- [ ] `pyproject.toml` — replace bare `requirements.txt` with proper project metadata, `[project.optional-dependencies]` for dev/docs/deploy groups
- [ ] `Makefile` or `justfile` — common tasks: `make train`, `make convert`, `make test`, `make deploy`, `make docs`, `make lint`, `make test-unit`, `make test-integration`

### 1.2 Licensing cleanup

- [ ] Consolidate `LICENSE.md`, `TERMS_OF_USE.md`, `TERMS_OF_USE.txt` into a single `LICENSE` (or keep LICENSE + TERMS_OF_USE, but not both .md and .txt)
- [ ] Add SPDX license identifier to `pyproject.toml`

### 1.3 Pre-commit hooks

- [ ] `.pre-commit-config.yaml` with: ruff (lint+format), mypy (type checking stubs), check-yaml, trailing-whitespace, end-of-file-fixer
- [ ] Add ruff config section to `pyproject.toml`

---

## 2. MkDocs Material Documentation

### 2.1 Scaffold

- [ ] `mkdocs.yml` — Material theme, nav structure, plugins (search, mkdocstrings, gen-files)
- [ ] `docs/index.md` — landing page (project overview, badges, quick links)
- [ ] `docs/assets/` — images, diagrams

### 2.2 User Guide

- [ ] `docs/getting-started.md` — prerequisites, install, quick train/eval workflow
- [ ] `docs/dataset.md` — dataset format, folder structure, noise classes, species lists, downloading iNatSounds
- [ ] `docs/training.md` — full training guide with all CLI args, frontend selection, tips
- [ ] `docs/conversion.md` — Keras → TFLite PTQ, representative dataset, similarity validation
- [ ] `docs/evaluation.md` — test.py usage, metrics explained (ROC-AUC, cmAP, F1, LME pooling)
- [ ] `docs/deployment.md` — STM32N6570-DK setup, X-CUBE-AI install, stedgeai, n6_loader, on-device validation (absorb content from current README)

### 2.3 Developer Guide

- [ ] `docs/dev/architecture.md` — pipeline diagram (audio → frontend → DS-CNN → pool → metrics), component boundaries, data flow
- [ ] `docs/dev/audio-frontends.md` — deep-dive into precomputed/hybrid/raw modes, mag scaling, trainability, N6 constraints
- [ ] `docs/dev/model.md` — DS-CNN architecture, alpha/depth multiplier, channel alignment, residual blocks
- [ ] `docs/dev/quantization.md` — PTQ details, float32 I/O, representative dataset strategy, cosine similarity targets, NPU op coverage
- [ ] `docs/dev/testing.md` — how to run tests, add tests, test fixtures, CI integration
- [ ] `docs/dev/contributing.md` — link to CONTRIBUTING.md + dev-specific workflow (branch naming, commit style, review checklist)

### 2.4 API Reference

- [ ] `docs/api/` — auto-generated from docstrings via mkdocstrings for each module

### 2.5 Build & deploy docs

- [ ] GitHub Actions workflow to build + deploy to GitHub Pages on push to `main`
- [ ] `docs/requirements.txt` or extras in `pyproject.toml` for mkdocs + plugins

---

## 3. README Revision

The current README is ~420 lines and mixes quick-start with exhaustive deployment instructions. Rewrite to be concise:

- [ ] Badges: CI status, docs link, license, Python 3.12+
- [ ] One-paragraph description + hero image
- [ ] Quick Start (5-step: clone, install, train, convert, evaluate) — link to docs for details
- [ ] Model Zoo table (pre-trained checkpoints with metrics, download links)
- [ ] Deployment teaser with link to `docs/deployment.md`
- [ ] Citation block (from CITATION.cff)
- [ ] Contributing + License links
- [ ] Remove the multi-page deployment instructions (move to docs)

---

## 4. Code Refactor — Package Structure

### Current structure (flat)

```
train.py          # ~1450 lines: AudioFrontendLayer, DS-CNN, data gen, training
test.py           # ~550 lines: runners, pooling, evaluation, metrics, ASCII viz
convert.py        # ~350 lines: PTQ conversion, validation
deploy.sh         # hardcoded paths
utils/audio.py    # ~370 lines: audio loading, spectrogram, SNR, plotting
```

### Target structure

```
birdnet_stm32/
├── __init__.py
├── __main__.py              # CLI entry point dispatcher
├── cli/
│   ├── __init__.py
│   ├── train.py             # argparse + main() for training
│   ├── convert.py           # argparse + main() for PTQ conversion
│   ├── evaluate.py          # argparse + main() for evaluation
│   └── deploy.py            # argparse + main() for deployment (replaces deploy.sh)
├── audio/
│   ├── __init__.py
│   ├── io.py                # load_audio_file, save_wav, fast_resample
│   ├── spectrogram.py       # get_spectrogram_from_audio, normalize
│   ├── activity.py          # SNR, activity ratio, sort_by_activity, sort_by_s2n
│   └── augmentation.py      # mixup, future: time-stretch, pitch-shift, SpecAugment
├── data/
│   ├── __init__.py
│   ├── dataset.py           # load_file_paths, get_classes, upsample, tf.data pipeline
│   └── species.py           # species list utilities (from dev/make_dev_set.py)
├── models/
│   ├── __init__.py
│   ├── frontend.py          # AudioFrontendLayer (all 3 modes + mag scaling)
│   ├── dscnn.py             # build_dscnn_model, ds_conv_block, _make_divisible
│   └── runners.py           # KerasRunner, TFLiteRunner
├── training/
│   ├── __init__.py
│   ├── trainer.py           # train_model, LR schedule, callbacks
│   └── config.py            # ModelConfig dataclass, save/load JSON
├── conversion/
│   ├── __init__.py
│   ├── quantize.py          # TFLite converter, representative data gen
│   └── validate.py          # cosine/MSE/MAE/Pearson validation
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py           # ROC-AUC, cmAP, F1, AP per class
│   ├── pooling.py           # avg, max, lme_pooling, pool_scores
│   └── reporting.py         # ASCII histogram, PR curve, CSV export
├── deploy/
│   ├── __init__.py
│   ├── stedgeai.py          # stedgeai generate/validate wrappers
│   └── n6_loader.py         # n6_loader invocation wrapper
└── utils/
    ├── __init__.py
    ├── plotting.py           # plot_spectrogram, sanity check visualizations
    └── seed.py               # set_global_seed (numpy, tf, random)
```

### Refactoring tasks

- [ ] Create `birdnet_stm32/` package with `__init__.py`
- [ ] Extract `AudioFrontendLayer` from `train.py` → `birdnet_stm32/models/frontend.py`
- [ ] Extract `build_dscnn_model`, `ds_conv_block`, `_make_divisible` → `birdnet_stm32/models/dscnn.py`
- [ ] Extract `KerasRunner`, `TFLiteRunner` → `birdnet_stm32/models/runners.py`
- [ ] Extract `data_generator`, `load_dataset`, `load_file_paths_from_directory`, `upsample_minority_classes`, `get_classes_with_most_samples` → `birdnet_stm32/data/dataset.py`
- [ ] Split `utils/audio.py` into `birdnet_stm32/audio/{io,spectrogram,activity}.py`
- [ ] Extract `mixup` logic into `birdnet_stm32/audio/augmentation.py`
- [ ] Extract `lme_pooling`, `pool_scores` → `birdnet_stm32/evaluation/pooling.py`
- [ ] Extract metrics → `birdnet_stm32/evaluation/metrics.py`
- [ ] Extract ASCII viz + CSV → `birdnet_stm32/evaluation/reporting.py`
- [ ] Extract conversion logic → `birdnet_stm32/conversion/{quantize,validate}.py`
- [ ] Create `birdnet_stm32/training/config.py` with a `ModelConfig` dataclass replacing raw dict passing
- [ ] Create CLI entry points in `birdnet_stm32/cli/` wrapping each script
- [ ] Create `birdnet_stm32/__main__.py` for `python -m birdnet_stm32 {train,convert,evaluate,deploy}`
- [ ] Move `dev/make_dev_set.py` utilities into `birdnet_stm32/data/species.py`
- [ ] Keep top-level `train.py`, `test.py`, `convert.py` as thin wrappers that import from the package (backward compat)
- [ ] Add `py.typed` marker for type-checking

---

## 5. Configuration & Paths

### Current problems

- `deploy.sh` has hardcoded absolute paths: `/home/mi/Code/X-CUBE-AI.10.2.0`, `/home/mi/Code/iNat-tiny/...`
- `config.json` has hardcoded `cubeide_path: "/home/mi/stm32cubeide"`
- `config_n6l.json` has hardcoded paths to `network.c`, project, objcopy
- `notes.txt` has personal dataset paths
- No env var support, no `.env` file, no CLI overrides

### Tasks

- [ ] Create `birdnet_stm32/deploy/config.py` — config resolution: CLI args > env vars > `config.toml` > defaults
- [ ] Replace `config.json` and `config_n6l.json` with a single `config.toml` (or keep JSON but with template + resolution logic)
- [ ] Add `config.toml.example` with placeholder paths and comments
- [ ] `deploy.sh` → `birdnet_stm32/cli/deploy.py` (Python, no more hardcoded paths)
- [ ] Support `XCUBEAI_PATH`, `STEDGEAI_PATH`, `CUBEIDE_PATH`, `ARM_TOOLCHAIN_PATH` env vars
- [ ] Add `--stedgeai-path`, `--model`, `--output-dir`, `--config` CLI args to deploy command
- [ ] Move `notes.txt` to `dev/notes.md` (or delete — it's personal scratch)
- [ ] Add `config.toml` and `config_n6l.json` to `.gitignore`, ship only `.example` files

---

## 6. Audio Frontend Improvements

### Current state

- 3 modes: precomputed (librosa mel), hybrid (linear STFT + learned mel mixer), raw (waveform → Conv2D)
- Mag scaling: pwl (depthwise branches), pcen (pool/conv), db (log, bad for quantization)
- Hybrid mel mixer: 1×1 Conv2D seeded from librosa Slaney basis, optionally trainable breakpoints
- Raw: explicit symmetric pad → VALID Conv2D → BN → ReLU6

### Improvement tasks

- [ ] **Standardize frontend naming**: remove `tf`/`precomputed` aliases → just `librosa`, `hybrid`, `raw` (with deprecation warnings for old names)
- [ ] **Add MFCC frontend**: mel → DCT → truncate (common baseline, cheap, well-understood)
- [ ] **Add log-mel frontend**: native TF `tf.signal.stft` + `tf.signal.linear_to_mel_weight_matrix` path as a quantization-friendly alternative to librosa precompute (keeps everything in-graph for TFLite)
- [ ] **SpecAugment**: add frequency masking + time masking as a configurable augmentation in the frontend or data pipeline (improves robustness, standard practice)
- [ ] **Mixup improvements**: support Beta distribution mixup (currently uses uniform), label smoothing option
- [ ] **Frontend registry**: register frontends by name, auto-discover via `__init_subclass__`, validate N6 compatibility at registration
- [ ] **N6 compatibility checker**: static method on AudioFrontendLayer that checks `sample_rate * chunk_duration < 65536` and channel alignment before building
- [ ] **Decouple mag scaling from frontend**: make MagnitudeScaling a separate layer that can be composed, tested, and quantized independently
- [ ] **Fix `pick_first` logic in `pick_random_samples`**: when `pick_first=True` and `num_samples > 1`, always returns first sample regardless — clarify or fix semantics
- [ ] **Add frontend unit tests**: test each mode produces correct output shapes, test mag scaling numerics

---

## 7. Model Architecture Improvements

### Current state

- DS-CNN: 4 stages, 3×3 depthwise-separable, ReLU6, residual when stride=1 + channels match
- Scaling: alpha (width), depth_multiplier (block repeats)
- Head: GAP → Dropout(0.5) → Dense

### Improvement tasks

- [ ] **Add model registry**: `build_model(name, **kwargs)` dispatcher (dscnn, mobilenetv2_tiny, efficientnet_lite0, ...)
- [ ] **MobileNetV2-style inverted residuals**: expand → DW → project with expansion factor; better accuracy/param trade-off
- [ ] **Squeeze-and-excite (SE) blocks**: lightweight channel attention, compatible with NPU (just pool + dense + sigmoid + mul)
- [ ] **Multi-head attention pooling**: replace simple GAP with lightweight attention pooling over time dimension (optional, check N6 op support)
- [ ] **Knowledge distillation**: add option to train with soft labels from a larger BirdNet teacher model
- [ ] **Model profiling utility**: print per-layer MACs, params, activation memory; flag layers likely to fail N6 compilation
- [ ] **N6 op compatibility table**: maintain a list of tested TFLite ops on the N6 NPU (from stedgeai reports); warn at model build time if using unsupported ops
- [ ] **Configurable dropout**: currently hardcoded 0.5; add CLI arg
- [ ] **Configurable weight decay**: currently hardcoded 1e-4; add CLI arg
- [ ] **Label smoothing**: add as training option

---

## 8. Training Pipeline

### Improvement tasks

- [ ] **Replace raw dict config with dataclass**: `ModelConfig` with validation + serialization
- [ ] **Deterministic training mode**: add `--deterministic` flag that sets all seeds + TF deterministic ops
- [ ] **Resumable training**: add `--resume` flag that loads optimizer state from checkpoint
- [ ] **Learning rate finder**: add utility to sweep LR and plot loss (one-cycle policy style)
- [ ] **Optuna hyperparameter tuning**: add `--tune` flag that uses Optuna to search over LR, alpha, depth_multiplier, dropout, batch size; persist study in SQLite for resumable sweeps
- ~**WandB / TensorBoard integration**~: decided against — keep logging simple (CSV + stdout). Do not add wandb or tensorboard as dependencies.
- [ ] **Multi-GPU / mixed precision**: add `--mixed-precision` flag (fp16 compute, fp32 accum) for faster training
- [ ] **Class weighting**: add `--class-weights` option (inverse frequency, effective number, focal loss)
- [ ] **Focal loss**: implement as alternative to cross-entropy for imbalanced datasets
- [ ] **Data pipeline performance**: profile and optimize `data_generator` — pre-fetch audio in separate threads, cache spectrograms
- [ ] **Configurable optimizer**: add `--optimizer adam|sgd|adamw` CLI arg
- [ ] **Gradient clipping**: add `--grad-clip` CLI arg
- [ ] **Training metrics dashboard**: save training curves as PNG/HTML alongside checkpoint

---

## 9. Quantization & Conversion

### Current state

- PTQ with representative dataset calibration
- Float32 I/O, INT8 internals
- Validates cosine similarity / MSE / MAE / Pearson between Keras and TFLite

### Improvement tasks

- [ ] **Quantization-aware training (QAT)**: add `--qat` flag to train.py that inserts fake-quant nodes (TF Model Optimization Toolkit)
- [ ] **Per-channel vs per-tensor quantization**: add flag to control granularity
- [ ] **Dynamic range quantization**: add as alternative (no representative dataset needed)
- [ ] **Audit INT8 input assumptions**: audio waveform/spectrogram inputs are continuous-valued and lose meaningful precision at INT8; verify that float32 I/O is enforced throughout the pipeline and remove any code paths that attempt INT8 input quantization
- ~**INT8-only mode**~: rejected — INT8 I/O does not make sense for audio inputs. Keep float32 I/O + INT8 internals.
- [ ] **ONNX export**: add `--export-onnx` path (stedgeai also accepts ONNX)
- [ ] **Automatic cosine similarity validation**: fail conversion if cosine sim < threshold (configurable, default 0.95)
- [ ] **Representative dataset curation**: add SNR filtering + stratified sampling per class (current: random shuffle + center chunk)
- [ ] **Conversion report**: generate a structured JSON/HTML report with per-layer quantization ranges, before/after histograms
- [ ] **Batch validation**: validate across multiple random seeds and report worst-case metrics

---

## 10. Evaluation & Testing Pipeline

### Improvement tasks

- [ ] **Confusion matrix**: add per-class confusion matrix output (ASCII + optional matplotlib)
- [ ] **Species-level AP report**: CSV/JSON with AP per species + confidence intervals (bootstrap)
- [ ] **Detection Error Tradeoff (DET) curve**: standard bioacoustics metric
- [ ] **Threshold optimization**: find optimal threshold per class via PR curve (not just fixed 0.5)
- [ ] **Benchmark mode**: standardized eval on a fixed test split with multiple metrics, saved as structured JSON for experiment tracking
- [ ] **Latency measurement**: add `--benchmark-latency` that measures per-chunk inference time (TFLite)
- [ ] **Memory profiling**: report peak memory usage during inference
- [ ] **Cross-validation**: add k-fold CV option for more robust metrics on small datasets
- [ ] **HTML report generation**: replace ASCII viz with optional HTML report (plotly/matplotlib)

---

## 11. Setup & Deploy Scripts

### 11.1 Setup

- [ ] `scripts/setup.sh` — one-command setup: create venv, install deps, download sample data
- [ ] `scripts/setup_stm32.sh` — download + install X-CUBE-AI, ARM toolchain, STM32CubeProgrammer (with version pinning)
- [ ] `scripts/download_data.sh` — download iNatSounds subset, organize into `data/{train,test}/` structure
- [ ] `scripts/download_checkpoints.sh` — fetch pre-trained checkpoints from GitHub Releases into `checkpoints/`
- [ ] Docker: `Dockerfile` + `docker-compose.yml` for reproducible training environment (GPU support)

### 11.2 Deploy

Replace `deploy.sh` (hardcoded paths) with:

- [ ] `birdnet_stm32/cli/deploy.py` — Python deploy command with proper arg parsing
- [ ] Auto-detect `stedgeai` from PATH or `STEDGEAI_PATH` env var
- [ ] Auto-detect board connection (`/dev/ttyACM*`)
- [ ] Pre-flight checks: model exists, board connected, tools installed, config valid
- [ ] Colored terminal output with progress indicators
- [ ] `--dry-run` flag to show what would happen without executing
- [ ] `--skip-validate` to skip on-device validation step

---

## 12. Unit & Integration Tests

### 12.1 Test framework setup

- [ ] `tests/` directory with `conftest.py` (fixtures, tmp paths, sample audio generation)
- [ ] pytest + pytest-cov configuration in `pyproject.toml`
- [ ] Synthetic test fixtures: generate short WAV files programmatically (sine waves, noise, chirps)
- [ ] Small test dataset: `tests/fixtures/data/{train,test}/class_a/*.wav` (2-3 classes, 3-5 files each, ~1s)

### 12.2 Unit tests

```
tests/
├── conftest.py
├── unit/
│   ├── test_audio_io.py           # load_audio_file, fast_resample, save_wav
│   ├── test_spectrogram.py        # get_spectrogram_from_audio, normalize, mag_scale variants
│   ├── test_activity.py           # sort_by_activity, sort_by_s2n, get_activity_ratio
│   ├── test_augmentation.py       # mixup logic, label merging
│   ├── test_frontend_layer.py     # AudioFrontendLayer: each mode, each mag_scale, shapes, trainability
│   ├── test_dscnn.py              # build_dscnn_model: shapes, channel alignment, N6 constraints
│   ├── test_runners.py            # KerasRunner, TFLiteRunner: predict shapes
│   ├── test_pooling.py            # lme_pooling, pool_scores edge cases
│   ├── test_metrics.py            # evaluate() with known inputs/expected outputs
│   ├── test_dataset.py            # load_file_paths, get_classes, upsample
│   ├── test_config.py             # ModelConfig serialization round-trip
│   └── test_conversion.py         # quantization smoke test (tiny model → TFLite → validate)
├── integration/
│   ├── test_train_to_eval.py      # end-to-end: train 1 epoch → convert → eval
│   ├── test_frontend_parity.py    # librosa spectrogram ≈ hybrid frontend output (within tolerance)
│   └── test_quantization_sim.py   # TFLite cosine sim > threshold on known model
└── fixtures/
    ├── generate_fixtures.py       # script to create synthetic test data
    └── data/
        ├── train/
        │   ├── bird_a/
        │   └── bird_b/
        └── test/
            ├── bird_a/
            └── bird_b/
```

### 12.3 Test tasks

- [ ] `tests/conftest.py` — shared fixtures: tmp audio files, tiny model, sample config
- [ ] `tests/fixtures/generate_fixtures.py` — create synthetic WAVs (sine + noise, 1-3 seconds)
- [ ] `tests/unit/test_audio_io.py` — load, resample, chunk, edge cases (empty file, wrong SR, mono/stereo)
- [ ] `tests/unit/test_spectrogram.py` — shape validation, mag_scale correctness (pwl matches manual computation)
- [ ] `tests/unit/test_activity.py` — known SNR ordering, threshold filtering, always-keep-one guarantee
- [ ] `tests/unit/test_augmentation.py` — mixup shapes, label OR-merge, alpha bounds
- [ ] `tests/unit/test_frontend_layer.py` — precomputed/hybrid/raw output shapes for known inputs, mag scaling each type, trainability toggle, N6 constraint error
- [ ] `tests/unit/test_dscnn.py` — model builds for all frontend+mag combos, output shape = [B, num_classes], channel alignment assertion
- [ ] `tests/unit/test_runners.py` — KerasRunner + TFLiteRunner predict on dummy input, shape check
- [ ] `tests/unit/test_pooling.py` — avg/max/lme on known arrays, edge cases (empty, single chunk)
- [ ] `tests/unit/test_metrics.py` — known true/pred → expected ROC-AUC, F1, cmAP
- [ ] `tests/unit/test_dataset.py` — file discovery, class filtering, upsampling ratios
- [ ] `tests/unit/test_config.py` — ModelConfig save → load round-trip, missing field errors
- [ ] `tests/unit/test_conversion.py` — build tiny model → PTQ → TFLite file exists + runs
- [ ] `tests/integration/test_train_to_eval.py` — train 1 epoch on synthetic data → convert → evaluate → metrics dict valid
- [ ] `tests/integration/test_frontend_parity.py` — librosa mel ≈ hybrid Conv2D mel (cosine sim > 0.99)
- [ ] `tests/integration/test_quantization_sim.py` — quantized vs float cosine sim > 0.90 on tiny model

---

## 13. CI/CD

- [ ] `.github/workflows/test.yml` — run unit tests on push/PR (Python 3.11, 3.12; ubuntu-latest)
- [ ] `.github/workflows/lint.yml` — ruff check + ruff format --check
- [ ] `.github/workflows/docs.yml` — build mkdocs, deploy to GitHub Pages on release/main push
- [ ] `.github/workflows/integration.yml` — integration tests (on schedule or manual trigger, needs GPU or larger runner)
- [ ] `.github/workflows/release.yml` — semantic versioning, changelog generation, checkpoint upload to Releases
- [ ] Badge integration in README

---

## 14. Agent Docs & Dev Guides

### Files in `.github/`

- [ ] `.github/copilot-instructions.md` — update with new package structure, build/test commands, conventions
- [ ] `.github/instructions/models.instructions.md` — `applyTo: "birdnet_stm32/models/**"`, model architecture conventions, N6 constraints
- [ ] `.github/instructions/tests.instructions.md` — `applyTo: "tests/**"`, test conventions (fixtures, naming, assertions)
- [ ] `.github/instructions/audio.instructions.md` — `applyTo: "birdnet_stm32/audio/**"`, audio processing conventions, dtype rules

### Dev guides (in docs, linked from CONTRIBUTING.md)

- [ ] `docs/dev/implementation.md` — implementation notes: why DS-CNN, why PWL over PCEN, why float32 I/O, N6 NPU operator coverage
- [ ] `docs/dev/adding-a-frontend.md` — step-by-step guide to add a new audio frontend mode
- [ ] `docs/dev/adding-a-model.md` — step-by-step guide to add a new model architecture
- [ ] `docs/dev/experiment-tracking.md` — naming conventions for eval runs, how to compare results
- [ ] `docs/dev/release-process.md` — versioning, changelog, tagging, PyPI publish

---

## Execution Order (Suggested Phases)

### Phase 1 — Foundation (do first, unblocks everything)

1. Project scaffolding (§1): CODE_OF_CONDUCT, CONTRIBUTING, CITATION.cff, SECURITY, CHANGELOG, pyproject.toml, .gitignore, pre-commit
2. Package structure (§4): create `birdnet_stm32/` package, extract modules, keep backward-compat wrappers
3. Configuration (§5): replace hardcoded paths with config resolution, create `config.toml.example`
4. Test framework (§12.1-12.2): conftest, fixtures, initial unit tests for audio + spectrogram + frontend

### Phase 2 — Documentation

5. MkDocs scaffold (§2.1): mkdocs.yml, theme, nav
6. Move README deployment content → docs (§2.2, §3)
7. Write developer guide (§2.3)
8. API reference setup (§2.4)

### Phase 3 — Quality & Testing

9. Complete unit tests (§12.2)
10. Integration tests (§12.3)
11. CI/CD (§13)
12. Lint + type checking setup

### Phase 4 — Improvements

13. Audio frontend improvements (§6)
14. Model architecture improvements (§7)
15. Training pipeline improvements (§8)
16. Quantization improvements (§9)
17. Evaluation improvements (§10)

### Phase 5 — Deploy & Polish

18. Deploy scripts rewrite (§11)
19. Demo application (WAV → species inference CLI + optional board demo)
20. Agent docs (§14)
21. Final README revision (§3)
22. Move checkpoints to GitHub Release assets + download script
23. Release v1.0.0

---

## Decisions (Resolved)

- **PyPI**: No — repo-only install (`pip install .` or git URL). No release workflow needed.
- **Checkpoints**: Ship as GitHub Release assets, not in-repo. Add `scripts/download_checkpoints.sh` to fetch them.
- **Demo app**: Yes — include a basic demo (feed WAV → get species predictions).
- **Python**: 3.12+ only.
- **TensorFlow**: 2.x only (2.16+). No tf-nightly or 3.x.
- **Species lists**: Keep taxonomy in `dev/` as the species name + info reference. Do not move to `data/`.
