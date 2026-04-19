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

- [x] `CODE_OF_CONDUCT.md` — Contributor Covenant v2.1
- [x] `CONTRIBUTING.md` — contribution guide (setup, branching, PR process, code style, testing expectations)
- [x] `CITATION.cff` — CFF for academic citation (authors, title, DOI placeholder, repo URL)
- [x] `SECURITY.md` — vulnerability reporting policy
- [x] `CHANGELOG.md` — keep-a-changelog format, start with `[Unreleased]`
- [x] `.gitignore` — comprehensive Python/TF/IDE/OS ignores
- [x] `pyproject.toml` — replace bare `requirements.txt` with proper project metadata, `[project.optional-dependencies]` for dev/docs/deploy groups
- [x] `Makefile` — common tasks: `make train`, `make convert`, `make test`, `make deploy`, `make docs`, `make lint`, `make test-unit`, `make test-integration`

### 1.2 Licensing cleanup

- [x] Consolidate `LICENSE.md`, `TERMS_OF_USE.md`, `TERMS_OF_USE.txt` into a single `LICENSE` (or keep LICENSE + TERMS_OF_USE, but not both .md and .txt)
- [x] Add SPDX license identifier to `pyproject.toml`

### 1.3 Pre-commit hooks

- [x] `.pre-commit-config.yaml` with: ruff (lint+format), check-yaml, trailing-whitespace, end-of-file-fixer
- [x] Add ruff config section to `pyproject.toml`

---

## 2. MkDocs Material Documentation

### 2.1 Scaffold

- [x] `mkdocs.yml` — Material theme, nav structure, plugins (search, mkdocstrings, gen-files, literate-nav)
- [x] `docs/index.md` — landing page (project overview, badges, quick links)
- [ ] `docs/assets/` — images, diagrams

### 2.2 User Guide

- [x] `docs/getting-started.md` — prerequisites, install, quick train/eval workflow
- [x] `docs/dataset.md` — dataset format, folder structure, noise classes, species lists, downloading iNatSounds
- [x] `docs/training.md` — full training guide with all CLI args, frontend selection, tips
- [x] `docs/conversion.md` — Keras → TFLite PTQ, representative dataset, similarity validation
- [x] `docs/evaluation.md` — test.py usage, metrics explained (ROC-AUC, cmAP, F1, LME pooling)
- [x] `docs/deployment.md` — STM32N6570-DK setup, X-CUBE-AI install, stedgeai, n6_loader, on-device validation

### 2.3 Developer Guide

- [x] `docs/dev/architecture.md` — pipeline diagram, component boundaries, data flow
- [x] `docs/dev/audio-frontends.md` — deep-dive into precomputed/hybrid/raw modes, mag scaling, trainability, N6 constraints
- [x] `docs/dev/model.md` — DS-CNN architecture, alpha/depth multiplier, channel alignment, residual blocks
- [x] `docs/dev/quantization.md` — PTQ details, float32 I/O, representative dataset strategy, cosine similarity targets, NPU op coverage
- [x] `docs/dev/testing.md` — how to run tests, add tests, test fixtures, CI integration
- [x] `docs/dev/contributing.md` — link to CONTRIBUTING.md + dev-specific workflow

### 2.4 API Reference

- [x] `docs/gen_ref_pages.py` — auto-generated from docstrings via mkdocstrings + gen-files + literate-nav

### 2.5 Build & deploy docs

- [x] GitHub Actions workflow to build + deploy to GitHub Pages on push to `master`
- [x] Docs extras in `pyproject.toml` (mkdocs-material, mkdocstrings, gen-files, literate-nav)

---

## 3. README Revision

The current README is ~420 lines and mixes quick-start with exhaustive deployment instructions. Rewrite to be concise:

- [x] Badges: license, Python 3.12+, docs link
- [x] One-paragraph description + hero image
- [x] Quick Start (5-step: clone, install, train, convert, evaluate) — link to docs for details
- [x] Model Zoo table (pre-trained checkpoint with metrics)
- [x] Deployment teaser with link to `docs/deployment.md`
- [x] Citation block
- [x] Contributing + License links
- [x] Remove multi-page deployment instructions (moved to docs)

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

- [x] Create `birdnet_stm32/` package with `__init__.py`
- [x] Extract `AudioFrontendLayer` from `train.py` → `birdnet_stm32/models/frontend.py`
- [x] Extract `build_dscnn_model`, `ds_conv_block`, `_make_divisible` → `birdnet_stm32/models/dscnn.py`
- [x] Extract `KerasRunner`, `TFLiteRunner` → `birdnet_stm32/models/runners.py`
- [x] Extract `data_generator`, `load_dataset`, `load_file_paths_from_directory`, `upsample_minority_classes`, `get_classes_with_most_samples` → `birdnet_stm32/data/dataset.py`
- [x] Split `utils/audio.py` into `birdnet_stm32/audio/{io,spectrogram,activity}.py`
- [x] Extract `mixup` logic into `birdnet_stm32/audio/augmentation.py`
- [x] Extract `lme_pooling`, `pool_scores` → `birdnet_stm32/evaluation/pooling.py`
- [x] Extract metrics → `birdnet_stm32/evaluation/metrics.py`
- [x] Extract ASCII viz + CSV → `birdnet_stm32/evaluation/reporting.py`
- [x] Extract conversion logic → `birdnet_stm32/conversion/{quantize,validate}.py`
- [x] Create `birdnet_stm32/training/trainer.py` with training loop
- [x] Create CLI entry points in `birdnet_stm32/cli/` wrapping each script
- [x] Create `birdnet_stm32/__main__.py` for `python -m birdnet_stm32 {train,convert,evaluate,deploy}`
- [x] Move `dev/make_dev_set.py` utilities into `birdnet_stm32/data/species.py`
- [x] Keep top-level `train.py`, `test.py`, `convert.py` as thin wrappers that import from the package (backward compat)
- [x] Add `py.typed` marker for type-checking

---

## 5. Configuration & Paths

### Current problems

- `deploy.sh` has hardcoded absolute paths: `/home/mi/Code/X-CUBE-AI.10.2.0`, `/home/mi/Code/iNat-tiny/...`
- `config.json` has hardcoded `cubeide_path: "/home/mi/stm32cubeide"`
- `config_n6l.json` has hardcoded paths to `network.c`, project, objcopy
- `notes.txt` has personal dataset paths
- No env var support, no `.env` file, no CLI overrides

### Tasks

- [x] Create `birdnet_stm32/deploy/config.py` — config resolution: CLI args > env vars > `config.toml` > defaults
- [ ] Replace `config.json` and `config_n6l.json` with a single `config.toml` (or keep JSON but with template + resolution logic)
- [x] Add `config.toml.example` with placeholder paths and comments
- [x] `deploy.sh` → `birdnet_stm32/cli/deploy.py` (Python, no more hardcoded paths)
- [x] Support `XCUBEAI_PATH`, `STEDGEAI_PATH`, `CUBEIDE_PATH`, `ARM_TOOLCHAIN_PATH` env vars
- [x] Add `--stedgeai-path`, `--model`, `--output-dir`, `--config` CLI args to deploy command
- [x] Move `notes.txt` to `dev/notes.md` (or delete — it's personal scratch)
- [ ] Add `config.toml` and `config_n6l.json` to `.gitignore`, ship only `.example` files

---

## 6. Audio Frontend Improvements

### Current state

- 3 modes: precomputed (librosa mel), hybrid (linear STFT + learned mel mixer), raw (waveform → Conv2D)
- Mag scaling: pwl (depthwise branches), pcen (pool/conv), db (log, bad for quantization)
- Hybrid mel mixer: 1×1 Conv2D seeded from librosa Slaney basis, optionally trainable breakpoints
- Raw: explicit symmetric pad → VALID Conv2D → BN → ReLU6

### Improvement tasks

- [x] **Standardize frontend naming**: remove `tf`/`precomputed` aliases → just `librosa`, `hybrid`, `raw` (with deprecation warnings for old names)
- [x] **Add MFCC frontend**: mel → DCT → truncate (common baseline, cheap, well-understood)
- [x] **Add log-mel frontend**: native TF `tf.signal.stft` + `tf.signal.linear_to_mel_weight_matrix` path as a quantization-friendly alternative to librosa precompute (keeps everything in-graph for TFLite)
- [x] **SpecAugment**: add frequency masking + time masking as a configurable augmentation in the frontend or data pipeline (improves robustness, standard practice)
- [x] **Mixup improvements**: support Beta distribution mixup (currently uses uniform), label smoothing option
- [x] **Frontend registry**: register frontends by name, auto-discover via `__init_subclass__`, validate N6 compatibility at registration
- [x] **N6 compatibility checker**: static method on AudioFrontendLayer that checks `sample_rate * chunk_duration < 65536` and channel alignment before building
- [x] **Decouple mag scaling from frontend**: make MagnitudeScaling a separate layer that can be composed, tested, and quantized independently
- [x] **Fix `pick_first` logic in `pick_random_samples`**: when `pick_first=True` and `num_samples > 1`, always returns first sample regardless — clarify or fix semantics
- [x] **Add frontend unit tests**: test each mode produces correct output shapes, test mag scaling numerics

---

## 7. Model Architecture Improvements

### Current state

- DS-CNN: 4 stages, 3×3 depthwise-separable, ReLU6, residual when stride=1 + channels match
- Scaling: alpha (width), depth_multiplier (block repeats)
- Head: GAP → Dropout(0.5) → Dense

### Improvement tasks

- [x] **Add model registry**: `build_model(name, **kwargs)` dispatcher (dscnn, mobilenetv2_tiny, efficientnet_lite0, ...)
- [x] **MobileNetV2-style inverted residuals**: expand → DW → project with expansion factor; better accuracy/param trade-off
- [x] **Squeeze-and-excite (SE) blocks**: lightweight channel attention, compatible with NPU (just pool + dense + sigmoid + mul)
- [x] **Multi-head attention pooling**: replace simple GAP with lightweight attention pooling over time dimension (optional, check N6 op support)
- [x] **Knowledge distillation**: add option to train with soft labels from a larger BirdNet teacher model
- [x] **Model profiling utility**: print per-layer MACs, params, activation memory; flag layers likely to fail N6 compilation
- [x] **N6 op compatibility table**: maintain a list of tested TFLite ops on the N6 NPU (from stedgeai reports); warn at model build time if using unsupported ops
- [x] **Configurable dropout**: currently hardcoded 0.5; add CLI arg
- [x] **Configurable weight decay**: currently hardcoded 1e-4; add CLI arg
- [x] **Label smoothing**: add as training option

---

## 8. Training Pipeline

### Improvement tasks

Use dev dataset (or a 25 species / 500 files subset) at /home/mi/Datasets/stm32_1k for fast iteration. Build as a train and test set in data/train and data/test. use common european and north american species.

- [x] **Replace raw dict config with dataclass**: `ModelConfig` with validation + serialization
- [x] **Deterministic training mode**: add `--deterministic` flag that sets all seeds + TF deterministic ops
- [x] **Resumable training**: add `--resume` flag that loads optimizer state from checkpoint
- [x] **Learning rate finder**: add utility to sweep LR and plot loss (one-cycle policy style)
- [x] **Optuna hyperparameter tuning**: add `--tune` flag that uses Optuna to search over hyperparameters- ~**WandB / TensorBoard integration**~: decided against — keep logging simple (CSV + stdout). Do not add wandb or tensorboard as dependencies.
- [x] **Multi-GPU / mixed precision**: add `--mixed-precision` flag (fp16 compute, fp32 accum) for faster training
- [x] **Class weighting**: add `--class-weights` option (inverse frequency, effective number, focal loss)
- [x] **Focal loss**: implement as alternative to cross-entropy for imbalanced datasets
- [x] **Data pipeline performance**: profile and optimize `data_generator` — pre-fetch audio in separate threads, cache spectrograms
- [x] **Configurable optimizer**: add `--optimizer adam|sgd|adamw` CLI arg
- [x] **Gradient clipping**: add `--grad-clip` CLI arg
- [x] **Training metrics dashboard**: save training curves as PNG/HTML alongside checkpoint

---

## 9. Quantization & Conversion

### Current state

- PTQ with representative dataset calibration
- Float32 I/O, INT8 internals
- Validates cosine similarity / MSE / MAE / Pearson between Keras and TFLite

### Improvement tasks

- [x] **Quantization-aware training (QAT)**: `--qat` flag for shadow-weight fake-quantization fine-tuning (Keras 3 compatible, no tfmot dependency). Freezes BN, injects INT8 noise into kernels, maintains FP32 shadow weights. No FakeQuant ops in saved model — N6 compatible. +1.5pp cmAP improvement over PTQ-only.
- [x] **Per-channel vs per-tensor quantization**: add flag to control granularity
- [x] **Dynamic range quantization**: add as alternative (no representative dataset needed)
- [x] **Audit INT8 input assumptions**: audio waveform/spectrogram inputs are continuous-valued and lose meaningful precision at INT8; verify that float32 I/O is enforced throughout the pipeline and remove any code paths that attempt INT8 input quantization
- ~**INT8-only mode**~: rejected — INT8 I/O does not make sense for audio inputs. Keep float32 I/O + INT8 internals.
- [x] **ONNX export**: add `--export-onnx` path (stedgeai also accepts ONNX)
- [x] **Automatic cosine similarity validation**: fail conversion if cosine sim < threshold (configurable, default 0.95)
- [x] **Representative dataset curation**: add SNR filtering + stratified sampling per class (current: random shuffle + center chunk)
- [x] **Conversion report**: generate a structured JSON/HTML report with per-layer quantization ranges, before/after histograms
- [x] **Batch validation**: validate across multiple random seeds and report worst-case metrics

---

## 10. Evaluation & Testing Pipeline

### Improvement tasks

- [x] **Confusion matrix**: add per-class confusion matrix output (ASCII + optional matplotlib)
- [x] **Species-level AP report**: CSV/JSON with AP per species + confidence intervals (bootstrap). `--species_report`, `--n_bootstrap`.
- [x] **Detection Error Tradeoff (DET) curve**: standard bioacoustics metric. `--det_curve` (ASCII) + `--save_det_plot` (matplotlib).
- [x] **Threshold optimization**: find optimal threshold per class via PR curve (not just fixed 0.5)
- [x] **Benchmark mode**: standardized eval on a fixed test split with multiple metrics, saved as structured JSON for experiment tracking. `--benchmark`.
- [x] **Latency measurement**: add `--benchmark_latency` that measures per-chunk inference time (TFLite). Per-sample mean/median/p95/p99 stats.
- [ ] **Memory profiling**: report peak memory usage during inference
- [ ] **Cross-validation**: add k-fold CV option for more robust metrics on small datasets
- [x] **HTML report generation**: self-contained HTML report with inline CSS, metrics table, per-species AP table, and confusion matrix heatmap (base64 matplotlib). `--report_html`.

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

- [x] `tests/` directory with `conftest.py` (fixtures, tmp paths, sample audio generation)
- [x] pytest + pytest-cov configuration in `pyproject.toml`
- [x] Synthetic test fixtures: generate short WAV files programmatically (sine waves, noise, chirps)
- [x] Small test dataset: `tests/fixtures/data/{train,test}/class_a/*.wav` (2-3 classes, 3-5 files each, ~1s)

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

- [x] `tests/conftest.py` — shared fixtures: tmp audio files, tiny model, sample config
- [x] `tests/fixtures/generate_fixtures.py` — create synthetic WAVs (sine + noise, 1-3 seconds)
- [x] `tests/unit/test_audio_io.py` — load, resample, chunk, edge cases
- [x] `tests/unit/test_spectrogram.py` — shape validation, mag_scale correctness
- [x] `tests/unit/test_activity.py` — known SNR ordering, threshold filtering, always-keep-one guarantee
- [x] `tests/unit/test_augmentation.py` — mixup shapes, label OR-merge, alpha bounds
- [x] `tests/unit/test_frontend_layer.py` — precomputed/hybrid/raw output shapes for known inputs, mag scaling each type, trainability toggle, N6 constraint error
- [x] `tests/unit/test_dscnn.py` — model builds for all frontend+mag combos, output shape = [B, num_classes], channel alignment assertion
- [x] `tests/unit/test_runners.py` — KerasRunner + TFLiteRunner predict on dummy input, shape check
- [x] `tests/unit/test_pooling.py` — avg/max/lme on known arrays, edge cases
- [x] `tests/unit/test_metrics.py` — known true/pred → expected ROC-AUC, F1, cmAP
- [x] `tests/unit/test_dataset.py` — file discovery, class filtering, upsampling ratios
- [x] `tests/unit/test_config.py` — ModelConfig save → load round-trip, missing field errors
- [x] `tests/unit/test_conversion.py` — build tiny model → PTQ → TFLite file exists + runs
- [x] `tests/integration/test_train_to_eval.py` — train 1 epoch on synthetic data → convert → evaluate → metrics dict valid
- [x] `tests/integration/test_frontend_parity.py` — librosa mel ≈ hybrid Conv2D mel (cosine sim > 0.99)
- [x] `tests/integration/test_quantization_sim.py` — quantized vs float cosine sim > 0.90 on tiny model

---

## 13. CI/CD

- [x] `.github/workflows/test.yml` — run unit tests on push/PR (Python 3.12; ubuntu-latest)
- [x] `.github/workflows/lint.yml` — ruff check + ruff format --check
- [x] `.github/workflows/docs.yml` — build mkdocs, deploy to GitHub Pages on push to master
- [ ] `.github/workflows/integration.yml` — integration tests (on schedule or manual trigger, needs GPU or larger runner)
- [ ] `.github/workflows/release.yml` — semantic versioning, changelog generation, checkpoint upload to Releases
- [ ] Badge integration in README

---

## 14. Agent Docs & Dev Guides

### Files in `.github/`

- [x] `.github/copilot-instructions.md` — project guidelines, build/test commands, conventions, N6 pitfalls
- [x] `.github/instructions/models.instructions.md` — `applyTo: "birdnet_stm32/models/**"`, model architecture conventions, N6 constraints
- [x] `.github/instructions/tests.instructions.md` — `applyTo: "tests/**"`, test conventions (fixtures, naming, assertions)
- [x] `.github/instructions/audio.instructions.md` — `applyTo: "birdnet_stm32/audio/**"`, audio processing conventions, dtype rules

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
