# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] — 2026-08-05

### Added

- A fixed-validation training path with explicit output ordering and all-zero
  noise examples for leakage-safe multi-label experiments.
- Host-memory guards and bounded loader settings for long-running training on
  large audio collections.
- The public model naming convention
  `BirdNET_Tiny_N6_<REGION>_<SPECIES_COUNT>_V<MAJOR.MINOR>` and a gitignored
  release-staging workflow with validation reports and checksums.
- Native Keras 3 activation-aware QAT with differentiable per-channel kernel
  and per-tensor activation INT8 grids, including the model input and internal
  raw-frontend boundaries. QAT and final conversion share one exact
  deterministic calibration manifest; frozen-teacher Bernoulli KL and cosine
  consistency preserve calibrated outputs and lower-tail parity, and
  synchronized clean checkpoints contain no training-only quantization
  operators.

### Audio frontend rewrite

The raw frontend is rebuilt around a learned **Gabor quadrature filterbank**, and
both in-model frontends drop the per-sample normalization. Measured with
`stedgeai analyze` on a 100-output profiling model (24 kHz, 2.5 s, `raw`,
`pwl`): **software epochs 11 → 3**, and the filterbank convolutions moved off the
Cortex-M55 onto the NPU. MACC rises 93.2M → 108.4M, which buys a filterbank that
reads the whole signal instead of 5.7% of it.

- **The old raw filterbank skipped most of the audio.** It used a fixed 16-tap
  kernel against a hop of `ceil(T / spec_width)` — 282 samples at 24 kHz × 3 s —
  so each frame saw 0.67 ms of audio and **94% of every waveform was never read**.
  The window is now tied to the hop (`window = 2 × hop`), so every input sample
  reaches at least one frame. Covered by `test_responds_to_every_region_of_the_input`.
- **A single real-valued filter cannot measure magnitude.** Its output oscillates
  with the carrier, and sampling that at the frame rate aliases. The bank is now a
  cosine/sine quadrature pair combined as `max(|re|,|im|) + 0.4·min(|re|,|im|)` —
  within ~4% of the true modulus, against ~17% ripple for `|re| + |im|`, and it
  keeps INT8 range far better than `sqrt(re² + im²)`.
- **Filters are seeded from mel-spaced Gabor atoms.** Untrained, the bank already
  reproduces a librosa mel spectrogram (Pearson r ≈ 0.89 on real recordings);
  training refines it rather than starting from noise.
- **Per-band temporal lowpass after the modulus** (learnable 5-tap depthwise,
  Hann-initialized), the standard fix for envelope fluctuation in a Gabor
  filterbank. Raises per-band agreement with librosa mel from ~0.43 to ~0.69 for
  ~82k MACs.
- **The waveform is folded into `hop/2` interleaved channels** before the
  convolution. Free — NHWC memory is contiguous, so `[T, 1]` and `[T/fold, fold]`
  are the same bytes. This is what keeps the filterbank on the NPU: measured,
  **the N6 runs convolutions with stride > 2 in software**, so the natural
  large-hop "learned STFT" formulation executes entirely on the Cortex-M55.
  `raw_filterbank_geometry()` derives a layout with folded stride 2 and a fold
  that is a multiple of 8.
- **Per-sample max normalization removed** from the raw and hybrid paths. It cost
  a `ReduceMax` and a `Div` — both software epochs, each dragging a
  quantize/dequantize pair with it — and it was a data-dependent gain: one loud
  transient rescaled the whole spectrogram, so the same call at a different
  distance produced different features. Per-band calibration is now learned
  (BatchNorm plus an always-trainable magnitude-scaling layer).
- **`ReLU6` after the filterbank replaced with `ReLU`.** The upper clip emitted a
  `MINIMUM` against a constant, which the N6 ran in software.
- **Hybrid input drops the Nyquist bin**: `fft_bins` is now `fft_length // 2`
  (256 rather than 257), a multiple of 8. This removes the runtime `FILL` +
  `CONCATENATION` pair that padded the channel axis on every inference.
  `hybrid_fft_bins()` is the single source of truth; the spectrogram producer,
  data pipeline, evaluation and calibration paths all use it.

### Changed

- **Best-checkpoint selection now tracks validation ROC-AUC** instead of `val_loss`. With long-tail class priors, `val_loss` keeps improving after ranking quality has peaked, so the saved checkpoint was not the best detector. Early stopping follows the same metric.
- **Learning rate now warms up linearly for 2 epochs** before cosine decay (`WarmupCosineDecay` in `birdnet_stm32/training/trainer.py`).
- `--resume` continues along the original LR schedule. Previously the cosine schedule restarted from step 0 while `initial_epoch` jumped ahead, so a resumed run trained its remaining epochs at close to the peak learning rate.
- The classifier head is pinned to `dtype="float32"` so `--mixed_precision` does not run the sigmoid in float16.
- `train` prints the per-layer MAC/N6-compatibility profile instead of `model.summary()`. The profiler was previously documented but unreachable.
- `MagnitudeScalingLayer` accepts `pcen_pool_width` (default 3) controlling the width of each PCEN smoothing stage.
- Conversion now uses deterministic, exactly sized, disjoint calibration and
  validation samples; requires both mean and fifth-percentile cosine parity;
  records reproducible manifest identities; and atomically promotes TFLite
  artifacts only after all gates pass.
- ONNX export now uses the Keras 3 exporter and atomically promotes an artifact
  only after full ONNX checker validation and ONNX Runtime parity.

### Fixed

- All-zero noise examples now survive class upsampling and are included in
  evaluation without being mistaken for output classes.
- Evaluation reports the number of files actually evaluated rather than the
  number discovered before filtering.
- Training aborts cleanly under sustained host-memory pressure instead of
  allowing worker processes to exhaust the machine.
- **PCEN was a no-op.** Its smoothing stages used `pool_size=(1, 1)`, making the AGC branch an identity, which silently reduced `--mag_scale pcen` to a fixed affine rescale. Smoothing now runs over the time axis.
- **QAT was weight-only and skipped frontend kernels.** It now simulates
  the quantized model input, fused activation requantization, and nested
  raw-frontend kernels and elementwise boundaries as part of the clean Keras 3
  workflow.
- Validation `.npz` files no longer contain a duplicated batch dimension.
- **`--frontend_trainable` did nothing for the hybrid frontend.** The mel mixer was hard-wired to `trainable=False` and never saw `is_trainable`.
- `compute_det_curve()` ran one full pass per distinct score, which is intractable on multi-class evaluation sets (hundreds of thousands of distinct scores). It now uses a single sort plus cumulative counts, and no longer contains a dead indexing statement.
- `validate_models()` shares the `TFLiteRunner` code path instead of driving its own interpreter.

### Removed

- `birdnet_stm32/training/lr_finder.py` and `birdnet_stm32/training/distillation.py` — never imported or reachable from any CLI.
- `birdnet_stm32/models/registry.py` (frontend registry) — only its own test referenced it; `normalize_frontend_name()` is the mechanism actually in use.
- `build_model()` / `register_model()` / `list_models()` from `birdnet_stm32/models/__init__.py`. There is one architecture and the CLI calls `build_dscnn_model()` directly.
- Deprecated frontend aliases `precomputed` → `librosa` and `tf` → `raw`; the canonical names are now the only accepted values.
- Dead `train_mel_scale` learnable-mel-breakpoint branch in `AudioFrontendLayer` (permanently disabled by the constructor).
- Unused `sort_by_s2n()`, `get_s2n_from_spectrogram()`, `get_s2n_from_audio()` (the pipeline uses `sort_by_activity()`) and `save_wav()`.

## [0.4.0] — 2026-05-12

### Added

- **Memory-aware data loader**: per-frontend reservoir sizing via `_compute_reservoir_limits()`, configurable through the new `loader_buffer_mb` kwarg (default 128 MB). Replaces fixed reservoir constants that ignored sample size.
- **Bounded random-offset reads**: training pipeline now reads only the bytes it needs from each long file (`load_duration` capped by `candidate_chunks_per_file × chunk_duration`), instead of decoding `max_duration` and discarding most of the audio.
- `load_audio_window()` and `split_audio_into_chunks()` helpers in `birdnet_stm32/audio/io.py` for callers that want a single-pass read followed by their own chunk selection.
- `convert` CLI now writes a `{output_path_stem}_labels.txt` file alongside the converted TFLite model so downstream consumers can interpret the output tensor without the full Keras model config.

### Changed

- `representative_data_gen()` now bounds calibration reads via `cfg["max_duration"]` (or a sensible per-chunk multiple) instead of a hard-coded 30 s window.
- Mixup is now a single co-vocalization augmentation path: additive multi-source mixing with union labels, matching the project's overlapping-species training goal.
- **Always multi-label**: the classifier head is now hard-wired to sigmoid + binary crossentropy. Soundscape recordings are inherently multi-label even when the source label is single-class, so the softmax/categorical-crossentropy code path has been removed.
- `tf.keras.metrics.AUC` now passes `num_labels=num_classes` (read from the model output shape) for a more accurate multi-label ROC-AUC estimate.

### Removed

- `BinaryFocalLoss` and the `--loss` / `--focal_gamma` CLI knobs. Focal loss over-fits weak/noisy labels by emphasising "hard" examples, which is the opposite of what we want for crowdsourced bird soundscape data.
- `--label_smoothing` CLI knob. With ~100 sparse classes, smoothing pushes BCE toward the constant-prediction trivial minimum (AUC ≈ 0.5).
- `--no_class_weights` and the balanced inverse-frequency class weighting. Keras `class_weight=` is single-label only; with multi-hot targets it silently `argmax`es the label and scales the whole per-sample loss, which is incorrect.
- `class_activation` parameter from `build_dscnn_model()`; the head is always sigmoid.
- `is_multilabel`, `class_weights`, and `loss_fn=categorical_crossentropy` defaults from `train_model()`.

### Fixed

- Short audio files (< chunk size) now preserve the leading samples and pad once, instead of being silently dropped or zero-filled before the salient region.
- Removed duplicate `if __name__ == "__main__"` block in `birdnet_stm32/cli/board_test.py`.

### Removed

- Legacy backup scripts at the repo root: `convert.py.bak`, `test.py.bak`, `train.py.bak`.
- Unused `utils/audio.py` (superseded by `birdnet_stm32/audio/*.py`).

## [0.9.0] — 2026-04-20

### Added

- **Dynamic GPU memory growth**: training now calls `tf.config.experimental.set_memory_growth` so TensorFlow allocates GPU VRAM incrementally instead of grabbing the full device.
- **Smart crop** for weakly-labeled long recordings: short-time energy (STE) analysis finds salient audio regions, reducing label noise from silent chunks (`birdnet_stm32/audio/activity.py::smart_crop`).
- **Dirichlet multi-source mixup**: replaces Beta-distribution blending with Dirichlet sampling over 2–3 sources, realistically emulating overlapping bird vocalizations in soundscapes.
- **Linear probing** (`--linear_probe`): freeze a pretrained backbone and train only a new classification head on custom species data (`birdnet_stm32/training/linear_probe.py`).

### Changed

- **Consolidated CLI defaults**: SE blocks, inverted residuals, SpecAugment, deterministic training, balanced class weights, label smoothing (0.1), and gradient clipping (1.0) are now **on by default**. Use `--no_se`, `--no_inverted_residual`, `--no_spec_augment`, `--no_class_weights` to disable. `--max_duration` raised from 30 to 60 s.
- Removed `--deterministic` flag — training is always deterministic.

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
