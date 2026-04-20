# Testing

## Running tests

```bash
# All tests
pytest

# Verbose with coverage
pytest -v --cov=birdnet_stm32

# Skip slow/integration tests
pytest -m "not slow and not integration"

# Single file
pytest tests/test_audio_io.py
```

## Test structure

```
tests/
├── conftest.py                  # Shared fixtures
├── fixtures/                    # Static test data
├── test_activity.py             # Activity detection logic
├── test_audio_io.py             # Audio loading, resampling, chunking
├── test_augmentation.py         # Audio augmentation pipeline
├── test_config.py               # ModelConfig serialization
├── test_conversion.py           # TFLite conversion pipeline
├── test_dataset.py              # File discovery, class handling
├── test_dscnn.py                # DS-CNN model building and scaling
├── test_eval_reports.py         # Evaluation reporting (species AP, DET, HTML, benchmark)
├── test_focal_loss.py           # Focal loss function
├── test_frontend_layer.py       # AudioFrontendLayer shapes and modes
├── test_frontend_parity.py      # Float/quantized frontend parity
├── test_frontend_registry.py    # Frontend name normalization and aliases
├── test_magnitude.py            # Magnitude scaling modes (pwl, pcen, db)
├── test_metrics.py              # ROC-AUC, cmAP, F1 computation
├── test_optimizer.py            # Optimizer configuration
├── test_pooling.py              # avg/max/lme pooling
├── test_qat.py                  # Quantization-aware training
├── test_quantization_sim.py     # Quantization simulation utilities
├── test_runners.py              # Evaluation runner pipeline
├── test_spec_augment.py         # SpecAugment / frequency masking
├── test_spectrogram.py          # Spectrogram computation, shapes
├── test_threshold_opt.py        # Threshold optimization
└── test_train_to_eval.py        # End-to-end train → evaluate integration
```

## Writing tests

### Conventions

- **File naming**: `test_<module>.py` mirroring the source module.
- **Function naming**: `test_<function>_<scenario>`, e.g.,
  `test_load_audio_file_mono`.
- **Fixtures**: use `conftest.py` fixtures for shared setup (tmp dirs, sample
  audio, configs).
- **Markers**: use `@pytest.mark.slow` for tests > 5 seconds,
  `@pytest.mark.integration` for end-to-end tests.

### Fixtures from conftest.py

| Fixture | Description |
|---|---|
| `sample_rate` | Default sample rate (22050 Hz) |
| `chunk_duration` | Default chunk duration (3 seconds) |
| `mel_bins` | Default number of mel bins (64) |
| `spec_width` | Default spectrogram width in frames (256) |
| `fft_length` | Default FFT length (512) |
| `num_classes` | Default number of classes (10) |
| `sine_wave` | 1 kHz sine wave (float32 array) |
| `silence` | All-zeros signal |
| `white_noise` | Random white noise (seeded) |
| `tmp_dataset` | Temp directory with class_a/class_b WAV files |

### TensorFlow-dependent tests

Some tests require TensorFlow, which may not be available in all environments.
Use `pytest.importorskip`:

```python
tf = pytest.importorskip("tensorflow")
```

This skips the test gracefully if TF is not installed rather than failing.

## Adding a new test

1. Create `tests/test_<module>.py`.
2. Import the module under test.
3. Use fixtures from `conftest.py` where applicable.
4. Add appropriate markers (`@pytest.mark.slow`, etc.).
5. Run: `pytest tests/test_<module>.py -v`
