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
├── conftest.py              # Shared fixtures
├── test_audio_io.py         # Audio loading, resampling, chunking
├── test_spectrogram.py      # Spectrogram computation, shapes
├── test_pooling.py          # avg/max/lme pooling
├── test_dataset.py          # File discovery, class handling
├── test_config.py           # ModelConfig serialization
└── (more to come)
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
| `tmp_audio_dir` | Temporary directory with synthetic WAV files |
| `sample_wav` | Path to a single generated sine-wave WAV |
| `sample_config` | Dictionary with typical model config values |

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
