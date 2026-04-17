---
description: "Use when creating, writing, or editing unit or integration tests for the project."
applyTo: "tests/**/*.py"
---

# Test Conventions

When writing tests in the `tests/` directory, follow these conventions:

## Naming & Fixtures
- Test files must be prefixed with `test_` (e.g. `test_dataset.py`, `test_frontend_layer.py`).
- Test functions must be prefixed with `test_`.
- Prefer explicitly testing edge cases (batch size of 1, sample size mismatches, random seeding checks).
- Re-use synthetic fixtures defined in `conftest.py` (e.g., sine waves, random audios, tiny dummy models) rather than creating new ones.
- Avoid calling `os.chdir()`. If file manipulation is needed, use pytest's `tmp_path` fixture to keep test runs isolated and reproducible.

## Assertions
- Use standard `assert` rather than `unittest.TestCase`.
- For shape checks, strictly check both batch size and dimension. E.g., `assert outputs.shape == (batch_size, expected_dim)`.
- For numerical accuracy, use `np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)`.

## Integration vs Unit
- Fast unit tests check logic without compiling huge TF graphs or training loops.
- `integration` marker is for things that execute larger chunks spanning the pipeline (train epoch -> convert -> evaluate). Use `@pytest.mark.integration`.
- Integration tests can be slow. Mark them appropriately so CI runs them separately if needed.
- `pytest -m "not integration and not slow"` is used by GitHub CI for fast PR checks.