# Release Process

How to prepare and publish a new release of BirdNET-STM32.

## Versioning

BirdNET-STM32 follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: breaking changes to CLI, config format, or model output format
- **MINOR**: new features (frontends, model options, evaluation metrics)
- **PATCH**: bug fixes, documentation updates, dependency bumps

The version is defined in two places (keep them in sync):

1. `pyproject.toml` → `[project] version`
2. `birdnet_stm32/__init__.py` → fallback `__version__`

## Release checklist

### 1. Prepare the release

```bash
# Ensure you're on a clean master branch
git checkout master && git pull

# Run full test suite
pytest -v --cov=birdnet_stm32

# Run linting
ruff check . && ruff format --check .

# Build docs
mkdocs build
```

### 2. Update version and changelog

1. Bump version in `pyproject.toml` and `birdnet_stm32/__init__.py`.
2. Update `CHANGELOG.md`:
   - Move items from `[Unreleased]` to `[X.Y.Z] - YYYY-MM-DD`.
   - Add a new empty `[Unreleased]` section.
3. Update the version badge in `README.md`.

### 3. Commit and tag

```bash
git add pyproject.toml birdnet_stm32/__init__.py CHANGELOG.md README.md
git commit -m "Release vX.Y.Z"
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin master --tags
```

### 4. Create GitHub Release

1. Go to the repository's Releases page.
2. Select the tag `vX.Y.Z`.
3. Title: `vX.Y.Z`
4. Body: copy the changelog section for this version.
5. Attach pre-trained checkpoint files (`.keras`, `.tflite`,
   `_model_config.json`, `_labels.txt`) as release assets.

### 5. Deploy documentation

The `docs.yml` GitHub Actions workflow automatically builds and deploys
MkDocs to GitHub Pages on push to `master`. Verify the deployed docs at
<https://birdnet-team.github.io/birdnet-stm32>.

## Pre-trained checkpoints

Checkpoints are **not** stored in the repository (they're in `.gitignore`).
Instead, attach them to GitHub Releases as binary assets. Users can download
them manually or via a future `scripts/download_checkpoints.sh` script.

### Recommended checkpoint bundle

For each release, include:

| File | Description |
|---|---|
| `{name}.keras` | Trained Keras model |
| `{name}_quantized.tflite` | INT8 quantized TFLite model |
| `{name}_model_config.json` | Model configuration |
| `{name}_labels.txt` | Class label list |
| `{name}_quantized_validation_data.npz` | On-device validation data |
