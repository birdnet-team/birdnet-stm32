# Release Process

How to prepare a BirdNET-STM32 software release and its model assets.

## Versioning

BirdNET-STM32 follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: breaking CLI, configuration, firmware, or model-contract changes;
- **MINOR**: backward-compatible features;
- **PATCH**: backward-compatible fixes and documentation updates.

Keep the version synchronized in:

1. `pyproject.toml` (`[project] version`);
2. `birdnet_stm32/__init__.py` (fallback `__version__`);
3. `CITATION.cff` (`version` and `date-released`);
4. the README badge and `CHANGELOG.md`.

Repository tags use full SemVer, such as `v1.0.0`.

## Model naming

Public model artifacts use:

```text
BirdNET_Tiny_N6_<REGION>_<SPECIES_COUNT>_V<MAJOR.MINOR>
```

For example, `BirdNET_Tiny_N6_USNE_30_V1.0` identifies the version 1.0 model
for the northeastern United States with 30 bird species. `SPECIES_COUNT` does
not include nuisance or background outputs; those remain in the ordered labels
and model config.

Use one exact basename for the deployable `.keras`, `.tflite`, and `.onnx`
files. If QAT changes the release checkpoint, also preserve the untouched
pre-QAT checkpoint as `<basename>_original.keras`.

## 1. Prepare the repository

Work from the intended release branch with a clean, up-to-date worktree, then
run the complete checks:

```bash
pytest -v --cov=birdnet_stm32
ruff check .
ruff format --check birdnet_stm32/ tests/
mkdocs build --strict
python -m build
```

Inspect the built wheel metadata and verify that the installed package reports
the intended version.

## 2. Validate the model

Do not equate file creation with successful conversion. The model release gate
requires all of the following:

1. Load the deployable Keras checkpoint and run a smoke inference.
2. Convert to full INT8 TFLite with float32 I/O and representative calibration.
3. Pass the configured Keras/TFLite parity threshold on held-out examples.
4. Evaluate float and INT8 models on the same frozen test set and report metric
   deltas. Threshold-free ROC-AUC and class-macro AP are the primary comparison;
   never optimize thresholds on the release test set.
5. Export ONNX, run the ONNX checker, and smoke-test it in an ONNX runtime.
6. Run `stedgeai analyze` or `stedgeai generate` for the STM32N6 target and
   retain its compatibility/memory report.
7. Run the custom firmware on the physical board and retain the board report.

If the converter writes a TFLite file and then fails its parity gate, that file
is invalid and must not be staged. Audit calibration first; use QAT if PTQ
cannot pass reliably, then repeat every downstream check.

## 3. Stage model assets

Binary assets are not committed. Stage them under the gitignored directory:

```text
release/<model-basename>/
```

The version 1.0 USNE bundle should contain:

| File | Purpose |
|---|---|
| `<basename>.keras` | Deployable Keras checkpoint used for exports |
| `<basename>_original.keras` | Untouched training checkpoint when QAT was used |
| `<basename>.tflite` | Validated full-INT8 model with float32 I/O |
| `<basename>.onnx` | Validated ONNX export |
| `<basename>_model_config.json` | Input, frontend, architecture, and class contract |
| `<basename>_labels.txt` | Ordered output labels |
| `<basename>_validation_data.npz` | Fixed on-device validation inputs |
| `<basename>_conversion.json` | Quantization and parity report |
| `<basename>_float_benchmark.json` | Float test metrics and latency |
| `<basename>_int8_benchmark.json` | INT8 test metrics and latency |
| `<basename>_stedgeai_report.txt` | STM32N6 compiler analysis |
| `<basename>_board_report.*` | Custom-firmware validation and timing |
| `manifest.json` | Model identity, sources, dataset/code revisions, gates, and sizes |
| `SHA256SUMS` | SHA-256 digest for every distributed file |

Only list an asset in the manifest after its corresponding validation passed.

## 4. Commit, tag, and publish

Keep the repository preparation and any quantization fix in focused commits.
After review, create an annotated tag and push it:

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin <release-branch> --tags
```

Create a GitHub release from the tag, copy the matching changelog section into
the release notes, and upload every file from the staged model directory.
Verify uploaded checksums before announcing the release.

The `docs.yml` workflow publishes MkDocs from the default branch. Confirm the
deployed documentation after the release merge.
