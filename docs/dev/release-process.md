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

Public model families use:

```text
BirdNET_Tiny_N6_<REGION>_<SPECIES_COUNT>_V<MAJOR.MINOR>
```

So `BirdNET_Tiny_N6_USNE_60_V1.2` would identify a version 1.2 model for the
northeastern United States covering 60 bird species. `SPECIES_COUNT` does not
include nuisance or background outputs; those remain in the ordered labels and
model config, so the output count is generally higher than the name suggests.

Every precision-bearing filename appends `_FP32`, `_FP16`, or `_INT8` to that
family basename. The precision token describes stored model computation, not
the external tensor type: a full-INT8 TFLite model with float32 audio I/O is
still named `<basename>_INT8.tflite`. If QAT changes the release checkpoint,
also preserve the untouched pre-QAT checkpoint as
`<basename>_original_FP32.keras`.

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

Accuracy floors are derived per release, not carried over. Freeze the float
baseline's catalog metrics *before* any compression runs, and gate every
compressed candidate against that frozen reference. A floor set after seeing a
compressed result is not a gate.

Each compression step gets its own stated budget. Full-INT8 inference may lose
at most 0.01 ROC-AUC or 0.015 class-macro AP relative to the candidate's own
float checkpoint. Give QAT a separate budget: it is a larger and more
class-count-dependent cost than the INT8 step, and reusing a budget measured on
a model with far fewer outputs will fail a healthy model. The mean/p05 cosine
gates remain 0.95/0.90 regardless.

The converter promotes its temporary TFLite file only after mean and tail
parity gates pass. A failed report is diagnostic only. Audit calibration first;
use QAT if PTQ cannot pass reliably, then repeat every downstream check.

## 3. Stage model assets

Binary assets are not committed. Stage them under the gitignored directory:

```text
release/<model-basename>/
```

### Published bundle

The uploaded bundle carries the models and the contract needed to run them —
nothing else. It must contain exactly:

| File | Purpose |
|---|---|
| `<basename>_FP32.keras` | Deployable FP32 Keras checkpoint used for exports |
| `<basename>_original_FP32.keras` | Untouched FP32 training checkpoint when QAT was used |
| `<basename>_INT8.tflite` | Validated full-INT8 model with float32 I/O |
| `<basename>_FP32.onnx` | Validated FP32 ONNX export |
| `<basename>_model_config.json` | Input, frontend, architecture, and class contract |
| `<basename>_labels.txt` | Ordered output labels |
| `<basename>_INT8_stedgeai_report.txt` | INT8 STM32N6 compiler analysis |
| `<basename>_model_card.md` | Contract, provenance, intended target, and file guide |
| `LICENSE-MODELS.md` | Apache License 2.0 covering the model artifacts |
| `ACCEPTABLE_USE.md` | BirdNET acceptable use policy (guidance, not a license condition) |

Benchmark and validation reports, fixed validation inputs, `manifest.json`, and
checksum files are **not** published. Sanitize what does ship: public files must
not expose workstation paths.

### Validation record

Dropping those files from the upload does not relax any gate — every check still
runs, and `assemble_release.py` writes the full record to a sibling directory:

```text
release/<model-basename>_audit/
```

This holds `manifest.json` (identity, dataset and code revisions, gate results,
sizes, and hashes) plus the conversion, benchmark, and board reports. Keep it for
provenance; never upload it. Training histories, raw logs, and temporary exports
stay out of both directories.

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
