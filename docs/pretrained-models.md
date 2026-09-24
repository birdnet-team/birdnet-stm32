# Pre-trained Models

Trained and converted models are published as GitHub release assets, not tracked
repository files. Download a bundle from the
[latest release](https://github.com/birdnet-team/birdnet-stm32/releases/latest).

## What's in a bundle

Every file in a bundle shares one basename,
`BirdNET_Tiny_N6_<REGION>_<SPECIES>_V<VERSION>`, where the species count covers
bird species only — nuisance and background outputs are not counted in the name
but are part of the ordered output contract.

| File | Use it for |
|---|---|
| `<basename>_INT8.tflite` | Deploying to the STM32N6 — this is the model you flash |
| `<basename>_model_config.json` | Input, frontend, and class contract; drives the firmware config |
| `<basename>_labels.txt` | Ordered output labels |
| `<basename>_FP32.keras` | Host inference, fine-tuning, re-conversion |
| `<basename>_original_FP32.keras` | Pre-QAT checkpoint, for retraining from an untouched state |
| `<basename>_FP32.onnx` | Host and interchange inference |
| `<basename>_INT8_stedgeai_report.txt` | Memory footprint and NPU operator coverage |
| `<basename>_model_card.md` | Contract, provenance, measured accuracy, and on-board timing |

A bundle whose model was exported split carries these as well. The firmware runs
a single network, so `_INT8.tflite` above is still what you flash; these exist
so a species-list change can be pushed without resending the whole model.

| File | Use it for |
|---|---|
| `<basename>_INT8_backbone.tflite` | Audio to embeddings. Flashed once and reused across heads |
| `<basename>_INT8_classifier.tflite` | Embeddings to scores. The only part that changes with the species list |
| `<basename>_INT8_backbone.tflite.gz`, `<basename>_INT8_classifier.tflite.gz` | The same two, gzipped; the head is what an over-the-air update carries |
| `<basename>_INT8_backbone.tflite.fingerprint.json` | Identity of the backbone. A replacement head must be calibrated against a matching one |
| `<basename>_INT8_classifier_labels.txt` | The head's own ordered labels, so an updated head brings its species list with it |

Chaining the backbone into the classifier reproduces `_INT8.tflite`; both halves
are gated on that equivalence before either is published. To build a further
head for an already-flashed backbone, see
[Updating the head against a flashed backbone](conversion.md#updating-the-head-against-a-flashed-backbone).

`LICENSE-MODELS.md` and `ACCEPTABLE_USE.md` ship alongside — see
[License & Acceptable Use](license.md).

!!! warning "The model, config, and labels are one contract"
    The TFLite model, `_model_config.json`, and `_labels.txt` describe a single
    trained model. Keep them together and never mix sidecars across bundles or
    versions — the frontend parameters and output ordering will not match.

The model card records the exact input contract (sample rate, chunk duration,
output count), the accuracy the model reached on its evaluation catalog, and its
measured on-board timing. Read it before deploying.

## Benchmark results

Catalog accuracy is measured on our own dataset split, so nobody outside the
project can reproduce it. These models are therefore also scored on **WABAD**
([Zenodo 14191524](https://zenodo.org/records/14191524)), a public passive
acoustic monitoring benchmark: 3,794 annotated calls across five northeastern
sites (MABI, CB, SD, NL, FEU), of which 44 species fall inside the 90-species
output list. Any model can be scored on it, so these numbers are comparable with
published results elsewhere.

Every score below is from the **INT8 deployment model** — the file you flash —
scored on 3-second windows at a 1.25 s hop, pooled over all five sites.

| Model | Event recall @0.25 | @0.5 | Macro recall @0.5 | Window cMAP | Window AUPRC |
|---|---:|---:|---:|---:|---:|
| `..._90_V1.1_INT8` | 0.213 | 0.122 | 0.096 | 0.167 | 0.224 |
| `..._90_V1.2_INT8` | 0.225 | 0.133 | 0.107 | 0.168 | 0.240 |
| `..._90_V1.3_Raw_INT8` | 0.220 | 0.137 | 0.107 | 0.183 | 0.246 |
| `..._90_V1.3_Hybrid_INT8` | 0.205 | 0.112 | 0.074 | 0.208 | 0.276 |
| `..._90_V1.4_Raw_INT8` | 0.228 | 0.154 | 0.131 | 0.204 | 0.250 |
| `..._90_V1.4_Hybrid_INT8` | 0.284 | 0.185 | 0.160 | 0.285 | 0.335 |

- **Event recall** is the share of annotated calls whose overlapping window
  scored above the threshold — what a recorder set to that threshold would log.
- **Window cMAP** averages a precision-recall curve per species, so a bird heard
  twice weighs as much as one heard four hundred times. **Window AUPRC** pools
  every window-species decision into one curve, which is closer to what a
  deployment returns. They move independently, so both are reported: a model can
  cover the species list well and still miss most of the calls a site produces.

For scale, BirdNET+ V3.0 — a server-class model with no memory budget — reaches
0.656 event recall @0.5 and 0.517 window cMAP on the same annotations. These are
sub-megabyte models running on a microcontroller.

## Running a bundle on the board

Everything the firmware needs is in the bundle; no extra downloads.

1. Install the toolchain and create `config.json` — see
   [Deployment](deployment.md) for X-CUBE-AI, STM32CubeProgrammer, and ARM GNU
   setup.
2. Prepare an SD card with test audio — see
   [SD card preparation](deployment.md#sd-card-preparation). **WAV sample rate
   must match the `sample_rate` in `_model_config.json`**; mismatched files are
   skipped.
3. Compile, flash, and run:

    ```bash
    # Wherever the extracted bundle lives; the globs below pick the files out
    # of it, so nothing depends on which region or version you downloaded.
    BUNDLE=~/Downloads/<bundle>

    python -m birdnet_stm32 board-test \
      --model_path    "$BUNDLE"/*_INT8.tflite \
      --model_config  "$BUNDLE"/*_model_config.json \
      --labels        "$BUNDLE"/*_labels.txt
    ```

`board-test` generates the N6-optimized binary with `stedgeai`, flashes it over
serial, runs inference on every WAV on the SD card, and streams the top
predictions back over UART. Add `--save_results results.csv` to capture them.

## Validating on device

Release bundles do not ship fixed validation inputs. To run `stedgeai validate`
against a downloaded model, supply your own representative audio — see
[Deployment](deployment.md#step-6-validate-on-device) — or re-run
[conversion](conversion.md) locally to generate a validation set.
