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

`LICENSE-MODELS.md` and `ACCEPTABLE_USE.md` ship alongside — see
[License & Acceptable Use](license.md).

!!! warning "The model, config, and labels are one contract"
    The TFLite model, `_model_config.json`, and `_labels.txt` describe a single
    trained model. Keep them together and never mix sidecars across bundles or
    versions — the frontend parameters and output ordering will not match.

The model card records the exact input contract (sample rate, chunk duration,
output count), the accuracy the model reached on its evaluation catalog, and its
measured on-board timing. Read it before deploying.

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
    BUNDLE=~/Downloads/BirdNET_Tiny_N6_USNE_30_V1.0

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
