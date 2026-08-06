# Repository agent conventions

## Model release names

Public model families use this exact basename:

`BirdNET_Tiny_N6_<REGION>_<SPECIES_COUNT>_V<MAJOR.MINOR>`

- Use an uppercase region code. The northeastern United States code is `USNE`.
- `SPECIES_COUNT` counts bird species only. Nuisance and background outputs stay
  in the label/config contract but do not change this number.
- Version 1.0 of the 30-species USNE model is
  `BirdNET_Tiny_N6_USNE_30_V1.0`.
- Precision-bearing artifacts append exactly one uppercase precision token:
  `_FP32`, `_FP16`, or `_INT8`. For example, the v1 files include
  `BirdNET_Tiny_N6_USNE_30_V1.0_FP32.keras` and
  `BirdNET_Tiny_N6_USNE_30_V1.0_INT8.tflite`. The token describes stored model
  computation; the INT8 TFLite model still has float32 audio I/O.
- Shared sidecars append `_model_config.json` or `_labels.txt` to the family
  basename. Precision-specific validation data and reports append the precision
  before their descriptive suffix, such as `_INT8_validation_data.npz`.
- If quantization-aware training changes the deployable checkpoint, preserve
  the untouched pre-QAT checkpoint as `<basename>_original_FP32.keras` and use
  `<basename>_FP32.keras` for the checkpoint from which release exports were
  made.
- Repository releases and Python packages use full SemVer (`v1.0.0`); the model
  basename intentionally uses the shorter `V1.0` token.
- Stage binary assets only under the gitignored `release/<basename>/` directory.

Do not publish an artifact that merely converted successfully. A release model
must pass the configured Keras/TFLite parity gate, load in each advertised
runtime, pass ONNX validation when ONNX is included, and compile for STM32N6.
Record checksums and the validation reports in the staged release directory.
Exclude raw training histories, temporary conversions, and raw logs from public
bundles. Sanitize copied reports and manifests so they do not expose
machine-local paths, and include a concise model card for each public bundle.
