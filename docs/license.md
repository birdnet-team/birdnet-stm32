# License and Acceptable Use

BirdNET-STM32 is distributed under two licenses: one for the source code, one
for the trained models.

## Source code — MIT

Everything in the Python package, the CLI, and the project's own firmware
sources is licensed under the
[MIT License](https://github.com/birdnet-team/birdnet-stm32/blob/main/LICENSE.md).

## Model artifacts — Apache License 2.0

The trained model artifacts are licensed under the
[Apache License 2.0](https://github.com/birdnet-team/birdnet-stm32/blob/main/LICENSE-MODELS.md).
This covers everything a release bundle ships as a model:

- FP32 Keras checkpoints (deployable and pre-QAT)
- Converted `.tflite` and `.onnx` exports
- Ordered label files
- Model configuration JSON

Every release bundle carries a copy of `LICENSE-MODELS.md` and
`ACCEPTABLE_USE.md` alongside the models.

## Third-party firmware sources

The `firmware/` directory vendors source files from STMicroelectronics
(BSD-3-Clause) and ChaN's FatFs, which retain their original licenses. See
`firmware/THIRD_PARTY_LICENSES.md`. ST's toolchain (X-CUBE-AI / `stedgeai`) is
licensed separately by STMicroelectronics — see its own documentation.

## Acceptable use

The
[BirdNET Acceptable Use Policy](https://github.com/birdnet-team/birdnet-stm32/blob/main/ACCEPTABLE_USE.md)
describes how we expect the code and models to be used. In short: BirdNET exists
to support biodiversity research, conservation, education, and citizen science,
and should not be used to facilitate poaching, to locate sensitive species for
harmful purposes, as part of weapons or military targeting systems, or for
covert surveillance of people.

The policy is guidance from the project, not an additional condition on either
license. Forks and derivative applications should also pick a distinct name
rather than reusing BirdNET branding in a way that implies endorsement.

Report misuse or ethical concerns to `ccb-birdnet@cornell.edu`.
