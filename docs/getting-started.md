# Getting Started

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.12+ |
| TensorFlow | 2.16+ (with CUDA for GPU training) |
| OS | Ubuntu 22.04+ (other Linux distros should work) |

For STM32 deployment you also need:

- [X-CUBE-AI](https://www.st.com/en/embedded-software/x-cube-ai.html) 10.2.0+
- [STM32CubeProgrammer](https://www.st.com/en/development-tools/stm32cubeprog.html) 2.20+
- [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html) 1.19+
- ARM GNU toolchain 14.3+
- Physical STM32N6570-DK board connected via USB

## Installation

```bash
git clone https://github.com/birdnet-team/birdnet-stm32.git
cd birdnet-stm32
```

Create a virtual environment and install:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

This installs the `birdnet_stm32` package in editable mode with development
dependencies (pytest, ruff, pre-commit).

Optional extras:

```bash
pip install -e ".[dev,docs]"   # + documentation tools (mkdocs)
pip install -e ".[tune]"       # + Optuna for hyperparameter search
pip install -e ".[all]"        # everything (dev + docs + deploy + tune)
```

## Quick workflow

### 1. Prepare data

Organize audio files into the expected folder structure:

```
data/
├── train/<species_name>/*.wav
└── test/<species_name>/*.wav
```

See [Dataset Preparation](dataset.md) for details on downloading and structuring
the iNatSounds subset.

### 2. Train

```bash
python -m birdnet_stm32 train \
  --data_path_train data/train \
  --audio_frontend hybrid \
  --mag_scale pwl \
  --checkpoint_path checkpoints/my_model.keras
```

### 3. Convert

```bash
python -m birdnet_stm32 convert \
  --checkpoint_path checkpoints/my_model.keras \
  --model_config checkpoints/my_model_model_config.json \
  --data_path_train data/train
```

### 4. Evaluate

```bash
python -m birdnet_stm32 evaluate \
  --model_path checkpoints/my_model_quantized.tflite \
  --model_config checkpoints/my_model_model_config.json \
  --data_path_test data/test \
  --pooling lme
```

### 5. Deploy

See the [Deployment](deployment.md) guide for flashing the quantized model to
the STM32N6570-DK.

## Pre-trained model

Pre-trained models are GitHub release assets, not tracked repository files.
Version 1.0 uses the basename `BirdNET_Tiny_N6_USNE_30_V1.0` for a model with
30 northeastern-US bird species and eight nuisance outputs. The number in the
name counts bird species only. Precision-bearing files add `_FP32`, `_FP16`, or
`_INT8` before the extension; for example, deploy the
`BirdNET_Tiny_N6_USNE_30_V1.0_INT8.tflite` asset on STM32N6.

Download the Keras, TFLite, ONNX, ordered labels, and model config from the same
release and keep them together. The config and labels define the input/output
contract; do not mix sidecars from another model version. Maintainers stage
validated release bundles locally under the gitignored `release/` directory.
The v1.0 INT8 model uses 2.5-second mono 24 kHz input and produces 38 ordered
scores: 30 bird species and eight nuisance sounds.
