# BirdNET-STM32

<p align="center">
  <img src="birdnet-logo.png" alt="BirdNET Live" width="250"><br>
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.12%2B-blue.svg" alt="Python 3.12+"></a>
  <a href="https://birdnet-team.github.io/birdnet-stm32"><img src="https://img.shields.io/badge/docs-mkdocs-blue.svg" alt="Docs"></a>
  <a href="https://github.com/birdnet-team/birdnet-stm32/releases/tag/v0.2.0"><img src="https://img.shields.io/badge/version-0.2.0-orange.svg" alt="Version"></a>
</p>

Bird sound classification for edge deployment on the [STM32N6570-DK](https://www.st.com/en/evaluation-tools/stm32n6570-dk.html) development board with neural processing unit (NPU).

<img src="https://my.avnet.com/wcm/connect/c651fc2f-a5b2-489c-9d63-d3f064753690/STMicroelectronics+STM32N6570-DK.jpg?MOD=AJPERES&CACHEID=ROOTWORKSPACE-c651fc2f-a5b2-489c-9d63-d3f064753690-phBdXih" alt="STM32N6570-DK board" style="width: 100%;" />

A compact DS-CNN trained on mel spectrograms, quantized to INT8 via post-training quantization, and deployed using ST's X-CUBE-AI toolchain. Inference on a 3-second audio chunk takes ~3.3 ms on the NPU (~900× real-time).

**NOTE: We are currently refining the project scope and roadmap. The current codebase is a work in progress and may not be fully functional. Please check back soon for updates!**

## Quick start

```bash
# Install
git clone https://github.com/birdnet-team/birdnet-stm32.git
cd birdnet-stm32
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Train
python train.py --data_path_train data/train --audio_frontend hybrid --mag_scale pwl

# Convert to quantized TFLite
python convert.py --checkpoint_path checkpoints/best_model.keras \
  --model_config checkpoints/best_model_model_config.json --data_path_train data/train

# Evaluate
python test.py --model_path checkpoints/best_model_quantized.tflite \
  --model_config checkpoints/best_model_model_config.json --data_path_test data/test

# Deploy to STM32N6570-DK (requires config.json; see config.example.json)
python -m birdnet_stm32 deploy
```

See the [full documentation](https://birdnet-team.github.io/birdnet-stm32) for detailed guides on [dataset preparation](https://birdnet-team.github.io/birdnet-stm32/dataset/), [training](https://birdnet-team.github.io/birdnet-stm32/training/), [conversion](https://birdnet-team.github.io/birdnet-stm32/conversion/), [evaluation](https://birdnet-team.github.io/birdnet-stm32/evaluation/), and [deployment](https://birdnet-team.github.io/birdnet-stm32/deployment/).

## Pre-trained model

| Model | Classes | Frontend | ROC-AUC | Inference (NPU) |
|---|---|---|---|---|
| `birdnet_stm32n6_100.tflite` | 100 (NE US + EU + Brazil) | hybrid, 257×256, PWL | 0.84 | 3.3 ms / chunk |

## License

- **Source code and models**: [MIT License](LICENSE.md)
- **STM tools and scripts**: see respective documentation for license details.

## Citation

```bibtex
@article{kahl2025birdnetstm32,
  title={A quantization-friendly audio classification pipeline for embedded bioacoustics on microcontroller NPUs},
  author={Kahl, Stefan and Marshall, Isabella and Chaopricha, Patrick T. and Aceto, Jordan and Klinck, Holger},
  year={2025}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. AI-assisted contributions are welcome — keep PRs focused and review every line.

## Terms of Use

See [TERMS_OF_USE.md](TERMS_OF_USE.md) for detailed terms and conditions.

## Funding

Our work in the Cornell K. Lisa Yang Center for Conservation Bioacoustics is made possible by the generosity of K. Lisa Yang to advance innovative conservation technologies to inspire and inform the conservation of wildlife and habitats.

The development of BirdNET is supported by the German Federal Ministry of Research, Technology and Space (FKZ 01|S22072), the German Federal Ministry for the Environment, Climate Action, Nature Conservation and Nuclear Safety (FKZ 67KI31040E), the German Federal Ministry of Economic Affairs and Energy (FKZ 16KN095550), the Deutsche Bundesstiftung Umwelt (project 39263/01) and the European Social Fund.

## Partners

BirdNET is a joint effort of partners from academia and industry.
Without these partnerships, this project would not have been possible.
Thank you!

![Logos of all partners](https://tuc.cloud/index.php/s/KSdWfX5CnSRpRgQ/download/box_logos.png)




