# BirdNET-STM32

<p align="center">
  <img src="birdnet-logo.png" alt="BirdNET Live" width="250"><br>
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.12%2B-blue.svg" alt="Python 3.12+"></a>
  <a href="https://birdnet-team.github.io/birdnet-stm32"><img src="https://img.shields.io/badge/docs-mkdocs-blue.svg" alt="Docs"></a>
  <a href="https://github.com/birdnet-team/birdnet-stm32/releases/tag/v0.9.0"><img src="https://img.shields.io/badge/version-0.9.0-orange.svg" alt="Version"></a>
</p>

Bird sound classification for edge deployment on the [STM32N6570-DK](https://www.st.com/en/evaluation-tools/stm32n6570-dk.html) development board with neural processing unit (NPU).

<img src="https://my.avnet.com/wcm/connect/c651fc2f-a5b2-489c-9d63-d3f064753690/STMicroelectronics+STM32N6570-DK.jpg?MOD=AJPERES&CACHEID=ROOTWORKSPACE-c651fc2f-a5b2-489c-9d63-d3f064753690-phBdXih" alt="STM32N6570-DK board" style="width: 100%;" />

A compact DS-CNN trained on audio waveforms or mel spectrograms, quantized to INT8 via post-training quantization, and deployed using ST's X-CUBE-AI toolchain. Depending on the chosen audio frontend, a 2-3 second audio chunk takes approximately **10-14 ms** to infer directly on the NPU (0ms STFT overhead for the **raw** audio frontend, eliminating CPU cycles).

## Quick start

```bash
# Install
git clone https://github.com/birdnet-team/birdnet-stm32.git
cd birdnet-stm32
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Train
python -m birdnet_stm32 train \
  --data_path_train data/train \
  --audio_frontend hybrid --mag_scale pwl

# Convert to quantized TFLite
python -m birdnet_stm32 convert \
  --checkpoint_path checkpoints/best_model.keras \
  --model_config checkpoints/best_model_model_config.json \
  --data_path_train data/train

# Evaluate
python -m birdnet_stm32 evaluate \
  --model_path checkpoints/best_model_quantized.tflite \
  --model_config checkpoints/best_model_model_config.json \
  --data_path_test data/test --pooling lme

# Deploy to STM32N6570-DK (requires config.json; see config.example.json)
python -m birdnet_stm32 deploy

# On-board integration test (requires SD card with test audio)
python -m birdnet_stm32 board-test
```

### SD card preparation for board-test

The `board-test` command runs inference entirely on the STM32N6570-DK: it reads WAV
files from the SD card, computes the STFT on the Cortex-M55, and runs the model on the
NPU. **WAV files on the SD card must match the model's sample rate** (printed in the
`_model_config.json` file, e.g. 24000 Hz). Files with a mismatched sample rate are
skipped as errors.

Prepare the SD card as follows:

1. Format as FAT32.
2. Create an `audio/` directory at the root.
3. Copy `.wav` files (mono or stereo, 16-bit PCM) into `audio/`.
   Each file should be at least as long as the model's chunk duration (default 3 s).
4. Insert the SD card into the STM32N6570-DK board slot.

See the [full documentation](https://birdnet-team.github.io/birdnet-stm32) for detailed guides on [dataset preparation](https://birdnet-team.github.io/birdnet-stm32/dataset/), [training](https://birdnet-team.github.io/birdnet-stm32/training/), [conversion](https://birdnet-team.github.io/birdnet-stm32/conversion/), [evaluation](https://birdnet-team.github.io/birdnet-stm32/evaluation/), and [deployment](https://birdnet-team.github.io/birdnet-stm32/deployment/).

## Features

### Training

- **Audio frontends**: `hybrid` (STFT + learned mel mixer), `raw` (waveform → learned filterbank), `librosa` (precomputed mel), `mfcc`, `log_mel`
- **Magnitude scaling**: `pwl` (piecewise-linear, quantization-friendly), `pcen`, `db`, `none`
- **Model**: DS-CNN with configurable width (`--alpha`) and depth (`--depth_multiplier`), SE attention and inverted residuals (on by default; disable with `--no_se`, `--no_inverted_residual`), and optional attention pooling (`--use_attention_pooling`)
- **Augmentation**: Dirichlet multi-source mixup, SpecAugment (on by default), smart crop for long recordings, label smoothing
- **Optimization**: cosine LR decay, Adam/SGD/AdamW, gradient clipping (on by default), mixed precision (FP16), balanced class weights (on by default)
- **QAT**: quantization-aware fine-tuning via `--qat` — shadow-weight fake-quantization, no FakeQuant ops in saved model
- **Linear probing**: `--linear_probe` freezes a pretrained backbone and trains only the classifier head
- **Hyperparameter tuning**: Optuna search via `--tune --n_trials N`

### Conversion

- **Post-training quantization**: INT8 internals, float32 I/O, per-channel (default) or per-tensor
- **Dynamic range quantization**: `--quantization dynamic` — no calibration data needed
- **Validation**: cosine similarity, MSE, Pearson r between Keras and TFLite outputs
- **Batch validation**: `--batch_validate N` for worst-case metrics across seeds
- **ONNX export**: `--export_onnx` (requires `tf2onnx`)

### Evaluation

- **Pooling**: avg, max, LME (log-mean-exponential)
- **Metrics**: ROC-AUC, cmAP, mAP, precision, recall, F1
- **Species AP report**: per-species AP with bootstrap 95% CI (`--species_report`)
- **DET curve**: detection error tradeoff (`--det_curve`, `--save_det_plot`)
- **Latency measurement**: per-chunk inference timing (`--benchmark_latency`)
- **Benchmark JSON**: structured report for experiment tracking (`--benchmark`)
- **HTML report**: self-contained evaluation report (`--report_html`)

### Deployment

- **X-CUBE-AI / stedgeai**: generate → flash → validate pipeline
- **Board test**: standalone on-device inference (`board-test`) — reads WAV from SD card, STFT on Cortex-M55, inference on NPU


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




