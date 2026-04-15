# Quantization

## Strategy

BirdNet-STM32 uses **post-training quantization (PTQ)** to convert trained
Keras models to INT8 TFLite for the STM32N6 NPU.

| Aspect | Choice | Rationale |
|---|---|---|
| Weight precision | INT8 | Required by N6 NPU |
| Activation precision | INT8 | Required by N6 NPU |
| I/O precision | Float32 | Audio inputs are continuous-valued; INT8 I/O destroys precision |
| Calibration | Representative dataset | 1024 samples from training data |

## Why not QAT?

Quantization-aware training (QAT) inserts fake-quantization nodes during
training, which can improve INT8 accuracy. However:

- **N6 compatibility risk**: QAT may leave artifacts (fake-quant nodes,
  unsupported fused ops) that prevent deployment on the STM32N6 NPU.
- **Operator coverage**: the N6 NPU operator set is limited; fused QAT ops
  may not be supported.
- **PTQ is sufficient**: with PWL magnitude scaling and proper representative
  dataset calibration, PTQ consistently achieves >0.95 cosine similarity.

!!! danger
    Do not enable QAT without first verifying the resulting TFLite model with
    `stedgeai analyze`. A model that trains fine may fail to deploy.

## Representative dataset

The calibration dataset is critical for PTQ quality:

- **Source**: randomly sampled training files, center-cropped to chunk duration.
- **Size**: 1024 samples (default). More is not necessarily better.
- **Diversity**: moderate diversity is ideal. Overly diverse datasets widen
  INT8 quantization ranges, reducing precision.
- **Target**: cosine similarity > 0.95 between Keras and TFLite outputs.

## Cosine similarity troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Cosine sim < 0.90 | `db` magnitude scaling | Switch to `pwl` |
| Cosine sim 0.90–0.95 | Too-diverse representative set | Reduce `--num_samples` or filter by SNR |
| Cosine sim varies across runs | Non-deterministic data order | Set `--deterministic` (when available) |
| stedgeai analyze fails | Unsupported op in model | Check operator, simplify model |

## Channel alignment

The N6 NPU vectorizes computation in groups of 8 channels. Misaligned channel
counts either:

- Waste compute cycles (padding to next multiple of 8)
- Fail compilation entirely

The model builder enforces alignment via `_make_divisible(channels, 8)`. When
adding new layers or architectures, always maintain this constraint.

## Validation workflow

After conversion, always:

1. Check cosine similarity in `convert.py` output (target > 0.95).
2. Run `stedgeai analyze` on the `.tflite` to verify N6 compatibility.
3. Run `stedgeai validate` on-device to confirm end-to-end correctness.
