# Architecture

## Pipeline overview

```
Audio (.wav)
    │
    ▼
┌──────────────────┐
│  Audio Frontend   │  librosa / hybrid / raw
│  (spectrogram)    │
└────────┬─────────┘
         │  [B, H, W, 1]  (mel spectrogram)
         ▼
┌──────────────────┐
│  Magnitude Scale  │  pwl / pcen / db / none
└────────┬─────────┘
         │  [B, H, W, 1]
         ▼
┌──────────────────┐
│    DS-CNN Body    │  4 stages × depth_multiplier blocks
│  (DW-Sep Conv)    │  channels scaled by alpha
└────────┬─────────┘
         │  [B, H', W', C]
         ▼
┌──────────────────┐
│  Global Avg Pool  │
│  Dropout → Dense  │
└────────┬─────────┘
         │  [B, num_classes]
         ▼
     Predictions
```

## Component boundaries

| Component | Module | Responsibility |
|---|---|---|
| Audio I/O | `birdnet_stm32.audio.io` | Load, resample, chunk WAV files |
| Spectrogram | `birdnet_stm32.audio.spectrogram` | Compute mel spectrograms (librosa path) |
| Frontend layer | `birdnet_stm32.models.frontend` | In-graph frontend (hybrid/raw modes + mag scaling) |
| Model builder | `birdnet_stm32.models.dscnn` | DS-CNN construction with scaling knobs |
| Data pipeline | `birdnet_stm32.data.dataset` | File discovery, upsampling, tf.data generation |
| Training | `birdnet_stm32.training.trainer` | Training loop, LR schedule, callbacks |
| Conversion | `birdnet_stm32.conversion.quantize` | PTQ, representative dataset, TFLite export |
| Validation | `birdnet_stm32.conversion.validate` | Keras vs. TFLite output comparison |
| Evaluation | `birdnet_stm32.evaluation` | Pooling, metrics (ROC-AUC, cmAP, F1), reporting |
| Deployment | `birdnet_stm32.deploy` | Config resolution, stedgeai/n6_loader wrappers |

## Data flow

### Training

1. `dataset.load_file_paths_from_directory()` discovers WAV files and class labels.
2. `dataset.upsample_minority_classes()` balances class representation.
3. `data.generator` yields `(spectrogram, label)` or `(waveform, label)` batches.
4. The `AudioFrontendLayer` (if hybrid/raw) processes inputs inside the model graph.
5. DS-CNN body produces logits; sigmoid activation gives per-class scores.
6. Cosine LR decay + early stopping; best model saved to `.keras`.

### Inference

1. Audio loaded and chunked into fixed-length segments.
2. Each chunk passes through the full model (frontend + backbone + head).
3. Chunk-level scores are pooled to file-level (avg/max/LME).
4. Metrics computed against ground-truth labels.

### Deployment

1. `stedgeai generate` converts TFLite → N6-optimized binary.
2. `n6_loader.py` flashes the binary to the STM32N6570-DK via serial.
3. `stedgeai validate` runs on-device inference and compares to reference.

## Key design decisions

- **Float32 I/O**: Audio spectrograms are continuous-valued; INT8 inputs would
  lose meaningful precision. Only internal weights/activations are quantized.
- **PWL over PCEN/dB**: Piecewise-linear magnitude scaling quantizes cleanly
  (no log ops, no running statistics). PCEN is acceptable; dB should be avoided.
- **Hybrid as default frontend**: Keeps the STFT offline (cheaper) while
  learning the mel projection in-graph, giving the TFLite model a complete
  mel-to-prediction path.
- **Channel alignment to 8**: The N6 NPU vectorizes in groups of 8. Misaligned
  channels waste compute cycles or fail compilation.
