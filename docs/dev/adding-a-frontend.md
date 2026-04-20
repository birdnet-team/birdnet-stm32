# Adding a Frontend

Step-by-step guide to adding a new audio frontend mode to BirdNET-STM32.

## Overview

Audio frontends are defined in `birdnet_stm32/models/frontend.py`. Each
frontend transforms a specific input representation (waveform, STFT, mel
spectrogram, etc.) into a fixed-size tensor for the DS-CNN backbone.

## Steps

### 1. Register the frontend name

Add the canonical name to `VALID_FRONTENDS` in `frontend.py`:

```python
VALID_FRONTENDS = ("librosa", "hybrid", "raw", "mfcc", "log_mel", "your_frontend")
```

### 2. Implement the frontend branch

In the `AudioFrontendLayer.call()` method, add a branch for your frontend:

```python
elif self.frontend_mode == "your_frontend":
    x = self._your_frontend_ops(inputs)
```

Keep the implementation as a sequence of standard Keras layers (Conv2D,
DepthwiseConv2D, BatchNormalization, ReLU6, etc.) to ensure NPU compatibility.

### 3. Define the input shape

In `AudioFrontendLayer.build()` or the model builder, specify the expected
input shape for your frontend. The model builder in `dscnn.py` uses this to
construct the correct `Input` layer.

### 4. Add magnitude scaling support

If your frontend produces a spectrogram-like output, apply the
`MagnitudeScalingLayer` after your frontend's feature extraction:

```python
x = self.mag_layer(x)  # PWL/PCEN/dB/none
```

### 5. Update the data pipeline

In `birdnet_stm32/data/dataset.py`, add the preprocessing logic for your
frontend. The data generator must produce inputs in the shape your frontend
expects.

### 6. N6 compatibility checklist

Before merging, verify:

- [ ] All ops are in the [N6 NPU supported set](implementation.md#n6-npu-operator-coverage)
- [ ] Channel counts are multiples of 8
- [ ] Input size does not exceed the 16-bit activation limit (65,536 elements)
- [ ] Run `stedgeai analyze` on the exported TFLite model
- [ ] Cosine similarity > 0.95 after PTQ

### 7. Add tests

Create `tests/test_frontend_your_frontend.py` with:

- Output shape test for known input dimensions
- Magnitude scaling integration test
- Round-trip test: build model → export TFLite → run inference

### 8. Update documentation

- Add a section to `docs/dev/audio-frontends.md`
- Update the frontend count in `docs/index.md`
- Update the mermaid diagram in `docs/dev/architecture.md`

## Example: the `mfcc` frontend

The `mfcc` frontend is a good reference for a simple precomputed frontend:

1. Input: `[B, num_mfcc, spec_width, 1]` — precomputed offline
2. In-graph ops: magnitude scaling only
3. No trainable parameters in the frontend itself
4. Data pipeline computes MFCCs using `librosa.feature.mfcc()`

For an in-graph frontend (like `hybrid` or `raw`), the implementation is more
involved because the feature extraction layers must be NPU-compatible.
