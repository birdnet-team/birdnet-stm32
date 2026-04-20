# Implementation Notes

Design rationale for key architectural decisions in BirdNET-STM32.

## Why DS-CNN?

Depthwise-separable convolutions (DS-CNN) are the backbone because:

- **Parameter efficiency**: a depthwise-separable block uses ~8-9× fewer
  parameters than a standard convolution with the same receptive field.
- **NPU compatibility**: the STM32N6 NPU natively supports `DepthwiseConv2D`
  and `Conv2D` (pointwise). No custom ops needed.
- **Proven track record**: MobileNetV1/V2-style architectures are the de facto
  standard for on-device audio classification (Google's keyword spotting, ARM
  ML Zoo, etc.).

The 4-stage design with stride-2 downsampling gives a 16× spatial reduction,
which is sufficient for mel spectrograms of typical size (64×256).

## Why PWL over PCEN/dB?

| Scaling | Quantization behavior | N6 compatibility | Notes |
|---|---|---|---|
| **PWL** (piecewise-linear) | Excellent — depthwise conv + ReLU only | Full | Recommended default |
| **PCEN** | Good — pooling + conv + ReLU | Full | Slightly more complex |
| **dB** (log scale) | Poor — log op creates wide dynamic range | Partial | Avoid for deployment |

PWL achieves comparable compression to PCEN while using only operations that
quantize cleanly to INT8. The learned breakpoints adapt to the dataset's
dynamic range during training.

## Why float32 I/O?

Audio spectrograms are continuous-valued signals with meaningful precision at
small magnitudes. Quantizing model inputs to INT8 would:

1. **Destroy quiet details**: bird calls often have low-energy harmonics that
   fall below INT8 resolution.
2. **Waste quantization range**: spectrogram values are not uniformly
   distributed — most energy concentrates in a few frequency bands.
3. **Complicate preprocessing**: the STM32 firmware would need to quantize
   float STFT output to INT8 before feeding the NPU, adding complexity
   and latency.

The pipeline enforces **float32 inputs and outputs** with **INT8 internal
weights and activations**. This is the standard approach for audio/speech
models on edge devices.

## N6 NPU operator coverage

The STM32N6 Neural-ART NPU supports a subset of TFLite operators. Verified
compatible ops (as of X-CUBE-AI 10.2):

| Category | Supported operators |
|---|---|
| Convolution | Conv2D, DepthwiseConv2D |
| Normalization | BatchNormalization (fused into conv) |
| Activation | ReLU, ReLU6, Sigmoid |
| Pooling | GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D |
| Arithmetic | Add, Multiply |
| Reshape | Reshape, Flatten |
| Linear | Dense (MatMul + BiasAdd) |
| Other | Concatenate, Pad |

!!! danger "Always verify with stedgeai"
    This table is a guideline. Always run `stedgeai analyze` on your TFLite
    model before attempting deployment. Op support can change between
    X-CUBE-AI versions.

### Known unsupported ops

- `Softmax` — use `Sigmoid` for multi-label classification instead
- `LayerNormalization` — use `BatchNormalization`
- `GRU` / `LSTM` — no recurrent op support
- `ResizeBilinear` / `ResizeNearestNeighbor` — no upsampling
- `Exp`, `Log`, `Pow` — no transcendental math (this is why `db` scaling is
  problematic)

## Channel alignment

The N6 NPU vectorizes computation in groups of 8 channels. The model builder
enforces this via `_make_divisible(channels, 8)` in
`birdnet_stm32/models/blocks.py`.

When `alpha=0.25`, stage 1 gets `64 × 0.25 = 16` channels (aligned).
When `alpha=0.1`, stage 1 would get `64 × 0.1 = 6.4` → rounded to `8`.

Misaligned channels either waste compute (the NPU pads to the next multiple
of 8) or fail compilation entirely.

## QAT implementation

The QAT implementation uses **shadow-weight fake-quantization** rather than
TensorFlow Model Optimization Toolkit (tfmot), because:

1. **tfmot is incompatible with Keras 3** (as of 2026).
2. **tfmot injects FakeQuant ops** that may not be supported by the N6 NPU.
3. **Shadow weights are simpler**: during the forward pass, kernel weights are
   fake-quantized to INT8 range; gradients flow through the straight-through
   estimator to update the original float32 weights.

The saved `.keras` model contains only standard float32 weights — no FakeQuant
nodes. Standard PTQ then quantizes the QAT-hardened weights, typically
recovering 1-3% accuracy compared to PTQ-only.

See `birdnet_stm32/training/qat.py` for the implementation.
