# Model

## DS-CNN architecture

The backbone is a depthwise-separable convolutional neural network (DS-CNN)
with 4 stages, inspired by MobileNetV1. All variants are built by the single
`build_dscnn_model()` function in `birdnet_stm32/models/dscnn.py`.

### Block structure

Each depthwise-separable block (default):

```mermaid
flowchart TD
    X["Input"] --> DW["Depthwise Conv2D 3×3"]
    DW --> BN1["BatchNorm"]
    BN1 --> R1["ReLU6"]
    R1 --> PW["Pointwise Conv2D 1×1"]
    PW --> BN2["BatchNorm"]
    BN2 --> R2["ReLU6"]
    X -. "residual\n(stride=1, same channels)" .-> ADD["Add"]
    R2 --> ADD
    ADD --> Y["Output"]
```

When `stride=1` and input/output channels match, a **residual skip connection**
is added.

#### Removed: inverted residual and squeeze-and-excite blocks

Earlier versions could build a MobileNetV2-style backbone with inverted
residual blocks and squeeze-and-excite attention. Both were removed because
that backbone could not reach the INT8 parity gates: per-channel PTQ landed far
below the 0.95 minimum, and quantization-aware training did not recover the
lower tail. The dominant residual errors sit at the signed linear projection and
the `Add` boundaries inside an inverted-residual block, where the requantization
step has to reconcile two differently scaled tensors.

Plain depthwise separable blocks have no such boundary and quantize cleanly, so
they are the only path the builder offers. Removing the options also removed a
trap: they were enabled by default, which made the quantization-hostile
architecture the one you got unless you opted out.

### Stage configuration

| Stage | Base output channels | Stride | Base repeats |
|---|---|---|---|
| Stem | 16 × alpha | (1, 2) | 1 |
| 1 | 32 × alpha | (2, 2) | 2 × depth_multiplier |
| 2 | 64 × alpha | (2, 2) | 3 × depth_multiplier |
| 3 | 128 × alpha | (2, 2) | 4 × depth_multiplier |
| 4 | 256 × alpha | (2, 2) | 2 × depth_multiplier |

All channel counts are rounded to the nearest multiple of 8 via
`_make_divisible(channels, 8)` (defined in `birdnet_stm32/models/blocks.py`).

### Head

After the final stage:

1. **Global Average Pooling** (or **Attention Pooling** with `--use_attention_pooling`)
2. **Dropout** (0.5)
3. **Dense** with sigmoid activation → `[B, num_classes]`

Attention pooling learns per-channel weights before averaging, giving the
model a soft spatial attention mechanism while remaining NPU-compatible.

## Building blocks

All reusable building blocks live in `birdnet_stm32/models/blocks.py`:

- `_make_divisible(v, divisor)` — round channel counts to multiples of `divisor`
- `AttentionPooling` — Keras Layer that learns per-channel spatial attention weights

## Scaling knobs

### `alpha` (width multiplier)

Scales channel counts across all stages. Default 1.0.

| alpha | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Relative params |
|---|---|---|---|---|---|
| 0.25 | 8 | 16 | 32 | 64 | ~6% |
| 0.5 | 16 | 32 | 64 | 128 | ~25% |
| 1.0 | 32 | 64 | 128 | 256 | 100% |
| 1.5 | 48 | 96 | 192 | 384 | ~225% |

### `depth_multiplier` (block repeats)

Repeats each DS block within a stage. Default 1. Only the first block in each
stage uses stride 2; subsequent blocks use stride 1 with residual connections.

## Model profiler

Use `birdnet_stm32/models/profiler.py` to analyze a model before deployment:

- `profile_model(model)` — per-layer MACs, params, and activation memory
- `check_n6_compatibility(model)` — flags layers using ops outside the N6 NPU
  supported set

## N6 NPU constraints

!!! danger "N6 compatibility is the absolute priority"
    Every model architecture change must be validated against the STM32N6 NPU
    operator set. Always run `stedgeai analyze` before committing model changes.

Key constraints:

- **Channel alignment**: all channel counts must be multiples of 8 for NPU
  vectorization.
- **Supported ops**: Conv2D, DepthwiseConv2D, BatchNormalization, ReLU6,
  GlobalAveragePooling2D, Dense, Add, Multiply (SE), Reshape, Sigmoid.
  Verify exotic ops with `stedgeai analyze`.
- **Activation memory**: intermediate activations must fit in NPU SRAM. Large
  spatial dimensions or channel counts may exceed limits.
- **QAT is safe**: the activation-aware QAT graph (`--qat`) checkpoints a
  synchronized clean deployment model and uses final conversion's exact
  calibration manifest, so the resulting `.keras` file has no FakeQuant ops
  and remains compatible with standard PTQ conversion.
