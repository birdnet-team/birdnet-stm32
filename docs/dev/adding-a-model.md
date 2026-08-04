# Adding a Model

Step-by-step guide to adding a new model architecture to BirdNET-STM32.

## Overview

A model is a builder function that returns an uncompiled Keras model.
`birdnet_stm32/models/dscnn.py` is the reference implementation; the training
CLI calls its builder directly.

## Steps

### 1. Create the model module

Create `birdnet_stm32/models/your_model.py` with a builder function:

```python
"""Your model architecture for BirdNET-STM32."""

import tensorflow as tf
from tensorflow.keras import layers, Model

from birdnet_stm32.models.blocks import _make_divisible


def build_your_model(
    input_shape: tuple[int, ...],
    num_classes: int,
    alpha: float = 1.0,
    **kwargs,
) -> Model:
    """Build your model.

    Args:
        input_shape: Input tensor shape (H, W, C).
        num_classes: Number of output classes.
        alpha: Width multiplier for channel counts.

    Returns:
        Uncompiled Keras model.
    """
    inputs = layers.Input(shape=input_shape)
    # ... your architecture ...
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)
    return Model(inputs, outputs, name="your_model")
```

### 2. Wire the builder into an entry point

There is intentionally no model registry: BirdNET-STM32 currently exposes one
architecture and `birdnet_stm32/cli/train.py` calls `build_dscnn_model()`
directly. For an experiment, call your builder from a dedicated script. To add
a supported architecture, introduce an explicit CLI choice and dispatch at the
call site in `train.py`, then mirror that choice in the saved `ModelConfig`.

```python
from birdnet_stm32.models.your_model import build_your_model

model = build_your_model(input_shape=(64, 256, 1), num_classes=len(classes))
```

### 3. N6 compatibility constraints

Your model **must** satisfy these constraints:

| Constraint | Requirement |
|---|---|
| Channel alignment | All channel counts must be multiples of 8 |
| Supported ops | Only ops in the [N6 supported set](implementation.md#n6-npu-operator-coverage) |
| Activation memory | Intermediate activations must fit in NPU SRAM |
| Output activation | Use `sigmoid` (not `softmax`) for multi-label classification |
| No FakeQuant ops | Model must be standard Keras — no QAT artifacts |

Use `_make_divisible(channels, 8)` from `birdnet_stm32/models/blocks.py` for
all channel counts.

### 4. Support scaling knobs

The training CLI exposes `--alpha` (width) and `--depth_multiplier` (depth).
Your builder should accept and honor these parameters for consistency with
the rest of the pipeline.

### 5. Add the model profiler check

Run `birdnet_stm32/models/profiler.py` on your model to verify:

```python
from birdnet_stm32.models.profiler import profile_model, check_n6_compatibility

model = build_your_model(input_shape=(64, 256, 1), num_classes=100)
profile_model(model)       # Per-layer MACs, params, activation memory
check_n6_compatibility(model)  # Flags unsupported ops
```

### 6. Add tests

Create `tests/test_your_model.py` with:

- Shape test: output is `[B, num_classes]`
- Channel alignment test: all conv layers have channels divisible by 8
- Scaling test: different `alpha` values produce different model sizes
- Quantization smoke test: model converts to TFLite without errors

### 7. Update documentation

- Add a section to `docs/dev/model.md`
- Update `docs/index.md` if the model is a significant addition

## Reference: DS-CNN builder

See `birdnet_stm32/models/dscnn.py` for the reference implementation. Key
patterns to follow:

- Use `_make_divisible()` for all channel computations
- Support residual connections when stride=1 and channels match
- Prefer simple NPU-supported activations such as ReLU or ReLU6; verify the
  generated graph because an upper clip can become a software `MINIMUM` epoch
- Add `BatchNormalization` after every convolution
- Use `GlobalAveragePooling2D` → `Dropout` → `Dense(sigmoid)` as the head
