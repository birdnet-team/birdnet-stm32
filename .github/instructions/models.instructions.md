---
description: "Use when working with model architecture, adding new layers, scaling, or compiling specific structures for the N6 NPU."
applyTo: "birdnet_stm32/models/**/*.py"
---

# Model Architecture Conventions

When editing or creating models in `birdnet_stm32/models/`, strictly adhere to these conventions:

## N6 NPU Constraints
- **Compatibility First**: STM32N6 NPU is the deployment target. Every new layer or change must be verifiable via `stedgeai analyze` / `stedgeai generate`.
- **Supported Operations**: Try to stick to basic CNN layers (Conv2D, DepthwiseConv2D, Dense, Add, Multiply, AveragePooling2D, MaxPooling2D, ReLU/ReLU6/Sigmoid). Stick to standard Keras ops. Avoid complex custom operations unless manually lowering them.
- **Input Dimension Check**: Due to the N6 16-bit DMA buffer limits, the number of input samples sent directly to the N6 (such as the `raw` frontend) must not exceed 65536 samples (e.g., 24000Hz * 2s = 48000 <= 65536).
- **Channel Alignment**: Ensure channel dimensions (width, width multipliers) remain multiples of 8 to maintain optimal NPU vectorization.

## Network Structure (DS-CNN)
- Use Depthwise Separable convolutions extensively.
- Rely on scaling parameters `alpha` (width multiplier) and `depth_multiplier` (block repeats).
- Optional enhancements: `--use_se` (squeeze-and-excite channel attention), `--use_inverted_residual` (MobileNetV2-style blocks), `--use_attention_pooling` (learned spatial attention instead of GAP).
- Building blocks are in `birdnet_stm32/models/blocks.py`: `se_block()`, `inverted_residual_block()`, `AttentionPooling`.
- Use `birdnet_stm32/models/profiler.py` to check N6 compatibility and estimate MACs/params.
- Enforce the `_make_divisible` check (in `blocks.py`) to ensure the channel counts are suitable for integer operations.
- Quantization is done post-training (PTQ). Ensure model architecture supports PTQ. Avoid intermediate states that explode INT8 dynamic ranges.