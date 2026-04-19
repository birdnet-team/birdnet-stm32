# Training

## Basic usage

```bash
python -m birdnet_stm32 train \
  --data_path_train data/train \
  --audio_frontend hybrid \
  --mag_scale pwl \
  --checkpoint_path checkpoints/my_model.keras
```

The script saves these files alongside the checkpoint:

- `my_model.keras` — trained Keras model
- `my_model_model_config.json` — conversion metadata (frontend, shapes, etc.)
- `my_model_labels.txt` — ordered class names
- `my_model_history.csv` — per-epoch training metrics (loss, ROC-AUC)
- `my_model_curves.png` — loss and ROC-AUC training curves plot
- `my_model_train_state.json` — epoch counter for `--resume`

## Audio frontends

| Frontend | Input to model | Description |
|---|---|---|
| `hybrid` (default) | Linear magnitude STFT | Model applies a learned mel mixer and magnitude scaling. Best for deployment. |
| `librosa` | Mel spectrogram | Spectrogram computed offline with librosa. Simplest, but frontend is not in the graph. |
| `raw` | Raw waveform | Model learns the filterbank from scratch via Conv2D. Most flexible, highest memory. |

!!! note "Deprecated aliases"
    `precomputed` (now `librosa`) and `tf` (now `raw`) still work but emit
    deprecation warnings and will be removed in a future release.

!!! warning "Raw frontend memory limit"
    At 22 kHz × 3 s the raw input exceeds 65,536 samples (the 16-bit activation
    size limit on the N6 NPU). Use `hybrid` or `librosa` for deployment, or
    reduce sample rate / chunk duration.

## Magnitude scaling

| Mode | Description | Quantization friendliness |
|---|---|---|
| `pwl` (default) | Piecewise-linear learned compression | Excellent — recommended for deployment |
| `pcen` | Per-channel energy normalization | Good — uses pooling + convolution |
| `db` | Log-scale (decibels) | Poor — avoid for quantized models |
| `none` | No compression | Baseline only |

## Model architecture

The DS-CNN is scaled with two knobs:

- **`--alpha`** (width multiplier): scales channel counts across all stages.
  Default 1.0. Values like 0.5 or 0.75 produce smaller models.
- **`--depth_multiplier`**: repeats each depthwise-separable block. Default 1.
  Increase to 2 for deeper models.

!!! tip "Channel alignment"
    Keep channel counts as multiples of 8 for optimal NPU vectorization. The
    model builder enforces this automatically via `_make_divisible`.

## Training options

### Data augmentation

- **Mixup**: controlled by `--mixup_alpha` (default 0.2, 0 disables) and
  `--mixup_probability` (default 0.25). Blends pairs of training examples
  and their labels.
- **SpecAugment**: enable with `--spec_augment`. Applies random frequency
  and time masking to spectrograms during training. Control mask widths with
  `--freq_mask_max` (default 8 bins) and `--time_mask_max` (default 25 frames).

### Loss function

- **Binary crossentropy** (default): standard multi-label loss.
- **Focal loss**: `--loss focal` down-weights well-classified examples,
  focusing on hard negatives. Tune with `--focal_gamma` (default 2.0).
  Useful for imbalanced class distributions.

### Optimizer

Select with `--optimizer` (default `adam`):

| Optimizer | Description |
|---|---|
| `adam` | Adaptive moment estimation (default) |
| `sgd` | SGD with momentum 0.9 |
| `adamw` | AdamW with decoupled weight decay |

Set weight decay with `--weight_decay` (default 0, only used by `adamw`).

### Deterministic mode

Use `--deterministic` to set all random seeds (Python, NumPy, TensorFlow)
and enable `TF_DETERMINISTIC_OPS`. Optionally specify `--seed` (default 42).

### Gradient clipping

Use `--grad_clip 1.0` to clip gradients by global norm. Prevents exploding
gradients, especially useful with large models or unstable training. Default 0
(disabled).

### Class weighting

Use `--class_weights balanced` to apply inverse-frequency class weights.
Useful for imbalanced datasets where some species have fewer training files.

### Mixed precision

Use `--mixed_precision` to enable FP16 compute with FP32 accumulation.
Reduces memory usage and speeds up training on GPUs with Tensor Cores.

### Resumable training

Use `--resume` to continue training from a previously saved checkpoint.
The optimizer state is recompiled and training resumes from the last saved
epoch. Example:

```bash
# Initial training (interrupted or completed at epoch 30)
python -m birdnet_stm32 train --epochs 30 --checkpoint_path ckpt/model.keras ...

# Resume and extend to 50 epochs
python -m birdnet_stm32 train --epochs 50 --resume --checkpoint_path ckpt/model.keras ...
```

### Quantization-Aware Training (QAT)

Use `--qat` to fine-tune a pretrained model with simulated INT8 quantization
noise. This closes the accuracy gap between the float Keras model and the
quantized TFLite model by teaching the weights to survive quantization.

QAT works by injecting fake-quantization noise into kernel weights during
training while maintaining full-precision shadow copies. BatchNorm layers are
frozen to prevent running statistics drift. No FakeQuant ops remain in the
saved model, so the N6 NPU runs it without issues.

```bash
# Step 1: Normal training
python -m birdnet_stm32 train --data_path_train data/train \
  --epochs 50 --checkpoint_path checkpoints/model.keras

# Step 2: QAT fine-tuning (lower LR, fewer epochs)
python -m birdnet_stm32 train --data_path_train data/train --qat \
  --checkpoint_path checkpoints/model.keras \
  --epochs 10 --learning_rate 0.0001

# Step 3: Convert the QAT model
python -m birdnet_stm32 convert \
  --checkpoint_path checkpoints/model_qat.keras \
  --model_config checkpoints/model_model_config.json \
  --data_path_train data/train
```

The QAT model is saved as `{name}_qat.keras` alongside the original.

### Learning rate

Cosine decay schedule from `--learning_rate` (default 0.001) to near-zero
over `--epochs` (default 50). Early stopping on validation loss with patience
of 10 epochs.

### Hyperparameter tuning with Optuna

Use `--tune` to run an automated hyperparameter search using Optuna (requires
`pip install -e ".[tune]"`). The tuner explores alpha, depth_multiplier,
embeddings_size, learning_rate, dropout, batch_size, mixup_alpha,
label_smoothing, optimizer, weight_decay, grad_clip, use_se,
use_inverted_residual, use_attention_pooling, se_reduction, and
expansion_factor. It maximizes `val_roc_auc` with MedianPruner.

```bash
python -m birdnet_stm32 train \
  --data_path_train data/train \
  --tune --n_trials 20 --epochs 30
```

Set `--n_trials` to control how many configurations to try (default 20).

## Full argument reference

| Argument | Default | Description |
|---|---|---|
| `--data_path_train` | *(required)* | Path to training data |
| `--max_samples` | None | Max files per class |
| `--upsample_ratio` | 0.5 | Minority class upsample ratio |
| `--sample_rate` | 24000 | Audio sample rate (Hz) |
| `--num_mels` | 64 | Number of mel frequency bins |
| `--spec_width` | 256 | Spectrogram width (frames) |
| `--fft_length` | 512 | FFT window length |
| `--chunk_duration` | 3 | Chunk duration (seconds) |
| `--max_duration` | 30 | Max seconds to load per file |
| `--audio_frontend` | hybrid | `librosa`, `hybrid`, `raw`, `mfcc`, or `log_mel` |
| `--mag_scale` | pwl | `pwl`, `pcen`, `db`, or `none` |
| `--embeddings_size` | 256 | Embedding channels before head |
| `--alpha` | 1.0 | Model width scaling |
| `--depth_multiplier` | 1 | Block repeats per stage |
| `--frontend_trainable` | False | Make frontend weights trainable |
| `--mixup_alpha` | 0.2 | Mixup alpha (0 disables) |
| `--mixup_probability` | 0.25 | Fraction of batch to mix |
| `--spec_augment` | False | Enable SpecAugment masking |
| `--freq_mask_max` | 8 | Max frequency mask width (bins) |
| `--time_mask_max` | 25 | Max time mask width (frames) |
| `--dropout` | 0.5 | Dropout rate before classifier head |
| `--optimizer` | adam | `adam`, `sgd`, or `adamw` |
| `--weight_decay` | 0.0 | Weight decay (adamw only) |
| `--loss` | auto | `auto` (BCE) or `focal` |
| `--focal_gamma` | 2.0 | Focal loss focusing parameter |
| `--label_smoothing` | 0.0 | Label smoothing factor (0 = off) |
| `--use_se` | False | Add SE channel attention to each block |
| `--se_reduction` | 4 | SE channel reduction factor |
| `--use_inverted_residual` | False | Use inverted residual blocks |
| `--expansion_factor` | 6 | Expansion factor for inverted residuals |
| `--use_attention_pooling` | False | Use attention pooling instead of GAP |
| `--n_mfcc` | 20 | Number of MFCC coefficients (mfcc frontend only) |
| `--grad_clip` | 0.0 | Max gradient norm for clipping (0 = disabled) |
| `--class_weights` | none | `none` or `balanced` (inverse-frequency) |
| `--mixed_precision` | False | Enable FP16 mixed precision training |
| `--resume` | False | Resume training from checkpoint |
| `--deterministic` | False | Enable deterministic training |
| `--seed` | 42 | Random seed (with `--deterministic`) |
| `--batch_size` | 32 | Batch size |
| `--epochs` | 50 | Number of epochs |
| `--learning_rate` | 0.001 | Initial learning rate |
| `--val_split` | 0.2 | Validation split fraction |
| `--checkpoint_path` | *(required)* | Output path (.keras) |
| `--tune` | False | Run Optuna hyperparameter search |
| `--n_trials` | 20 | Number of Optuna trials |
| `--qat` | False | Quantization-aware fine-tuning |

## Noise classes

Place audio in folders named `noise`, `silence`, `background`, or `other`
under `data/train/`. These receive all-zero label vectors and help the model
learn to reject non-bird sounds.
