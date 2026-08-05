# Training

## Basic usage

```bash
python -m birdnet_stm32 train \
  --data_path_train data/train \
  --audio_frontend hybrid \
  --mag_scale pwl \
  --checkpoint_path checkpoints/my_model.keras
```

For a leakage-safe precomputed validation split and stable output order, pass a
separate validation root and a one-label-per-line classes file:

```bash
python -m birdnet_stm32 train \
  --data_path_train data/train \
  --data_path_val data/validation \
  --classes_file data/labels.txt
```

The class file controls model-output order. Folders named `noise`, `silence`,
`background`, or `other` are still loaded as all-zero examples and must not be
listed as outputs. When `--data_path_val` is present, `--val_split` is ignored.

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
| `raw` | Peak-normalized waveform | Model applies a mel-seeded, trainable Gabor quadrature filterbank. Most flexible, highest input memory. |
| `mfcc` | MFCC features | Host-precomputed compact representation. |
| `log_mel` | Log-mel spectrogram | Host-precomputed log-scaled mel features. |

Only the canonical names above are accepted; the former `precomputed` and `tf`
aliases have been removed.

!!! warning "Raw frontend memory limit"
    The raw input must contain fewer than 65,536 samples. At 24 kHz, use a
    chunk no longer than about 2.7 seconds; 2.5 seconds is the tested setting.

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
  `--mixup_probability` (default 0.25). Uses Dirichlet multi-source mixing
  (2–3 sources per sample) to realistically emulate overlapping bird
  vocalizations. Labels are combined via element-wise max.
- **SpecAugment**: enabled by default. Applies random frequency and time
  masking to spectrograms during training. Disable with `--no_spec_augment`.
  Control mask widths with `--freq_mask_max` (default 8 bins) and
  `--time_mask_max` (default 25 frames).
- **Smart crop**: long recordings (> 2 chunks) are automatically cropped to
  salient regions using short-time energy (STE) analysis, reducing label
  noise from silent or irrelevant segments.
- **Multi-chunk I/O reuse**: long files (e.g. 60 s recordings) yield up to
  `--max_chunks_per_file` (default 3) salient chunks per file open, stored
  in a memory-bounded shuffled reservoir. This avoids redundant FLAC decode +
  resample for the same file across epochs.

### Loss function

The classifier head is always sigmoid + binary crossentropy. Soundscape
recordings are inherently multi-label, so we always optimise per-class
probabilities even when the source label is single-class.

### Optimizer

Select with `--optimizer` (default `adam`):

| Optimizer | Description |
|---|---|
| `adam` | Adaptive moment estimation (default) |
| `sgd` | SGD with momentum 0.9 |
| `adamw` | AdamW with decoupled weight decay |

Set weight decay with `--weight_decay` (default 0, only used by `adamw`).

### Deterministic mode

Training is always deterministic — all random seeds (Python, NumPy,
TensorFlow) are set and `TF_DETERMINISTIC_OPS` is enabled automatically.
Use `--seed` (default 42) to change the RNG seed.

### Gradient clipping

Gradient clipping by global norm is enabled by default (`--grad_clip 1.0`).
Set to 0 to disable. Prevents exploding gradients, especially useful with
large models or unstable training.

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

!!! warning "QAT requires a pretrained model"
    Always train normally first, then fine-tune with `--qat`. Do **not** use
    `--qat` from scratch — the quantization noise destabilizes randomly
    initialized weights and the model will not converge. The dataset must
    have the same classes as the pretrained model; use `--linear_probe` to
    adapt to a different class set first.

QAT calibrates activation ranges on the converter's exact deterministic,
class-stratified training manifest (1,024 samples by default), then trains with
per-channel INT8 kernel grids and per-tensor INT8 activation grids. The
simulation includes the quantized waveform input and the kernels and
elementwise boundaries inside the custom raw frontend. BatchNorm layers are
frozen. Standard variables are shared with a clean deployment graph; cloned
frontend variables are synchronized before each checkpoint. Only the clean
graph is saved, so no FakeQuant ops remain in the model.

A frozen copy of the untouched checkpoint acts as the teacher. QAT minimizes
the normal label loss, per-output Bernoulli KL divergence, and per-sample
cosine distance from that teacher. This constrains background and
low-confidence probabilities while directly optimizing the lower tail that
the release parity gate measures.

```bash
# Step 1: Normal training
python -m birdnet_stm32 train --data_path_train data/train \
  --epochs 50 --checkpoint_path checkpoints/model.keras

# Step 2: QAT fine-tuning (lower LR, fewer epochs)
python -m birdnet_stm32 train --data_path_train data/train \
  --data_path_val data/validation --classes_file data/labels.txt --qat \
  --checkpoint_path checkpoints/model.keras \
  --qat_calibration_samples 1024 \
  --epochs 6 --learning_rate 0.00002

# Step 3: Convert the QAT model
python -m birdnet_stm32 convert \
  --checkpoint_path checkpoints/model_qat.keras \
  --model_config checkpoints/model_model_config.json \
  --data_path_train data/train
```

The QAT model is saved as `{name}_qat.keras` alongside the original.

### Linear probing

Use `--linear_probe` to freeze a pretrained backbone and train only a new
classification head on your custom species dataset. This is useful when you
have a pretrained model (e.g. a large BirdNET checkpoint) and want to adapt
it to a different set of species with limited data.

```bash
python -m birdnet_stm32 train --data_path_train data/my_species \
  --linear_probe --checkpoint_path checkpoints/pretrained.keras \
  --epochs 20 --learning_rate 0.001
```

The probe model is saved as `{name}_probe.keras` with a new labels file.

### Learning rate

A two-epoch linear warmup reaches `--learning_rate` (default 0.001), followed
by cosine decay to near-zero over `--epochs` (default 50). Best-checkpoint
selection and early stopping monitor validation ROC-AUC with patience 10.

### Hyperparameter tuning with Optuna

Use `--tune` to run an automated hyperparameter search using Optuna (requires
`pip install -e ".[tune]"`). The tuner explores alpha, depth_multiplier,
embeddings_size, learning_rate, dropout, batch_size, mixup_alpha, optimizer,
weight_decay, grad_clip, use_se,
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
| `--data_path_val` | None | Separate validation root; disables random validation splitting |
| `--classes_file` | None | Ordered one-label-per-line output schema |
| `--max_classes` | None | Use only the N most populated classes |
| `--max_samples` | None | Max files per class |
| `--upsample_ratio` | 0.5 | Minority class upsample ratio |
| `--sample_rate` | 24000 | Audio sample rate (Hz) |
| `--num_mels` | 64 | Number of mel frequency bins |
| `--spec_width` | 256 | Spectrogram width (frames) |
| `--fft_length` | 512 | FFT window length |
| `--chunk_duration` | 3 | Chunk duration (seconds) |
| `--max_duration` | 60 | Max seconds to load per file |
| `--audio_frontend` | hybrid | `librosa`, `hybrid`, `raw`, `mfcc`, or `log_mel` |
| `--mag_scale` | pwl | `pwl`, `pcen`, `db`, or `none` |
| `--embeddings_size` | 256 | Embedding channels before head |
| `--alpha` | 1.0 | Model width scaling |
| `--depth_multiplier` | 1 | Block repeats per stage |
| `--frontend_trainable` | False | Make frontend weights trainable |
| `--mixup_alpha` | 0.2 | Mixup alpha (0 disables) |
| `--mixup_probability` | 0.25 | Fraction of batch to mix |
| `--no_spec_augment` | False | Disable SpecAugment masking (on by default) |
| `--freq_mask_max` | 8 | Max frequency mask width (bins) |
| `--time_mask_max` | 25 | Max time mask width (frames) |
| `--dropout` | 0.5 | Dropout rate before classifier head |
| `--optimizer` | adam | `adam`, `sgd`, or `adamw` |
| `--weight_decay` | 0.0 | Weight decay (adamw only) |
| `--no_se` | False | Disable SE channel attention (on by default) |
| `--se_reduction` | 8 | SE channel reduction factor |
| `--no_inverted_residual` | False | Use plain DS blocks (inverted residuals on by default) |
| `--expansion_factor` | 2 | Expansion factor for inverted residuals |
| `--use_attention_pooling` | False | Use attention pooling instead of GAP |
| `--n_mfcc` | 20 | Number of MFCC coefficients (mfcc frontend only) |
| `--grad_clip` | 1.0 | Max gradient norm for clipping (0 = disabled) |
| `--mixed_precision` | False | Enable FP16 mixed precision training |
| `--resume` | False | Resume training from checkpoint |
| `--seed` | 42 | Random seed |
| `--batch_size` | 32 | Batch size |
| `--num_workers` | 8 | Parallel data loading workers (0 = sequential) |
| `--max_chunks_per_file` | 3 | Max salient chunks per file open (reduces redundant I/O) |
| `--prefetch_batches` | 2 | Loader prefetch depth in batches |
| `--epochs` | 50 | Number of epochs |
| `--learning_rate` | 0.001 | Initial learning rate |
| `--val_split` | 0.2 | Validation split fraction when `--data_path_val` is not supplied |
| `--checkpoint_path` | checkpoints/best_model.keras | Output path (.keras) |
| `--tune` | False | Run Optuna hyperparameter search |
| `--n_trials` | 20 | Number of Optuna trials |
| `--qat` | False | Quantization-aware fine-tuning |
| `--linear_probe` | False | Freeze backbone and train only classifier head |

## Data pipeline

The training pipeline uses a **multiprocessing pool** for parallel data
loading, bypassing the GIL so FLAC decode, resampling, smart-crop, and
spectrogram computation run across separate CPU cores.

When `--max_chunks_per_file` is greater than 1 (default 3), each file open
extracts multiple salient chunks which are buffered in a shuffled in-memory
**reservoir** sized from the sample representation and the loader's memory
budget. This dramatically reduces I/O for
long recordings: a 60 s file decoded once yields 3 usable chunks instead of
re-opening the same file 3 times across epochs.

The reservoir maintains batch diversity by shuffling samples from many
different files before yielding them.

Training also checks host-available memory every 25 batches. It aborts before
available RAM falls below an adaptive reserve (20% of host RAM, bounded between
2 and 12 GiB), leaving the last completed-epoch checkpoint available for a
safer restart. This protects the host from multiprocessing and TensorFlow
memory spikes; it is not a substitute for choosing a conservative worker count
and batch size.

Tune with:

- `--num_workers N` — number of worker processes (default 8, 0 = sequential)
- `--max_chunks_per_file N` — chunks per file open (default 3, 1 = original behavior)
- `--prefetch_batches N` — queued batches (default 2; higher uses more RAM)

## Noise classes

Place audio in folders named `noise`, `silence`, `background`, or `other`
under `data/train/`. These receive all-zero label vectors and help the model
learn to reject non-bird sounds.
