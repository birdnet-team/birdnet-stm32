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
the normal label loss, per-output Bernoulli KL divergence, and both mean and
worst-sample per-sample cosine distance from that teacher. The defaults apply
the tail loss to the worst 10% of each batch with 0.75 weight; both values are
configurable. This constrains background and low-confidence probabilities
while directly optimizing the lower tail that the release parity gate measures.

```bash
# Step 1: Normal training
python -m birdnet_stm32 train --data_path_train data/train \
  --epochs 50 --checkpoint_path checkpoints/model.keras

# Step 2: QAT fine-tuning (lower LR, fewer epochs)
python -m birdnet_stm32 train --data_path_train data/train \
  --data_path_val data/validation --classes_file data/labels.txt --qat \
  --checkpoint_path checkpoints/model.keras \
  --qat_calibration_samples 1024 \
  --qat_cosine_tail_fraction 0.10 --qat_cosine_tail_weight 0.75 \
  --epochs 6 --learning_rate 0.00002

# Step 3: Convert the QAT model
python -m birdnet_stm32 convert \
  --checkpoint_path checkpoints/model_qat.keras \
  --model_config checkpoints/model_model_config.json \
  --data_path_train data/train
```

The QAT model is saved as `{name}_qat.keras` alongside the original.

When the input checkpoint came from `--prune`, QAT re-applies the pruning
mask after every step so the sparsity survives; see [Pruning](#pruning).

### Pruning

Use `--prune` to fine-tune a pretrained model while a growing fraction of its
convolution weights is forced to zero. Like `--qat`, it is a separate step that
starts from a converged checkpoint.

!!! warning "Pruning requires a pretrained model"
    Magnitude pruning decides which weights to keep by looking at the weights
    themselves, so it is meaningless on a randomly initialized network. Train
    normally first. The dataset must have the same classes as the pretrained
    model, in the same order.

```bash
# Step 1: normal training
python -m birdnet_stm32 train --data_path_train data/train \
  --data_path_val data/validation --classes_file data/labels.txt \
  --epochs 50 --checkpoint_path checkpoints/model.keras

# Step 2: prune the backbone to 50% and the shipped head to 75%
python -m birdnet_stm32 train --data_path_train data/train \
  --data_path_val data/validation --classes_file data/labels.txt --prune \
  --checkpoint_path checkpoints/model.keras \
  --prune_final_sparsity 0.5 --prune_head_sparsity 0.75 \
  --epochs 12 --learning_rate 0.0002

# Step 3: QAT on the pruned checkpoint (sparsity is preserved automatically)
python -m birdnet_stm32 train --data_path_train data/train \
  --data_path_val data/validation --classes_file data/labels.txt --qat \
  --checkpoint_path checkpoints/model_pruned.keras \
  --epochs 6 --learning_rate 0.00002

# Step 4: convert
python -m birdnet_stm32 convert \
  --checkpoint_path checkpoints/model_pruned_qat.keras \
  --model_config checkpoints/model_pruned_model_config.json \
  --data_path_train data/train
```

The step writes `{name}_pruned.keras`, a matching
`{name}_pruned_model_config.json`, and `{name}_pruned_pruning_report.json` with
the per-layer sparsity breakdown and the accuracy-gate result.

#### What gets pruned

Pruning removes individual weights (unstructured sparsity) from the dense
convolution kernels — the expand, project, and embedding 1×1 convolutions,
which hold roughly 70% of the parameters of a default DS-CNN. Four groups are
exempt because they are small but highly sensitive:

| Exempt | Why |
|---|---|
| Depthwise kernels | 9 weights per channel; nothing is redundant |
| Audio frontend | Mel mixers and Gabor filterbanks are signal-processing filters, not spare capacity |
| Any kernel below `--prune_min_layer_params` (default 1024) | Rounding error in the budget, first place accuracy breaks |

The classifier head is **not** exempt. `convert --split_head` can ship it as
its own artifact (see
[Backbone and classifier split](conversion.md#backbone-and-classifier-split)),
in which case its weights are the entire cost of an over-the-air model update —
and zeroed INT8 weights are the only ones that compress. Pruning it by default
means the head is already small whenever you do export it separately.
`--prune_head_sparsity` gives the head its own target so it can be compressed
harder than the backbone:

```bash
--prune_final_sparsity 0.5 --prune_head_sparsity 0.75
```

The head follows the same cubic ramp and the same accuracy gate; it just aims
at a different endpoint. Pass `--no_prune_head` to leave it dense.

#### How sparsity is reached

Sparsity follows the cubic ramp of Zhu & Gupta: it rises quickly at first,
while the surviving weights still have most of the run to compensate, and
flattens as it approaches the target. The ramp occupies the first
`--prune_ramp_fraction` (default 0.5) of the run; the mask is then frozen and
the remaining epochs fine-tune the surviving weights against a fixed
architecture.

Masks are re-derived from the current weight magnitudes at every update
(`--prune_frequency`, default every 100 steps) and are applied in the forward
pass only. The underlying variables stay dense during the ramp, so a weight
that was masked early can re-enter the network if it recovers. Only the final,
frozen mask is baked into the saved checkpoint.

!!! tip "Leave room after the ramp"
    `--prune_ramp_fraction` must leave at least one epoch after the ramp: the
    post-ramp epochs are the only ones eligible for best-checkpoint selection,
    and they are where the surviving weights recover. If the ramp consumes the
    whole run the step warns and falls back to the final-epoch weights.

`--prune_scope layerwise` (default) gives every prunable layer the same
sparsity. `--prune_scope global` ranks all prunable weights against one
threshold, so layers with genuinely redundant weights absorb more of the
budget; no single layer is taken past 95%. Global scope usually wins above
about 60% sparsity and is the first thing to try when the accuracy gate fails.

#### Protecting accuracy

Four mechanisms keep the pruned model on the unpruned model's decision surface:

1. A frozen copy of the pre-pruning checkpoint acts as a teacher. Pruning
   minimizes the label loss plus per-output Bernoulli KL divergence and both
   mean and worst-sample cosine distance from that teacher — the same
   objective QAT uses, controlled by the `--prune_*` loss weights.
2. BatchNorm layers are frozen, so removing weights cannot shift the
   normalization statistics the rest of the network was trained against.
3. Checkpoint selection and early stopping do not start until the ramp has
   finished. A mid-ramp epoch always scores better and would otherwise win
   with sparsity it never reached.
4. The step ends by scoring the pruned model and the unpruned teacher on the
   same `--prune_eval_samples` (default 1024) held-out samples and **fails**
   if macro ROC-AUC dropped by more than `--prune_max_auc_drop` (default
   0.005). The checkpoint is kept for inspection, but the command exits with
   an error rather than handing you a quietly degraded model.

If the gate fails, lower `--prune_final_sparsity`, raise `--epochs`, or switch
to `--prune_scope global`.

#### What pruning buys on STM32N6

Be clear about the trade before spending a training run on it:

| | Effect |
|---|---|
| `.tflite` size | **Unchanged.** TFLite stores INT8 weights densely; a zero costs the same byte as any other value. |
| Compressed / OTA size | **Smaller** — around 20% off the gzipped model at 50% sparsity, and more off a hard-pruned classifier head. |
| NPU latency | **Unchanged.** ST Neural-ART executes dense kernels; it does not skip zero weights. |
| Robustness | Slightly better, in the way any capacity reduction with distillation regularizes. |

The compressed-size row is the one that matters for a satellite-updated
classifier head: pruning is what turns the head's INT8 weights into bytes gzip
can actually collapse.

Unstructured pruning is therefore a *storage and regularization* tool here, not
a latency tool. To make the model faster on the NPU, reduce `--alpha` or
`--depth_multiplier` and retrain; that removes whole channels, which the NPU
actually skips.

#### Pruning and QAT together

Run pruning first, then QAT. QAT detects the pruned zeros in the checkpoint it
loads and re-applies them after every training step, so quantization-aware
fine-tuning cannot silently refill the pruned weights. The `[QAT] Preserving
pruning masks` line confirms it. Pass `--no_qat_preserve_sparsity` to disable
that behaviour.

Symmetric per-channel INT8 quantization maps zero to zero exactly, so the
sparsity survives conversion unchanged.

### Linear probing

Use `--linear_probe` to freeze a pretrained backbone and train only a new
classification head on your custom species dataset. This is useful when you
have a pretrained model (e.g. a large BirdNET checkpoint) and want to adapt
it to a different set of species with limited data.

```bash
python -m birdnet_stm32 train --data_path_train data/my_species \
  --linear_probe --checkpoint_path checkpoints/pretrained.keras \
  --data_path_val data/my_species_val \
  --classes_file data/my_species/labels.txt \
  --epochs 20 --learning_rate 0.001
```

The probe model is saved as `{name}_probe.keras` with a new labels file and
`{name}_probe_model_config.json`.

Pass `--classes_file` whenever the head will be shipped. The head's output order
*is* its labels file, and without an explicit schema that order comes from a
directory listing, which is not a contract. `--data_path_val` gives the probe a
fixed validation root instead of a random slice of the training set, so repeated
runs are comparable.

Probing a checkpoint from an earlier compression step needs `--model_config`:
a QAT checkpoint writes no config of its own, since its architecture is the base
model's.

```bash
python -m birdnet_stm32 train --data_path_train data/my_species \
  --linear_probe --checkpoint_path checkpoints/pretrained_qat.keras \
  --model_config checkpoints/pretrained_model_config.json \
  --classes_file data/my_species/labels.txt --epochs 20
```

To ship the resulting head on its own, convert it against the backbone already
on the device — see
[Updating the head against a flashed backbone](conversion.md#updating-the-head-against-a-flashed-backbone).

### Learning rate

A two-epoch linear warmup reaches `--learning_rate` (default 0.001), followed
by cosine decay to near-zero over `--epochs` (default 50). Best-checkpoint
selection and early stopping monitor validation ROC-AUC with patience 10 for
standard training. QAT instead minimizes validation teacher KL, configurable
with `--qat_checkpoint_monitor`; the paired catalog evaluation remains the
release-deciding accuracy gate.
Pruning keeps the ROC-AUC monitor but ignores every epoch before its sparsity
ramp finishes.

### Hyperparameter tuning with Optuna

Use `--tune` to run an automated hyperparameter search using Optuna (requires
`pip install -e ".[tune]"`). The tuner explores alpha, depth_multiplier,
embeddings_size, learning_rate, dropout, batch_size, mixup_alpha, optimizer,
weight_decay, grad_clip, and use_attention_pooling.
It maximizes `val_roc_auc` with MedianPruner.

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
| `--qat_calibration_samples` | 1024 | Exact stratified samples used for QAT and conversion calibration |
| `--qat_distillation_weight` | 1.0 | Frozen-teacher Bernoulli-KL weight |
| `--qat_cosine_weight` | 0.10 | Mean teacher/student cosine-loss weight |
| `--qat_cosine_tail_weight` | 0.75 | Worst-sample cosine-loss weight |
| `--qat_cosine_tail_fraction` | 0.10 | Fraction of each batch included in the worst-sample loss |
| `--no_qat_preserve_sparsity` | False | Let QAT refill weights a previous `--prune` run zeroed |
| `--prune` | False | Gradual magnitude pruning |
| `--prune_final_sparsity` | 0.5 | Target fraction of prunable weights zeroed |
| `--prune_scope` | layerwise | `layerwise` or `global` sparsity allocation |
| `--prune_ramp_fraction` | 0.5 | Fraction of the run spent ramping up to the target |
| `--prune_frequency` | 100 | Steps between mask recomputations during the ramp |
| `--prune_min_layer_params` | 1024 | Smallest kernel (in weights) eligible for pruning |
| `--no_prune_head` | False | Leave the classifier head dense |
| `--prune_head_sparsity` | -1 | Separate target for the classifier head (-1 follows `--prune_final_sparsity`) |
| `--prune_max_auc_drop` | 0.005 | Largest tolerated macro ROC-AUC regression |
| `--prune_eval_samples` | 1024 | Validation samples scored by the accuracy gate |
| `--prune_distillation_weight` | 1.0 | Frozen-teacher Bernoulli-KL weight |
| `--prune_cosine_weight` | 0.10 | Mean teacher/student cosine-loss weight |
| `--prune_cosine_tail_weight` | 0.75 | Worst-sample cosine-loss weight |
| `--prune_cosine_tail_fraction` | 0.10 | Fraction of each batch included in the worst-sample loss |
| `--linear_probe` | False | Freeze backbone and train only classifier head |
| `--model_config` | *(inferred)* | Architecture config for `--qat`, `--prune`, `--linear_probe`; required when the checkpoint has no sibling config |
| `--qat_checkpoint_monitor` | `val_distillation_kl` | Validation metric selecting the kept QAT epoch |

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
