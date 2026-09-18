# INT8 quality on device

Where INT8 quality stands after the work that followed the 1.2.0 release
(September 2026, shipped in 1.3.0), what moved it, and what is settled. Read
this before starting new quantization work: it records the measurements,
including the negative ones, so they are not repeated.

## Summary

- **The device reproduces the host.** Since 1.2.0 the on-target check and the
  board parity test pass for every model below, so every loss on this page is
  host-side quantization.
- **Compress the spectrogram before the first INT8 tensor.** Taking the square
  root of the STFT magnitude on the host / Cortex-M55, before the model
  quantizes its input, cut the float-to-INT8 loss of the `hybrid` frontend from
  −0.042 to −0.013 cMAP. It is the best INT8 model measured, on every metric.
- **The `raw` frontend loses its precision in the filterbank.** Per-band gain
  equalization (`equalize`) and a faithful QAT simulation recover about 40% of
  the loss: catalog INT8 0.5895 → 0.6173 (1.3 release).
- **`raw` stays the release frontend** (the whole pipeline is one NPU model,
  71 ms per file). `hybrid` + `--input_compression sqrt`, quantized
  post-training, is the best INT8 model measured and a verified alternative at
  117 ms per file. 1.3.0 ships both (`_Raw`, `_Hybrid`).
- **The 1.3 raw model is the equalized QAT checkpoint without the later
  simulation fixes**, converted with the 1.3 graph (magnitude without
  MIN/MAX). That conversion alone lifted it from 0.6069 to 0.6173, level with
  the arm that had every fix, and it beats v1.2 at every threshold with fewer
  false alarms, where that arm raised more.

## Results

Catalog cMAP: 9,767 files, 1.25 s overlap, max pooling. All models use the v1.2
USNE architecture (100 classes, `alpha` 1.0, 512-d embedding).

| model | float | INT8 | float → INT8 | ms / file |
|---|---:|---:|---:|---:|
| raw, v1.2 shipped | 0.6595 | 0.5895 | −0.070 | 71 |
| hybrid, v1.2 recipe | 0.6458 | 0.6042 | −0.042 | 113 |
| raw, equalized, QAT | 0.6595 | 0.6069 | −0.053 | 72 |
| raw, equalized, QAT with the simulation fixes | 0.6595 | 0.6161 | −0.043 | 71 |
| **raw, equalized, QAT, 1.3 release conversion** | 0.6595 | **0.6173** | **−0.042** | **71** |
| precomputed mel (`librosa`), `log` input | 0.6145 | 0.6068 | −0.008 | not timed |
| **hybrid, `sqrt` input, 1.3 release conversion** | **0.6624** | **0.6498** | **−0.013** | **117** |

Operational measurement (`measure-operational`, 3 seeds × 3,000 files, max
pooling), macro detection / false-alarm rate:

| model | @0.25 | @0.5 | @0.75 | hard-neg FA @0.5 | top-1 | MRR |
|---|---|---|---|---:|---:|---:|
| raw, v1.2 shipped | 0.598 / 0.331 | 0.562 / 0.229 | 0.481 / 0.133 | 0.237 | 0.602 | 0.701 |
| raw, equalized | 0.617 / 0.304 | 0.576 / 0.219 | 0.487 / 0.125 | 0.169 | 0.626 | 0.723 |
| raw, equalized + simulation fixes | 0.629 / 0.315 | 0.596 / 0.235 | 0.523 / 0.141 | 0.293 | 0.634 | 0.728 |
| **raw, 1.3 release** | **0.630 / 0.301** | **0.591 / 0.218** | **0.502 / 0.127** | **0.192** | **0.638** | **0.732** |
| **hybrid, `sqrt` input, 1.3 release** | **0.655 / 0.290** | **0.614 / 0.212** | **0.527 / 0.132** | **0.146** | **0.659** | **0.755** |

Device checks (`stedgeai validate --mode target`, gate cos ≥ 0.99 and
mae ≤ 1/256; `board-test --host_audio_dir` on 25 files):

| model | cos | mae | board parity | max \|board − host\| |
|---|---:|---:|---|---:|
| raw, v1.2 shipped | 0.999648 | 0.000453 | 25/25, 3 flagged | 0.054 |
| raw, equalized | 0.999682 | 0.000609 | 25/25, 1 flagged | 0.071 |
| raw, equalized + simulation fixes | 0.999806 | 0.000359 | **24/25** | 0.075 |
| **raw, 1.3 release** | 0.999445 | 0.000633 | **25/25, 1 flagged** | 0.043 |
| **hybrid, `sqrt` input, 1.3 release** | 0.998863 | 0.000588 | **25/25, 0 flagged** | **0.032** |

The one parity failure is a score of 0.476 on the board against 0.551 on the host,
across the 0.5 detection threshold. All 25 files keep the same top-1. See
[Open questions](#open-questions).

## What worked

### Compress the input where it is computed

With `hybrid` and precomputed mel, the spectrogram is computed outside the model
(on the host in training, on the Cortex-M55 on device). `--input_compression
sqrt|log` compresses it there, before min-max normalization, so the model's
first INT8 tensor already holds compressed values. Compressing inside the graph
cannot do this, because the linear magnitude is rounded onto the INT8 grid
first. The firmware applies the same compression (`spec_compress`, selected by
`APP_INPUT_COMPRESSION` from the model config) at a cost of 4 ms per file.

- `hybrid` + `sqrt`: +0.017 float and +0.045 INT8 over uncompressed hybrid. QAT
  made it worse at every epoch, so the shipped INT8 model is plain post-training
  quantization.
- Precomputed mel + `log`: the smallest quantization loss of any frontend
  (−0.008), but its float model is 0.048 weaker than `hybrid` + `sqrt`. It
  seems that 256 linear bins with a learned mel mixer beat 64 fixed mel bands.
  QAT did not help here either. It compiles for the N6 (only the input quantize
  and output dequantize run in software). It was not taken to the board because
  it cannot beat `hybrid` + `sqrt`.

In training, chunks are still ranked by activity on the **uncompressed**
spectrogram, so compression changes only what the model sees, not which chunks
a file contributes.

### Raw: find the loss, then fix it

Attribution with TFLite's `QuantizationDebugger`, keeping one group of ops in
float at a time, on the v1.2 raw QAT model (2,513-file validation subset):

| kept in float | cMAP | vs INT8 |
|---|---:|---:|
| nothing (the shipped INT8 model) | 0.6229 | 0 |
| filterbank convolutions and their sums (14 ops) | 0.6432 | +0.020 |
| whole frontend | 0.6470 | +0.024 |
| everything except backbone and head | 0.6535 | +0.031 |
| everything (float) | 0.6610 | +0.038 |

`|x|`, min/max, band smoothing and the PWL each recover ≤ 0.001, and so does the
backbone. The loss is in the linear filterbank tensors, before any compression:
at the filterbank output the median band's typical value spans 0.3 of the 255
INT8 steps, and one band near 500 Hz sets the range for all of them. Three
changes addressed it:

1. **Per-band gain equalization** (function-preserving, absorbed by the
   following BatchNorm). It is exact in float and gave +0.028 after
   post-training quantization and +0.017 on the catalog after QAT. Most of the
   remaining range problem is dynamic range *within* each band, which
   equalization cannot fix.
2. **QAT simulated what the converter does not do.** The fake-quant graph scored
   0.021 above the converted model, which is the same score as keeping the
   filterbank in float. QAT never quantized the partial filterbank convolution
   outputs, which are separate INT8 tensors in the converted graph. The frontend
   now marks each one as a quantization boundary; this is training-only and the
   deployment graph is unchanged. The simulation also ignored TFLite's range
   ties: INT8 `MINIMUM` forces its inputs onto its output's scale, which
   saturated `|re|` and `|im|` at 4.31 against a real 7.08. The magnitude is now
   `b + 0.4a + 0.6 relu(a − b)`, the same function as
   `max(a, b) + 0.4 min(a, b)` but without tying ops. With both fixes and
   `--qat_range_refresh` (ranges recalibrated on the current weights after every
   epoch), simulated and converted cMAP agree within 0.001–0.004.
3. Together: catalog INT8 0.6161 (+0.027 over v1.2).

All three are in the package: `python -m birdnet_stm32 equalize`, and the
simulation fixes plus `--qat_range_refresh` (on by default) in every QAT run.

## Settled: do not retry

| idea | result |
|---|---|
| Compressive PWL inside the graph (`--mag_scale cpwl`, removed in 1.3.0) | Best float of any raw model (0.6608), worst INT8 (post-training 0.4878, QAT 0.5798) |
| Percentile-clipped QAT ranges (`--qat_calibration_percentile`, removed in 1.3.0) | p99.9 and p99.99 both below p100 |
| Longer QAT (16 epochs instead of 8) | +0.002 on the unfixed simulation. With the fixes and frozen BatchNorm, 16 epochs at 5e-5: 0.6123 against 0.6161 |
| Faster QAT (LR 2e-4) | The float model collapses within two epochs (0.685 → 0.588) |
| QAT on a compressed-input model | Worse than post-training quantization at every epoch, for both `sqrt` and `log` |
| INT16 activations on the NPU | Not supported: Neural-ART takes INT8 only, and float ops run in software on the M55 |
| Float filterbank on the M55 | Recovers +0.020, but NPU stage 11 → 254 ms per file (6× the hybrid STFT doing the same job) |
| STFT inside the model | stedgeai 10.2 cannot compile it; see [Spectrogram Input](spectrogram-input.md#why-the-stft-is-not-inside-the-model) |

## Reproducing the models

Everything below uses tracked code and the CLI defaults, which are this recipe
(24 kHz, 2.5 s chunks, `alpha` 1.0, 512-d embedding, 50 epochs at 5e-4; QAT 8
epochs at 2e-5 with range refresh). Paths in angle brackets are yours.

**Raw (release frontend):** train, equalize, QAT, convert.

```bash
python -m birdnet_stm32 train --data_path_train <train> --data_path_val <validation> \
  --classes_file <labels.txt> --upsample_ratio 0.5 --validation_subset 2513 \
  --checkpoint_path <run>/model.keras
python -m birdnet_stm32 equalize --checkpoint_path <run>/model.keras \
  --data_path_train <train> --output_path <run>/model_eq.keras
python -m birdnet_stm32 train --qat --data_path_train <train> --data_path_val <validation> \
  --classes_file <labels.txt> --validation_subset 2513 --checkpoint_path <run>/model_eq.keras
python -m birdnet_stm32 convert --checkpoint_path <run>/model_eq_qat.keras \
  --model_config <run>/model_eq_model_config.json --data_path_train <train>
```

**Hybrid + `sqrt` (best INT8):** train with the spectrogram input compressed,
then convert directly; QAT only makes it worse.

```bash
python -m birdnet_stm32 train --data_path_train <train> --data_path_val <validation> \
  --classes_file <labels.txt> --upsample_ratio 0.5 --validation_subset 2513 \
  --audio_frontend hybrid --input_compression sqrt --checkpoint_path <run>/model.keras
python -m birdnet_stm32 convert --checkpoint_path <run>/model.keras --data_path_train <train>
```

The spectrogram input is defined by `birdnet_stm32.audio.stft` and specified in
[Spectrogram Input](spectrogram-input.md). The firmware computes the same thing,
and native tests (`tests/test_reference_stft.py`, `tests/test_firmware_stft.py`)
keep librosa, the reference and the C code within `1e-3` of each other.
`input_compression` is stored in the model config, so evaluation, conversion
calibration, `board-test` and the generated firmware config
(`gen_app_config.py`) all pick it up without further flags.

A model counts as validated only after the device checks in
[Release Process](release-process.md) pass.

## Open questions

- **Parity residual on equalized raw models.** The two experiment conversions
  of equalized raw models showed NPU-versus-TFLite score gaps of ~0.07 against
  0.054 for v1.2, and one crossed the detection threshold. The 1.3 release
  conversion of the same checkpoint (no MIN/MAX in the graph) measures 0.043,
  so the residual looks tied to the old magnitude graph rather than to
  equalization. Not yet confirmed per layer (`stedgeai validate
  --cut-output-layers`).
- **Truly earlier QAT.** Every QAT run fine-tuned a finished float model.
  Training with fake quantization from the start is untested.
- **Capacity.** The 512-d embedding was chosen before the raw frontend computed
  correctly on the NPU. Whether a narrower head quantizes as well is open.

## Measurement protocol

- **Catalog**: `evaluate --overlap 1.25 --pooling max` on the full 9,767-file
  catalog test set. Numbers on `--validation_subset` are biased upward and only
  comparable within one draw.
- **Operational**: `measure-operational`, 3 seeds × 3,000 files, max pooling,
  all models in one command.
- **Device**: `stedgeai validate --mode target` and
  `board-test --host_audio_dir` must both pass before a number counts. See
  [Release Process](release-process.md).
