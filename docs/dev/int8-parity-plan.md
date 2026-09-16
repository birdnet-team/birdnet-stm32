# INT8 quality on device: where we stand

Status after the 1.2.0 release (2026-09-16). This page exists so the next
session starts from measurements rather than from guesses.

## The device is no longer the problem

v1.2.0 fixed the three defects that made on-device numbers untrustworthy: the
raw filterbank's NPU miscomputation, the firmware's hybrid STFT, and the NPU
input copy size (see the changelog). What is left is measured:

| check | result |
|---|---|
| `stedgeai validate --mode target` | cos 0.999648, mae 0.000453 (gate: 0.99 / 1/256) |
| `board-test --host_audio_dir`, 25 files | 25/25 same top-1 as the host, 3 flagged |

So the device reproduces the host. **The remaining loss is host-side
quantization**, and that is what the work below targets.

## Where the accuracy goes

Catalog cMAP, 9,767 files, 1.25 s overlap, max pooling — the protocol every
number here uses:

| step | raw (shipped v1.2) | hybrid, same recipe |
|---|---|---|
| float | 0.6595 | 0.6458 |
| after QAT, still float | 0.6343 (−0.025) | — |
| after conversion to INT8 | **0.5895** (−0.045) | **0.6042** |
| total float → INT8 | **−0.070** | **−0.042** |

Two facts to design against:

- **The conversion step costs more than QAT does** (−0.045 against −0.025), so
  work aimed at the QAT schedule is aimed at the smaller half. Longer QAT
  confirms this: 16 epochs instead of 8 bought +0.002.
- **Hybrid loses 40% less to quantization than raw**, at the cost of 90 ms of
  M55 STFT per inference (161 ms per file against 71 ms). It is verified on
  device as of 1.2.0, so it is a live fallback, not a hypothesis.

Do not use C1a's 0.6193 as a target: it was a host-only number from a model
that computes wrong results on the NPU.

## What to try, in order

1. **Attribute the loss per layer before changing anything.** Run the float and
   INT8 graphs on the same chunks and compare activations layer by layer, and
   `stedgeai validate --cut-output-layers N` for the same cut on target. The
   −0.045 conversion loss is currently unattributed; every idea below is a
   guess until it is. Cheapest step, highest information.
2. **Fix the magnitude layer's output range.** On the shipped model, 50/90/99%
   of its values occupy 1/3/14 of the 255 INT8 codes, so typical values get
   ~2 bits. A compressive PWL (`--mag_scale cpwl`) was the obvious fix and
   **failed**: best float of any arm (0.6608), worst INT8 (PTQ 0.4878, QAT
   0.5798). Do not repeat it. Untried: a learned per-band affine after the
   magnitude, more hinges, or splitting the tensor so its ranges quantize
   separately.
3. **Calibrate where the error is, not globally.** Percentile clipping arms
   (p99.9, p99.99) both scored below p100, so global percentile bounds are
   settled — worse. Per-layer bounds placed where step 1 finds the error are
   not.
4. **Start QAT earlier.** Every QAT run so far fine-tunes the final float
   checkpoint. Training with fake quantization from an earlier epoch, or from
   scratch, changes what the float model converges to rather than patching it
   afterwards.
5. **Check whether the frontend can keep more than 8 bits.** The frontend holds
   the widest dynamic range in the graph. Whether `stedgeai` will place int16
   activations on N6 for those ops only is unknown and worth an hour.
6. **Re-test capacity once raw is fixed.** The 512-d embedding (C1a) was chosen
   on float and host INT8 numbers that never survived the device. Whether a
   narrower head quantizes better is open.

## Board parity residuals (low priority)

Three of 25 board files sit outside the 0.05 score tolerance or flip a
detection next to the 0.5 threshold; the largest gap is 0.054, always at
mid-range scores where the sigmoid is steepest. Inputs are identical, so this
is NPU-versus-TFLite requantization rounding. It matters only if a threshold
decision has to be bit-stable.

## Keep results comparable

- **Catalog**: `evaluate --overlap 1.25 --pooling max` on the full 9,767-file
  catalog test. Subset numbers (`--validation_subset`) are biased upward and
  are only comparable within one draw.
- **Operational**: `measure-operational`, 3 seeds x 3,000 files, max pooling.
  Incumbents: v1.1 macro detection 0.5228 / false alarms 0.2736; v1.2 **0.5624
  / 0.2276**.
- **Device**: the on-target gate and `board-test --host_audio_dir` must both
  pass before a number counts as real. See
  [Release Process](release-process.md).
- **Artifacts**: `/data3/ssw_magpie_rt_model/experiments/` holds
  `V12_RAWFIX_C1a` (shipped raw), `V12_HYBRID_C1a`, `V12_CPWL_C1a`; the v1.2
  release inputs are staged under `release_work/V12_RAW_V1_2/`.
