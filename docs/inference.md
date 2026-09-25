# Running Inference

A released bundle is a TFLite model plus the constants needed to feed it. This
page is the contract: everything between an audio file and a detection, and
which side runs each step. [`examples/reference_inference.py`](https://github.com/birdnet-team/birdnet-stm32/blob/master/examples/reference_inference.py)
implements exactly this in one file, with NumPy and TensorFlow Lite and nothing
from this package, so it can be read as a specification and ported to C.

```bash
python examples/reference_inference.py \
    --bundle release/BirdNET_Tiny_N6_USNE_90_V1.4_Raw \
    --audio recording.wav --explain
```

`--explain` prints every step with its shapes and value ranges, which is the
fastest way to find where a re-implementation diverges.

## The pipeline

| # | Step | Host | Device |
|---|---|---|---|
| 1 | Read audio as mono float32 at the model's sample rate | resample if needed | SD read; PCM16 at 24 kHz already, scale by 1/32768 |
| 2 | Cut into windows: `chunk_duration` long, half that as hop | ✔ | firmware |
| 3 | Normalize each window by its peak (**`raw` only**) | ✔ | firmware |
| 4 | `hybrid` only: magnitude STFT of the window | ✔ | Cortex-M55 |
| 5 | Run the model, one window at a time | ✔ | NPU |
| 6 | Apply the sigmoid if the bundle emits logits | ✔ | firmware, or use logit thresholds |
| 7 | Pool windows into per-file scores (max) and threshold | ✔ | firmware |

Every constant comes from the bundle's `*_model_config.json`: `sample_rate`,
`chunk_duration`, `audio_frontend`, `fft_length`, `spec_width`,
`input_compression`, and `output_activation`. Class order comes from
`*_labels.txt`. Never hard-code these; a bundle is free to change them.

## Windows

Windows are `chunk_duration` seconds long (2.5 s in every released bundle) and
step by half that (1.25 s), so each moment of audio is scored twice. The overlap
matters: a call that straddles a window boundary is cut in half in both windows
unless they overlap, and the released models are evaluated this way.

The **last window is right-aligned** to the end of the recording rather than
zero-padded, so the final seconds are scored at full weight. A recording shorter
than one window is zero-padded once, at the end.

## Normalization, and why it differs per frontend

**`raw` needs it.** Each window is divided by its own largest absolute sample,
so the loudest sample becomes 1.0. Per window, not per file: the models are
trained that way, and it makes the input independent of recording gain.

**`hybrid` does not.** Its spectrogram is min-max normalized to [0, 1] as the
last step of step 4, and scaling the audio scales every magnitude by the same
factor, which that normalization divides out. Feeding a peak-normalized window
to a hybrid bundle gives the same answer; it is simply unnecessary.

## The hybrid STFT

Specified in detail in [Spectrogram Input](dev/spectrogram-input.md), which the
firmware reproduces. Four details are easy to get wrong:

1. **The STFT is centered.** Zero-pad the window by `fft_length / 2` on both
   sides first, so frame *i* is centered on sample *i · hop*, not started there.
2. **The hop follows from the frame count**: `hop = samples // spec_width`. For
   2.5 s at 24 kHz into 384 frames, that is 156 samples.
3. **A periodic Hann window**, the one SciPy and librosa call `sym=False`.
4. **Drop the Nyquist bin**, then **compress** (`input_compression`, `sqrt` in
   the released hybrid bundle) and **min-max normalize to [0, 1]**. An
   `fft_length` of 512 gives 256 rows.

The result is `[fft_length // 2, spec_width, 1]` float32 — `[256, 384, 1]` for
the 1.4 hybrid bundle.

## Model output

The INT8 models take and return **float32**; quantization is internal. Output is
one value per class, in the order of `*_labels.txt`.

Check `output_activation` in the config. **Every model from 1.5 on says
`logit`**; 1.0 through 1.4 return probabilities.

- **`sigmoid`** (or absent): the values are probabilities in [0, 1]. Use them.
- **`logit`**: the values are logits. Apply `1 / (1 + exp(-x))` for
  probabilities — 100 values per window, negligible anywhere. Firmware can skip
  it entirely and compare logits against `log(t / (1 - t))`: thresholding is
  monotonic, so this is exact and free. A threshold of 0.5 becomes 0.0, and 0.25
  becomes −1.0986.

This project's firmware applies the sigmoid (one `expf` per class, against ~69 ms
of STFT on the hybrid model), so its score threshold and its reported percentages
are probabilities exactly as in earlier releases. A re-implementation is free to
choose either.

Why the released models emit logits: an INT8 probability sits on a 1/256 grid,
which floors every score below about 0.002 and ties the rest, costing 0.021
catalog cMAP on raw and 0.035 on hybrid. A logit grid is uniform in logits
instead, so it resolves small scores far better — and mid-range ones slightly
worse. At this release's output step of 0.122, one step spans 0.031 in
probability at p = 0.5 against the old 0.004, so a score near 0.5 is coarser than
it used to be. That is the trade: ranking quality, which is what detection is,
for resolution in the middle of a range where nothing is decided.

## From windows to detections

Pool each class over the windows of a recording with **max**, matching how these
models are evaluated. Averaging punishes a species that calls once in a long
recording. Then apply the detection threshold.

A threshold of 0.5 is the reported operating point, but it is a choice, not a
property of the model. The released models are deliberately conservative on
quiet field recordings; 0.25 finds considerably more at some cost in precision.
The model card for each bundle carries the measured detection and false-alarm
rates at 0.25, 0.5 and 0.75.

## Checking a re-implementation

Compare against `examples/reference_inference.py` on the same file, window by
window. Expect agreement to within a few INT8 steps rather than exactly: the
last activation grid is 1/256, and small input differences (the order in which
normalizations round, a different FFT library) move a score by a step or two.

Measured between the reference script and this package's own evaluation path,
over 60 windows of a 10-minute field recording:

| | raw | hybrid |
|---|---|---|
| median score difference | 0.00000 | 0.00000 |
| mean | 0.00055 | 0.00013 |
| worst single window | 0.051 | 0.035 |
| same detections at 0.5 | 100% | 99.98% |

That is the same tolerance the project accepts between host and board
(`board-test` flags differences above 0.05). If your implementation agrees this
closely, it is correct; if whole classes disagree, look at step 4 first — a
non-centered STFT or a missing min-max normalization is the usual cause.
