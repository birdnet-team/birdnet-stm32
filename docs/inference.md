# Reference Implementation

A released bundle is a TFLite model plus the constants needed to feed it. This
page is the contract: everything between an audio file and a detection, which
side runs each step, and how to prove a port computes the same thing.

The contract is implemented once, in
[`reference/birdnet_tiny_reference.py`](https://github.com/birdnet-team/birdnet-stm32/blob/master/reference/birdnet_tiny_reference.py):
one file, NumPy + SoundFile + TensorFlow Lite, nothing from this package, so it
can be read top to bottom and ported. The
[`reference/`](https://github.com/birdnet-team/birdnet-stm32/tree/master/reference)
directory also holds a synthetic test recording and, for each released bundle,
the value of every intermediate stage on it.

```bash
# Run a bundle on a recording; --explain prints each stage's shape and range
python reference/birdnet_tiny_reference.py --bundle <bundle-dir> --audio recording.wav --explain

# Check an implementation stage by stage against the recorded test vectors
python reference/birdnet_tiny_reference.py --bundle <bundle-dir> \
    --check-vectors reference/vectors/<bundle>.json

# Keep every intermediate as .npy, to compare with your own
python reference/birdnet_tiny_reference.py --bundle <bundle-dir> --audio recording.wav --dump stages/
```

A bundle directory is the unpacked release zip; the script reads its
`*_INT8.tflite`, `*_model_config.json` and `*_labels.txt`.

## The pipeline

| # | Step | Host | STM32N6 |
|---|---|---|---|
| 1 | Read audio as mono float32 at the model's sample rate | resample if needed | PCM16 at 24 kHz already; scale by 1/32768 |
| 2 | Cut into windows: `chunk_duration` long, half that as hop | ✔ | firmware |
| 3 | `raw`: divide each window by its peak | ✔ | firmware |
| 3 | `hybrid`: magnitude STFT, compress, min-max normalize | ✔ | Cortex-M55, CMSIS-DSP |
| 4 | Run the model, one window at a time | ✔ | NPU |
| 5 | Apply the sigmoid if the bundle emits logits | ✔ | firmware, or use logit thresholds |
| 6 | Pool windows into per-file scores (max) and threshold | ✔ | firmware |

Every constant comes from the bundle's `*_model_config.json`: `sample_rate`,
`chunk_duration`, `audio_frontend`, `fft_length`, `spec_width`,
`input_compression`, and `output_activation`. Class order comes from
`*_labels.txt`. Never hard-code these; a bundle is free to change them.

The two frontends differ only in step 3. A `raw` model computes its own
spectrogram inside the network, with a learned filterbank that runs on the NPU,
so the host only normalizes the waveform. A `hybrid` model needs a magnitude
STFT computed outside the network.

## Windows

Windows are `chunk_duration` seconds long (2.5 s in every released bundle) and
step by half that (1.25 s), so each moment of audio is scored twice. The overlap
matters: a call that straddles a window boundary is cut in half in both windows
unless they overlap, and the released models are evaluated this way.

The **last window is right-aligned** to the end of the recording rather than
zero-padded, so the final seconds are scored at full weight. A recording shorter
than one window is zero-padded once, at the end. Six seconds of audio give
windows starting at 0, 1.25, 2.5 and 3.5 s.

## Normalization, and why it differs per frontend

**`raw` needs it.** Each window is divided by its own largest absolute sample,
`x / (max|x| + 1e-6)`, so the loudest sample becomes 1.0. Per window, not per
file: the models are trained that way, and it makes the input independent of
recording gain.

**`hybrid` does not.** Its spectrogram is min-max normalized to [0, 1] as the
last step of step 3, and scaling the audio scales every magnitude by the same
factor, which that normalization divides out. Feeding a peak-normalized window
to a hybrid bundle gives the same answer; it is simply unnecessary.

## The hybrid spectrogram

Specified in detail in [Spectrogram Input](dev/spectrogram-input.md); the
firmware's `firmware/Src/audio_stft.c` computes it on the Cortex-M55. Six
details, each of which changes the model's input if it is wrong:

1. **The hop follows from the frame count**: `hop = samples // spec_width`. For
   2.5 s at 24 kHz into 384 frames, that is 156 samples.
2. **The STFT is centered.** Zero-pad the window by `fft_length / 2` on both
   sides first, so frame *i* is centered on sample *i · hop*, not started there.
3. **A periodic Hann window**, `0.5 − 0.5 cos(2πn / N)`: SciPy's and librosa's
   `sym=False`.
4. **Drop the Nyquist bin.** An `fft_length` of 512 gives 256 rows. The FFT is
   unscaled.
5. **Compress** per `input_compression`: `sqrt` in every released hybrid bundle.
6. **Min-max normalize** the window's whole spectrogram to [0, 1]:
   `(S − min) / (max − min + 1e-10)`.

The result is `[fft_length // 2, spec_width, 1]` float32, frequency-major —
`[256, 384, 1]` for the released hybrid bundles.

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

This project's firmware applies the sigmoid (one `expf` per class, against 33 ms
of STFT on the hybrid model), so its score threshold and its reported percentages
are probabilities exactly as in earlier releases. A re-implementation is free to
choose either.

Why the released models emit logits: an INT8 probability sits on a 1/256 grid,
which floors every score below about 0.002 and ties the rest, costing 0.021
catalog cMAP on raw and 0.035 on hybrid. A logit grid is uniform in logits
instead, so it resolves small scores far better — and mid-range ones slightly
worse. At an output step of 0.122, one step spans 0.031 in probability at
p = 0.5 against the old 0.004, so a score near 0.5 is coarser than it used to
be. That is the trade: ranking quality, which is what detection is, for
resolution in the middle of a range where nothing is decided.

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

`reference/vectors/test_signal.wav` is a 6 s synthetic recording (24 kHz, mono,
PCM16) with a sparse whistle, a frequency sweep and one loud transient. For each
released bundle, `reference/vectors/<bundle>.json` records every window's stages:

| Key | Stage |
|---|---|
| `window` | the cut audio (step 2) |
| `stft_magnitude` | hybrid only: the linear magnitude STFT (steps 3.1–3.4) |
| `compressed` | hybrid only: after compression (step 3.5) |
| `model_input` | the tensor the model is fed |
| `model_output` | the model's raw outputs (logits from 1.5 on) |

Each array is summarized as its shape, sum, minimum, maximum and first eight
values in row-major order — enough to find the first stage where a port
diverges, in any language. `pooled` holds the recording's pooled probabilities.

A faithful port agrees to about 1e-4 relative on every input stage (float32, a
different FFT library) and within one or two INT8 output steps on the model
output; detections at 0.5 match exactly. `--check-vectors` applies exactly these
tolerances. Typical mistakes, in the order they are usually made: a non-centered
STFT, a symmetric Hann window, keeping the Nyquist bin, forgetting the min-max
normalization, and normalizing a whole file instead of each window.

On real field audio, measured between the reference script and this package's
own evaluation path over 60 windows of a 10-minute recording:

| | raw | hybrid |
|---|---|---|
| median score difference | 0.00000 | 0.00000 |
| mean | 0.00055 | 0.00013 |
| worst single window | 0.051 | 0.035 |
| same detections at 0.5 | 100% | 99.98% |

That is the tolerance the project accepts between host and board
(`board-test` flags differences above 0.05). The test suite keeps the pieces in
step: `tests/test_reference_implementation.py` checks that the reference and the
training package build identical model inputs, and `tests/test_firmware_stft.py`
compiles the firmware's STFT natively and checks it against the host.
