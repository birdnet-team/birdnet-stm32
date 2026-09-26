# Spectrogram Input Specification

The `hybrid` and `librosa` (precomputed mel) frontends do not take audio. They
take a spectrogram computed **outside** the model: by the host during training
and evaluation, and by the Cortex-M55 in the firmware. Whoever deploys one of
these models on another device has to compute that spectrogram exactly as it was
computed in training, or the model runs on an input it has never seen. Nothing
in the model file, the compiler or `stedgeai validate` catches a mismatch: they
all start from a spectrogram you already have.

This page specifies that computation completely. It has three sources of truth
that are tested against each other:

| Implementation | File | Checked by |
|---|---|---|
| Python reference (host) | `birdnet_stm32/audio/stft.py` | `tests/test_reference_stft.py` (against librosa) |
| C firmware (Cortex-M55) | `firmware/Src/audio_stft.c` (CMSIS-DSP FFT), `audio_mel.c` | `tests/test_firmware_stft.py` (compiled natively, against the reference) |
| This page | | the reference implementation below is copied from it |

Deviating from any step below changes the model input. Tolerance for a correct
re-implementation is **max absolute error below `1e-3`** on the normalized
output, a quarter of one INT8 input step.

## Parameters

Everything comes from the model's `*_model_config.json`:

| Key | Meaning | v1.2 USNE value |
|---|---|---|
| `sample_rate` | Sample rate of the chunk (Hz) | 24000 |
| `chunk_duration` | Chunk length (s) | 2.5 |
| `fft_length` | FFT size `n_fft` | 512 |
| `spec_width` | Frames per chunk `W` | 256 |
| `audio_frontend` | `hybrid` (linear bins) or `librosa` (mel bands) | |
| `num_mels` | Mel bands (`librosa` only) | 64 |
| `input_compression` | `none`, `sqrt` or `log` (absent means `none`) | |

Derived:

| Symbol | Definition | Value |
|---|---|---|
| `N` | `sample_rate * chunk_duration` samples | 60000 |
| `hop` | `N // W` (integer division) | 234 |
| `B` | `n_fft // 2` frequency bins | 256 |
| rows | `B` for `hybrid`, `num_mels` for `librosa` | 256 / 64 |

## Algorithm

Input: one chunk `x[0 .. N-1]`, mono, float32. Its level does not matter:
step 8 normalizes the chunk, so peak-normalizing the audio first (as the host
does) changes nothing beyond round-off.

### 1. Pad

Prepend and append `n_fft // 2` **zeros**:

```
p[i] = x[i - n_fft/2]   for n_fft/2 <= i < N + n_fft/2
p[i] = 0                otherwise                      (length N + n_fft)
```

Zeros, not reflection. This centers frame `t` on sample `t * hop`.

### 2. Frame

Frame `t` is `p[t * hop .. t * hop + n_fft - 1]` for `t = 0 .. W-1`.

The padded signal holds `1 + N // hop` complete frames (257 for the v1.2 values).
Only the **first** `W` are used; the last one is dropped.

### 3. Window

Multiply each frame by the **periodic** Hann window:

```
w[n] = 0.5 - 0.5 * cos(2 * pi * n / n_fft)      n = 0 .. n_fft-1
```

Divide by `n_fft`, not `n_fft - 1`. The symmetric window is a different input.

### 4. Magnitude spectrum

```
X[t, k] = | sum_n  w[n] * p[t*hop + n] * exp(-2j * pi * k * n / n_fft) |      k = 0 .. B-1
```

This is the modulus of the real FFT (`rfft`) of the windowed frame, **not** the
power (no squaring) and with no scaling by `n_fft` or the window sum. Keep bins
`0 .. B-1`: the Nyquist bin `k = B` is dropped.

Result: `S[k, t] = X[t, k]`, shape `[B, W]`.

### 5. Mel projection (`librosa` frontend only)

`M[m, t] = sum_k F[m, k] * S[k, t]` with the filterbank `F` of shape
`[num_mels, B]` defined below. The `hybrid` frontend skips this step: its model
learns the projection itself.

**Mel scale** (Slaney, linear below 1 kHz):

```
hz_to_mel(f) = f / (200/3)                                  f <  1000
             = 15 + ln(f / 1000) / (ln(6.4) / 27)           f >= 1000

mel_to_hz(m) = m * (200/3)                                  m <  15
             = 1000 * exp((ln(6.4) / 27) * (m - 15))        m >= 15
```

This is not the HTK formula `2595 * log10(1 + f/700)`.

**Band edges:** `num_mels + 2` frequencies equally spaced in mel between
`fmin = 150 Hz` and `fmax = sample_rate / 2`:

```
edge[i] = mel_to_hz( hz_to_mel(150) + i * (hz_to_mel(fmax) - hz_to_mel(150)) / (num_mels + 1) )
```

**Filters:** with bin frequency `f_k = k * sample_rate / n_fft`,

```
rising  = (f_k - edge[m])   / (edge[m+1] - edge[m])
falling = (edge[m+2] - f_k) / (edge[m+2] - edge[m+1])
F[m, k] = max(0, min(rising, falling)) * 2 / (edge[m+2] - edge[m])
```

The factor `2 / (edge[m+2] - edge[m])` (Slaney area normalization) is part of
the definition. Because `fmax` is Nyquist, every filter is exactly zero at the
dropped Nyquist bin, so dropping it loses nothing.

### 6. Compression (`input_compression`)

Applied to the whole `[rows, W]` array:

| Value | Operation |
|---|---|
| `none` | unchanged |
| `sqrt` | `S = sqrt(S)` elementwise |
| `log` | `S = ln(max(S, floor))` with `floor = max(max(S), 1e-10) * 10^(-80/20)` |

The `log` floor is relative to **this chunk's** peak: 80 dB below it. The
logarithm is natural, not `log10` or dB. After step 7 the base does not matter,
but the floor does.

### 7. Normalize

Over the whole array, a single minimum and maximum:

```
S = (S - min(S)) / (max(S) - min(S) + 1e-10)
```

The result lies in `[0, 1]`.

### 8. Hand to the model

The model input is float32 `[1, rows, W, 1]` (NHWC). In C row-major memory, the
value for row `r` and frame `t` sits at index `r * W + t`. The model quantizes
this input to INT8 internally; pass float32.

## Reference implementation

This is the complete computation, copy-ready: numpy only, float32 as on the
device. `birdnet_stm32.audio.stft.spectrogram_input` is the same code, split into
one function per step.

```python
import numpy as np

def spectrogram_input(x, sample_rate, n_fft, spec_width, n_mels=0, compression="none"):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    hop = len(x) // spec_width
    bins = n_fft // 2

    # 1-3: zero-pad, frame (first spec_width frames), periodic Hann
    padded = np.pad(x, n_fft // 2)
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop][:spec_width]
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float32)

    # 4: magnitude, Nyquist dropped -> [bins, frames]
    S = np.abs(np.fft.rfft(frames * window, axis=1))[:, :bins].T.astype(np.float32)

    # 5: Slaney mel filterbank (librosa frontend only)
    if n_mels > 0:
        f_sp, logstep = 200 / 3, np.log(6.4) / 27
        hz_to_mel = lambda f: np.where(f >= 1000, 15 + np.log(np.maximum(f, 1000) / 1000) / logstep, f / f_sp)
        mel_to_hz = lambda m: np.where(m >= 15, 1000 * np.exp(logstep * (np.maximum(m, 15) - 15)), m * f_sp)
        edges = mel_to_hz(np.linspace(hz_to_mel(150.0), hz_to_mel(sample_rate / 2), n_mels + 2))
        f = np.arange(bins) * sample_rate / n_fft
        lo, mid, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
        F = np.maximum(0, np.minimum((f - lo) / (mid - lo), (hi - f) / (hi - mid))) * (2 / (hi - lo))
        S = F.astype(np.float32) @ S

    # 6: compression
    if compression == "sqrt":
        S = np.sqrt(S)
    elif compression == "log":
        S = np.log(np.maximum(S, np.float32(max(S.max(), 1e-10) * 10 ** (-80 / 20))))

    # 7: normalize the whole chunk
    return ((S - S.min()) / (S.max() - S.min() + np.float32(1e-10))).astype(np.float32)
```

## C implementation (firmware)

The firmware computes the same thing in float32 on the Cortex-M55. Main loop in
`firmware/Src/main.c`:

```c
stft_magnitude(audio, APP_CHUNK_SAMPLES, APP_FFT_LENGTH, APP_HOP_LENGTH,
               APP_SPEC_WIDTH, spec);                        /* steps 1-4 */
/* librosa frontend only: */
mel_filterbank(spec, APP_FFT_BINS, APP_SPEC_WIDTH, APP_NUM_MELS, mel);  /* step 5 */
spec_compress(buf, rows * APP_SPEC_WIDTH, APP_INPUT_COMPRESSION);       /* step 6 */
spec_minmax_normalize(buf, rows * APP_SPEC_WIDTH);                      /* step 7 */
```

| Function | File | Notes |
|---|---|---|
| `stft_magnitude` | `audio_stft.c` | Centered framing with zero padding, periodic Hann, `sqrt(re^2 + im^2)` |
| `arm_rfft_fast_f32`, `arm_cmplx_mag_f32` | `Drivers/CMSIS-DSP/` | CMSIS-DSP real FFT (Helium on the M55), packed output: `[0]` DC, `[1]` Nyquist, then `(re, im)` pairs; and its vectorized magnitude |
| `mel_init`, `mel_filterbank` | `audio_mel.c` | Builds `F` once at start-up (`fmin` 150 Hz, `fmax` Nyquist), then a sparse matrix product |
| `spec_compress` | `audio_stft.c` | `SPEC_COMPRESS_SQRT` / `SPEC_COMPRESS_LOG` (80 dB floor, natural log) |
| `spec_minmax_normalize` | `audio_stft.c` | `(S - min) / (max - min + 1e-10)` |

`APP_HOP_LENGTH`, `APP_INPUT_COMPRESSION` and the other constants come from
`app_config.h`, which `firmware/gen_app_config.py` writes from the model config.
Any FFT library works in place of CMSIS-DSP if it produces the same unscaled
complex spectrum.

Measured on the STM32N6570-DK at the default 400 MHz, for the released hybrid
geometry (384 frames of a 2.5 s chunk): the whole input, steps 1–4 with `sqrt`
compression and min-max normalization, takes **33 ms**. The mel projection has
not been timed on the board.

## Pitfalls

Each of these produces a plausible-looking spectrogram that the model was not
trained on.

| Mistake | Effect |
|---|---|
| Frames start at `t * hop` instead of centered (no padding) | Every frame shifted by half a window; measured cosine 0.32 against the host input |
| Reflect padding instead of zeros | Edge frames differ (librosa before 0.10 defaulted to reflect) |
| Symmetric Hann (`n_fft - 1`) | Different window; up to ~0.6% per sample |
| Keeping all 257 frames, or dropping the first instead of the last | Wrong width, or every column shifted by one hop |
| Keeping the Nyquist bin | `hybrid` expects exactly `n_fft // 2` rows |
| Power spectrum (`|X|^2`) or `|X| / n_fft` | Different dynamic range; normalization does not undo squaring |
| `hop` rounded or computed from seconds | `hop` is `N // W`, exactly |
| HTK mel formula, or `norm=None` | Different band edges or band gains |
| `fmin` 0 or `fmax` other than Nyquist | Different band edges |
| `log10`, dB, or a fixed absolute floor | The floor must be 80 dB below the chunk's own peak |
| Normalizing per row, per frame, or with a running min/max | Min and max are taken over the whole chunk |
| Feeding int8 or uint8 | The model takes float32 and quantizes internally |
| Copying fewer bytes than the input tensor holds | Measured before 1.2.0: the firmware copied 256 of 65,536 values and the model saw garbage |

## Checking a re-implementation

Generate a deterministic chunk, run both implementations, and compare:

```python
import numpy as np
from birdnet_stm32.audio.stft import spectrogram_input

sr, n = 24000, 60000
t = np.arange(n) / sr
rng = np.random.default_rng(0)
x = (0.05 * rng.standard_normal(n) + 0.5 * np.sin(2 * np.pi * (3000 * t + 400 * t**2))).astype(np.float32)
x.tofile("chunk.f32")                       # feed this to your implementation

reference = spectrogram_input(x, sr, 512, 256, n_mels=64, compression="log")
yours = np.fromfile("yours.f32", np.float32).reshape(reference.shape)
print("max abs error", np.abs(reference - yours).max())   # must be < 1e-3
```

Check every combination you deploy (frontend × compression), and at least one
chunk with silence at an edge: the padding and the log floor only show up there.

## Why the STFT is not inside the model

Tested on stedgeai 10.2 ([INT8 quality](int8-parity-plan.md)): a `.tflite` with the STFT
in-graph cannot be compiled for the N6. `tf.signal.stft` lowers to `RFFT2D` and
`COMPLEX_ABS`, which stedgeai does not implement. Written instead as a fixed
convolution bank (the only FFT-free form), the STFT is exact but needs 67M
float multiply-adds per chunk, about 1.1 s on the M55 against 42 ms for the
firmware FFT, and its float tensors exceed the M55's 1 MB RAM pool. The
spectrogram therefore stays a documented pre-processing step.
