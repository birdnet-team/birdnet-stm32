#!/usr/bin/env python3
"""Reference inference for a BirdNET Tiny N6 bundle, in one readable file.

This is the specification of what a host program or a firmware has to do,
written so it can be re-implemented in C. It uses NumPy and SoundFile for I/O
and TensorFlow Lite to run the model; nothing from ``birdnet_stm32``. Every
constant it needs comes from the bundle's ``*_model_config.json``.

    python examples/reference_inference.py \\
        --bundle release/BirdNET_Tiny_N6_USNE_90_V1.4_Raw \\
        --audio recording.wav

Add ``--explain`` to print each step with its shapes and value ranges, which is
the quickest way to check a re-implementation stage by stage.

The pipeline, and where each step runs on the device:

    1. read + resample to 24 kHz mono            firmware: SD read (PCM16 already 24 kHz)
    2. cut into 2.5 s windows, 1.25 s hop        firmware
    3. normalize each window by its peak         firmware (raw only; see step 4)
    4. raw:    feed the waveform                 NPU
       hybrid: STFT on the CPU, then feed it     Cortex-M55, then NPU
    5. run the model                             NPU
    6. per-window scores -> per-file scores      firmware (max over windows)
    7. threshold into detections                 firmware

The two frontends differ only in step 4. `raw` hands the waveform to the model
and the whole pipeline runs on the NPU. `hybrid` computes a magnitude STFT on
the CPU first, which is more accurate after quantization but costs M55 time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter
except ImportError:  # a bundle runs fine on the small runtime too
    from tflite_runtime.interpreter import Interpreter  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# 1. Audio
# ---------------------------------------------------------------------------


def read_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Read a file as mono float32 at *sample_rate*, in [-1, 1].

    The recorder already writes 24 kHz mono PCM16, so firmware skips the
    resampling and the downmix and only scales the int16 samples by 1/32768.
    """
    audio, file_rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if file_rate != sample_rate:
        from math import gcd

        from scipy.signal import resample_poly

        divisor = gcd(file_rate, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, file_rate // divisor).astype(np.float32)
    return audio


def windows(audio: np.ndarray, sample_rate: int, window_s: float, hop_s: float) -> np.ndarray:
    """Cut the recording into overlapping windows.

    Windows step by *hop_s*. The last window is right-aligned to the end of the
    recording rather than zero-padded, so the final seconds are scored at full
    weight. A recording shorter than one window is zero-padded once.
    """
    size = int(sample_rate * window_s)
    step = int(sample_rate * hop_s)
    if len(audio) <= size:
        return np.pad(audio, (0, size - len(audio)))[None, :]
    starts = list(range(0, len(audio) - size + 1, step))
    if starts[-1] + size < len(audio):
        starts.append(len(audio) - size)
    return np.stack([audio[s : s + size] for s in starts])


def normalize(window: np.ndarray) -> np.ndarray:
    """Scale a window so its loudest sample is 1.0.

    Per window, not per file: the model is trained this way, and it makes the
    input independent of recording gain. A silent window stays silent (the
    epsilon only avoids dividing by zero).
    """
    return window / (np.max(np.abs(window)) + 1e-6)


# ---------------------------------------------------------------------------
# 2. The hybrid frontend's STFT (skip this entirely for `raw`)
# ---------------------------------------------------------------------------


def hann_window(size: int) -> np.ndarray:
    """Periodic Hann window, the one librosa and scipy call ``sym=False``."""
    n = np.arange(size, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / size)).astype(np.float32)


def magnitude_stft(window: np.ndarray, fft_length: int, frames: int, compression: str) -> np.ndarray:
    """Magnitude STFT exactly as the model expects it.

    Specified in ``docs/dev/spectrogram-input.md``; the firmware reproduces this
    arithmetic on the Cortex-M55. Three details are easy to get wrong:

    1. **The STFT is centered.** The window is zero-padded by ``fft_length / 2``
       on both sides first, so frame ``i`` is *centered* on sample ``i * hop``,
       not started there.
    2. **The hop follows from the frame count**: ``hop = samples // frames``.
       For 2.5 s at 24 kHz into 384 frames that is 156 samples.
    3. **The result is min-max normalized to [0, 1]** after compression. This is
       what makes the hybrid input independent of recording gain: scaling the
       audio scales every magnitude by the same factor, which the normalization
       divides out. A hybrid bundle therefore does not need the per-window peak
       normalization that `raw` needs.

    Output is ``[fft_length // 2, frames]`` — the Nyquist bin is dropped, so
    512 -> 256 rows.
    """
    hop = len(window) // frames
    if hop < 1:
        raise ValueError(f"{len(window)} samples cannot hold {frames} frames")
    taper = hann_window(fft_length)
    padded = np.pad(window, fft_length // 2)
    spectrum = np.zeros((fft_length // 2, frames), dtype=np.float32)
    for i in range(frames):
        start = i * hop
        spectrum[:, i] = np.abs(np.fft.rfft(padded[start : start + fft_length] * taper))[: fft_length // 2]

    if compression == "sqrt":
        spectrum = np.sqrt(spectrum)
    elif compression == "log":
        floor = spectrum.max() * (10.0 ** (-80.0 / 20.0))  # LOG_FLOOR_DB below the peak
        spectrum = np.log(np.maximum(spectrum, floor))

    low, high = spectrum.min(), spectrum.max()
    return ((spectrum - low) / (high - low + 1e-10)).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. The model
# ---------------------------------------------------------------------------


def model_input(window: np.ndarray, config: dict) -> np.ndarray:
    """Turn one window into the model's input tensor.

    `raw` wants the peak-normalized waveform. `hybrid` wants the spectrogram,
    which normalizes itself, so it takes the window as recorded.
    """
    if config["audio_frontend"] == "raw":
        return normalize(window)[:, None].astype(np.float32)  # [samples, 1]
    spectrum = magnitude_stft(
        window,
        fft_length=int(config["fft_length"]),
        frames=int(config["spec_width"]),
        compression=config.get("input_compression", "none"),
    )
    return spectrum[:, :, None].astype(np.float32)  # [bins, frames, 1]


def predict(interpreter, batch: np.ndarray) -> np.ndarray:
    """Run the model once per window. Bundles are built for batch size 1."""
    in_index = interpreter.get_input_details()[0]["index"]
    out_index = interpreter.get_output_details()[0]["index"]
    out = []
    for item in batch:
        interpreter.set_tensor(in_index, item[None].astype(np.float32))
        interpreter.invoke()
        out.append(interpreter.get_tensor(out_index)[0].copy())
    return np.stack(out)


def to_probabilities(scores: np.ndarray, config: dict) -> np.ndarray:
    """Make the model's output comparable against thresholds.

    A bundle whose config says ``"output_activation": "logit"`` emits logits, and
    the sigmoid is applied here — 100 values per window, negligible on any CPU.
    Firmware can skip it and compare logits against ``log(t / (1 - t))`` instead,
    which is exact and costs nothing. Anything else already emits probabilities.
    """
    if config.get("output_activation", "sigmoid") == "logit":
        return 1.0 / (1.0 + np.exp(-scores))
    return scores


# ---------------------------------------------------------------------------
# 4. Windows -> detections
# ---------------------------------------------------------------------------


def pool(window_scores: np.ndarray) -> np.ndarray:
    """One score per class for the recording: the loudest evidence anywhere.

    Max over windows, matching how the models were evaluated. Averaging would
    punish a species that calls once in a long recording.
    """
    return window_scores.max(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bundle", type=Path, required=True, help="Directory of an unpacked model bundle")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--explain", action="store_true", help="Print every step's shapes and ranges")
    args = parser.parse_args()

    tflite = next(p for p in sorted(args.bundle.glob("*_INT8.tflite")) if "backbone" not in p.name)
    config = json.loads(next(args.bundle.glob("*_model_config.json")).read_text())
    labels = [line.strip() for line in next(args.bundle.glob("*_labels.txt")).read_text().splitlines() if line.strip()]

    sample_rate = int(config["sample_rate"])
    window_s = float(config["chunk_duration"])
    hop_s = window_s / 2.0  # the deployment hop: 1.25 s for a 2.5 s window

    audio = read_audio(args.audio, sample_rate)
    frames = windows(audio, sample_rate, window_s, hop_s)
    if args.explain:
        print(f"model            {tflite.name}")
        print(f"frontend         {config['audio_frontend']}  compression={config.get('input_compression', 'none')}")
        print(f"output           {config.get('output_activation', 'sigmoid')}")
        print(f"audio            {len(audio)} samples = {len(audio) / sample_rate:.1f} s at {sample_rate} Hz")
        print(f"windows          {frames.shape[0]} of {window_s} s, hop {hop_s} s")

    interpreter = Interpreter(model_path=str(tflite))
    interpreter.allocate_tensors()

    inputs = np.stack([model_input(w, config) for w in frames])
    if args.explain:
        expected = tuple(interpreter.get_input_details()[0]["shape"][1:])
        print(f"input tensor     {inputs.shape[1:]} (model expects {expected})")
        print(f"input range      [{inputs.min():.4f}, {inputs.max():.4f}]")

    raw_scores = predict(interpreter, inputs)
    scores = to_probabilities(raw_scores, config)
    if args.explain:
        print(f"model output     {raw_scores.shape[1]} classes, range [{raw_scores.min():.3f}, {raw_scores.max():.3f}]")
        print(f"as probabilities range [{scores.min():.4f}, {scores.max():.4f}]")

    pooled = pool(scores)
    order = np.argsort(pooled)[::-1][: args.top]
    print(f"\nTop {args.top} for {args.audio.name}:")
    for index in order:
        mark = "  <- detected" if pooled[index] >= args.threshold else ""
        print(f"  {labels[index]:32} {pooled[index]:.3f}{mark}")


if __name__ == "__main__":
    main()
