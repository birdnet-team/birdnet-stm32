#!/usr/bin/env python3
"""Reference implementation of BirdNET Tiny N6 inference, in one readable file.

This is the specification of everything between an audio file and a detection:
what a host program or a firmware has to compute, in the order it computes it.
It uses NumPy and SoundFile for I/O and TensorFlow Lite to run the model, and
nothing from the ``birdnet_stm32`` package, so it can be read top to bottom and
ported to another language. Every constant comes from the bundle's
``*_model_config.json``; nothing is hard-coded.

Run a bundle on a recording:

    python reference/birdnet_tiny_reference.py \\
        --bundle BirdNET_Tiny_N6_USNE_90_V1.6_Raw --audio recording.wav --explain

Check a re-implementation, stage by stage, against the committed test vectors:

    python reference/birdnet_tiny_reference.py --bundle <bundle> \\
        --check-vectors reference/vectors/BirdNET_Tiny_N6_USNE_90_V1.6_Raw.json

The pipeline, and where each step runs on the STM32N6:

    1. read + resample to the model's rate, mono   host; the device records 24 kHz PCM16
    2. cut into windows (2.5 s, 1.25 s hop)         firmware
    3. raw only:    divide each window by its peak  firmware
       hybrid only: magnitude STFT, compress,       Cortex-M55 (CMSIS-DSP, Helium)
                    min-max normalize
    4. run the model on each window                 NPU
    5. sigmoid, if the bundle emits logits          firmware (or threshold the logits)
    6. pool windows per file (max), threshold       firmware

The two frontends differ only in step 3. A `raw` model computes its own
spectrogram inside the network, with a learned filterbank that runs on the NPU,
so all it needs is the normalized waveform. A `hybrid` model needs a magnitude
STFT computed outside the network, on the CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter
except ImportError:  # a bundle runs fine on the small runtime too
    from tflite_runtime.interpreter import Interpreter  # type: ignore[no-redef]

HERE = Path(__file__).resolve().parent
TEST_SIGNAL = HERE / "vectors" / "test_signal.wav"


# ---------------------------------------------------------------------------
# Step 1: audio
# ---------------------------------------------------------------------------


def read_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Read a file as mono float32 at ``sample_rate``, in [-1, 1].

    Channels are averaged. Resampling is polyphase (``scipy.signal.resample_poly``
    with the reduced ratio), which is what the models were trained with; any
    good band-limited resampler gives scores within the tolerance below. A
    recorder that already writes mono PCM16 at the model's rate only scales the
    int16 samples by 1/32768.
    """
    audio, file_rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if file_rate != sample_rate:
        from scipy.signal import resample_poly

        divisor = gcd(file_rate, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, file_rate // divisor).astype(np.float32)
    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 2: windows
# ---------------------------------------------------------------------------


def window_starts(num_samples: int, size: int, step: int) -> list[int]:
    """Start sample of every window.

    Windows step by ``step``. The last window is right-aligned to the end of the
    recording instead of zero-padded, so the final seconds are scored at full
    weight. A recording shorter than one window gives one window, zero-padded.
    """
    if num_samples <= size:
        return [0]
    starts = list(range(0, num_samples - size + 1, step))
    if starts[-1] + size < num_samples:
        starts.append(num_samples - size)
    return starts


def windows(audio: np.ndarray, sample_rate: int, window_s: float, hop_s: float) -> np.ndarray:
    """Cut the recording into ``[n_windows, samples]`` overlapping windows."""
    size, step = int(sample_rate * window_s), int(sample_rate * hop_s)
    if len(audio) < size:
        audio = np.pad(audio, (0, size - len(audio)))
    return np.stack([audio[s : s + size] for s in window_starts(len(audio), size, step)])


# ---------------------------------------------------------------------------
# Step 3a: raw frontend input
# ---------------------------------------------------------------------------


def peak_normalize(window: np.ndarray) -> np.ndarray:
    """Divide a window by its largest absolute sample, so that becomes ~1.0.

    Per window, not per file: the models are trained this way, and it makes the
    input independent of recording gain. The epsilon only keeps a silent window
    silent. The result is the raw model's input, ``[samples, 1]``.
    """
    return window / (np.max(np.abs(window)) + 1e-6)


# ---------------------------------------------------------------------------
# Step 3b: hybrid frontend input
# ---------------------------------------------------------------------------


def hann_window(size: int) -> np.ndarray:
    """Periodic Hann window: ``0.5 - 0.5 cos(2 pi n / size)``, ``n = 0 .. size-1``.

    Periodic, not symmetric (the divisor is ``size``, not ``size - 1``); SciPy and
    librosa call it ``sym=False`` / ``fftbins=True``.
    """
    n = np.arange(size, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / size)).astype(np.float32)


def magnitude_stft(window: np.ndarray, fft_length: int, frames: int) -> np.ndarray:
    """Linear magnitude STFT, ``[fft_length // 2, frames]``, frequency-major.

    1. **Hop from the frame count**: ``hop = samples // frames`` (2.5 s at
       24 kHz into 384 frames: 156 samples).
    2. **Centered frames**: zero-pad the window by ``fft_length // 2`` on both
       sides, so frame ``i`` covers padded samples ``[i*hop, i*hop + fft_length)``
       and is centred on original sample ``i*hop``.
    3. Multiply by the periodic Hann window, take the real FFT, keep
       ``|X[k]|`` for ``k = 0 .. fft_length/2 - 1``: **the Nyquist bin is dropped**.
       No scaling of the FFT (no ``1/N``); the min-max step below divides any
       constant factor out anyway.
    """
    hop = len(window) // frames
    if hop < 1:
        raise ValueError(f"{len(window)} samples cannot hold {frames} frames")
    taper = hann_window(fft_length)
    padded = np.pad(window, fft_length // 2)
    spectrum = np.zeros((fft_length // 2, frames), dtype=np.float32)
    for i in range(frames):
        frame = padded[i * hop : i * hop + fft_length] * taper
        spectrum[:, i] = np.abs(np.fft.rfft(frame))[: fft_length // 2]
    return spectrum


def compress(spectrum: np.ndarray, mode: str) -> np.ndarray:
    """``input_compression`` from the config, applied before normalization.

    ``sqrt`` (every released hybrid bundle): element-wise square root. ``log``:
    natural log, floored 80 dB below the window's peak. ``none``: unchanged.
    """
    if mode == "sqrt":
        return np.sqrt(spectrum)
    if mode == "log":
        floor = max(float(spectrum.max()), 1e-10) * (10.0 ** (-80.0 / 20.0))
        return np.log(np.maximum(spectrum, floor))
    return spectrum


def minmax_normalize(spectrum: np.ndarray) -> np.ndarray:
    """``(S - min) / (max - min + 1e-10)`` over the whole window's spectrogram.

    This is what makes the hybrid input independent of recording gain, so a
    hybrid window is *not* peak-normalized first (doing so changes nothing).
    """
    low, high = float(spectrum.min()), float(spectrum.max())
    return ((spectrum - low) / (high - low + 1e-10)).astype(np.float32)


def hybrid_input(window: np.ndarray, fft_length: int, frames: int, compression: str) -> np.ndarray:
    """The hybrid model's input: ``[fft_length // 2, frames, 1]`` in [0, 1]."""
    return minmax_normalize(compress(magnitude_stft(window, fft_length, frames), compression))


# ---------------------------------------------------------------------------
# Step 3: either
# ---------------------------------------------------------------------------


def model_input(window: np.ndarray, config: dict) -> np.ndarray:
    """One window as the model's input tensor, without the batch axis."""
    if config["audio_frontend"] == "raw":
        return peak_normalize(window)[:, None].astype(np.float32)  # [samples, 1]
    spec = hybrid_input(
        window,
        fft_length=int(config["fft_length"]),
        frames=int(config["spec_width"]),
        compression=config.get("input_compression", "none"),
    )
    return spec[:, :, None]  # [bins, frames, 1]


# ---------------------------------------------------------------------------
# Steps 4-6: the model, the output, the recording
# ---------------------------------------------------------------------------


def load_bundle(bundle: Path):
    """The full INT8 model, its config and its labels from a bundle directory.

    A bundle also ships the model split into ``_INT8_backbone`` and
    ``_INT8_classifier``; chaining them computes the same thing.
    """
    tflite = next(p for p in sorted(bundle.glob("*_INT8.tflite")))
    config = json.loads(next(bundle.glob("*_model_config.json")).read_text())
    labels_file = next(p for p in sorted(bundle.glob("*_labels.txt")) if "_classifier_" not in p.name)
    labels = [line.strip() for line in labels_file.read_text().splitlines() if line.strip()]
    interpreter = Interpreter(model_path=str(tflite))
    interpreter.allocate_tensors()
    return interpreter, config, labels, tflite.name


def predict(interpreter, inputs: np.ndarray) -> np.ndarray:
    """Run the model once per window (bundles are built for batch size 1).

    INT8 bundles take and return float32: quantization is inside the model.
    Output is one value per class, in label order.
    """
    in_index = interpreter.get_input_details()[0]["index"]
    out_index = interpreter.get_output_details()[0]["index"]
    out = []
    for item in inputs:
        interpreter.set_tensor(in_index, item[None].astype(np.float32))
        interpreter.invoke()
        out.append(interpreter.get_tensor(out_index)[0].copy())
    return np.stack(out)


def to_probabilities(outputs: np.ndarray, config: dict) -> np.ndarray:
    """Apply the sigmoid when the config says ``"output_activation": "logit"``.

    Every bundle from 1.5 on emits logits; earlier ones emit probabilities.
    Firmware may skip the sigmoid and compare logits against
    ``log(t / (1 - t))`` instead — exact, since the sigmoid is monotonic.
    """
    if config.get("output_activation", "sigmoid") == "logit":
        return (1.0 / (1.0 + np.exp(-outputs.astype(np.float64)))).astype(np.float32)
    return outputs


def pool(probabilities: np.ndarray) -> np.ndarray:
    """One score per class for a recording: the maximum over its windows."""
    return probabilities.max(axis=0)


def run(bundle: Path, audio_path: Path, explain: bool = False) -> dict:
    """The whole pipeline; returns every intermediate for inspection."""
    interpreter, config, labels, model_name = load_bundle(bundle)
    sample_rate = int(config["sample_rate"])
    window_s = float(config["chunk_duration"])
    hop_s = window_s / 2.0  # the deployment hop: 1.25 s for 2.5 s windows

    audio = read_audio(audio_path, sample_rate)
    cut = windows(audio, sample_rate, window_s, hop_s)
    inputs = np.stack([model_input(w, config) for w in cut])
    outputs = predict(interpreter, inputs)
    probabilities = to_probabilities(outputs, config)
    pooled = pool(probabilities)
    if explain:
        expected = tuple(int(v) for v in interpreter.get_input_details()[0]["shape"][1:])
        print(f"model            {model_name}")
        print(f"frontend         {config['audio_frontend']}  compression={config.get('input_compression', 'none')}")
        print(f"output           {config.get('output_activation', 'sigmoid')}")
        print(f"audio            {len(audio)} samples = {len(audio) / sample_rate:.2f} s at {sample_rate} Hz")
        print(f"windows          {len(cut)} of {window_s} s, hop {hop_s} s")
        print(f"input tensor     {inputs.shape[1:]} (model expects {expected})")
        print(f"input range      [{inputs.min():.4f}, {inputs.max():.4f}]")
        print(f"model output     range [{outputs.min():.3f}, {outputs.max():.3f}]")
    return {
        "config": config,
        "labels": labels,
        "model": model_name,
        "audio": audio,
        "windows": cut,
        "inputs": inputs,
        "outputs": outputs,
        "probabilities": probabilities,
        "pooled": pooled,
    }


# ---------------------------------------------------------------------------
# Test vectors
# ---------------------------------------------------------------------------


def make_test_signal(sample_rate: int = 24000, seconds: float = 6.0, seed: int = 0) -> np.ndarray:
    """The synthetic recording the committed vectors are computed from.

    Deterministic and license-free: a low noise floor, a descending two-note
    whistle, a fast frequency sweep with a harmonic, and one loud click, so
    both a sparse and a transient-dominated window occur. It is committed as
    ``vectors/test_signal.wav`` (PCM16); a re-implementation reads that file.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    y = 0.003 * rng.standard_normal(t.size)
    for start in (0.4, 2.1, 3.8):  # whistle: 3.2 kHz then 2.6 kHz, 0.3 s each
        for offset, freq in ((0.0, 3200.0), (0.35, 2600.0)):
            m = (t >= start + offset) & (t < start + offset + 0.3)
            env = np.sin(np.pi * (t[m] - start - offset) / 0.3) ** 2
            y[m] += 0.25 * env * np.sin(2 * np.pi * freq * t[m])
    m = (t >= 1.2) & (t < 1.8)  # sweep 6 -> 3 kHz with its second harmonic
    tau = t[m] - 1.2
    phase = 2 * np.pi * (6000.0 * tau - 2500.0 * tau**2)
    y[m] += 0.15 * np.sin(np.pi * tau / 0.6) * (np.sin(phase) + 0.3 * np.sin(2 * phase))
    click = int(4.9 * sample_rate)  # a transient that dominates its windows' peak
    y[click : click + 24] += 0.9 * np.hanning(24)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def _summary(array: np.ndarray, head: int = 8) -> dict:
    flat = np.asarray(array, dtype=np.float64).ravel()
    return {
        "shape": list(np.shape(array)),
        "sum": round(float(flat.sum()), 6),
        "min": round(float(flat.min()), 7),
        "max": round(float(flat.max()), 7),
        "head": [round(float(v), 7) for v in flat[:head]],
    }


def hybrid_stages(window: np.ndarray, config: dict) -> dict:
    """The hybrid input's intermediate stages, for debugging a port of step 3b."""
    mag = magnitude_stft(window, int(config["fft_length"]), int(config["spec_width"]))
    return {
        "stft_magnitude": _summary(mag),
        "compressed": _summary(compress(mag, config.get("input_compression", "none"))),
    }


def write_vectors(bundle: Path, audio_path: Path, out: Path) -> None:
    """Record every stage of the pipeline on ``audio_path`` as JSON.

    Arrays are summarized (shape, sum, min, max, first values) rather than
    stored, which is enough to locate a divergence and keeps the file small.
    """
    r = run(bundle, audio_path)
    hybrid = r["config"]["audio_frontend"] == "hybrid"
    doc = {
        "bundle": bundle.name,
        "model": r["model"],
        "audio": audio_path.name,
        "audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
        "config": {
            k: r["config"].get(k)
            for k in (
                "sample_rate",
                "chunk_duration",
                "audio_frontend",
                "fft_length",
                "spec_width",
                "input_compression",
                "output_activation",
            )
        },
        "audio_samples": _summary(r["audio"]),
        "windows": [
            {
                "start_sample": int(s),
                "window": _summary(w),
                **(hybrid_stages(w, r["config"]) if hybrid else {}),
                "model_input": _summary(x),
                "model_output": [round(float(v), 5) for v in o],
            }
            for s, w, x, o in zip(
                window_starts(
                    len(r["audio"]),
                    r["windows"].shape[1],
                    int(int(r["config"]["sample_rate"]) * float(r["config"]["chunk_duration"]) / 2),
                ),
                r["windows"],
                r["inputs"],
                r["outputs"],
                strict=True,
            )
        ],
        "pooled": [round(float(v), 5) for v in r["pooled"]],
        "labels": r["labels"],
    }
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {out}: {len(doc['windows'])} windows")


def check_vectors(bundle: Path, vectors: Path, audio_path: Path | None = None) -> bool:
    """Compare this implementation's stages with recorded vectors.

    Tolerances are what a faithful re-implementation achieves in float32 with a
    different FFT or resampler: inputs to 1e-4 relative, and model outputs to
    two INT8 output steps. The detections at 0.5 must match exactly.
    """
    doc = json.loads(vectors.read_text())
    audio_path = audio_path or (vectors.parent / doc["audio"])
    r = run(bundle, audio_path)
    ok = True

    def report(stage: str, good: bool, detail: str) -> None:
        nonlocal ok
        ok &= good
        print(f"  {'ok ' if good else 'BAD'} {stage:14} {detail}")

    print(f"{bundle.name} against {vectors.name}:")
    report("windows", len(doc["windows"]) == len(r["windows"]), f"{len(r['windows'])} (expected {len(doc['windows'])})")

    def rel(a: float, b: float) -> float:
        return abs(a - b) / max(1.0, abs(b))

    stages = {"window": [_summary(w) for w in r["windows"]], "model_input": [_summary(x) for x in r["inputs"]]}
    if "stft_magnitude" in doc["windows"][0]:
        extra = [hybrid_stages(w, r["config"]) for w in r["windows"]]
        stages = {
            "window": stages["window"],
            "stft_magnitude": [e["stft_magnitude"] for e in extra],
            "compressed": [e["compressed"] for e in extra],
            "model_input": stages["model_input"],
        }
    for stage, mine in stages.items():
        worst = max(rel(m["sum"], w[stage]["sum"]) for m, w in zip(mine, doc["windows"], strict=False))
        report(stage, worst < 1e-4, f"worst relative sum difference {worst:.2e}")
    expected = np.array([w["model_output"] for w in doc["windows"]])
    worst_out = float(np.max(np.abs(r["outputs"][: len(expected)] - expected)))
    step = output_step(bundle)
    report("model output", worst_out <= 2 * step + 1e-5, f"worst difference {worst_out:.4f} (output step {step:.4f})")
    same = np.array_equal(np.array(doc["pooled"]) >= 0.5, r["pooled"] >= 0.5)
    report("detections", same, "at threshold 0.5")
    return ok


def _squeezed(shape: list[int]) -> list[int]:
    """A shape without trailing 1s: ``[60000, 1]`` and ``[60000]`` are the same tensor."""
    shape = list(shape)
    while len(shape) > 1 and shape[-1] == 1:
        shape.pop()
    return shape


def check_stages(vectors: Path, stages: Path) -> bool:
    """Compare a frontend port's stage summaries with recorded vectors, no model needed.

    ``stages`` holds one JSON object per window, in the format the vectors use
    (``start_sample`` plus ``window``, ``stft_magnitude``, ``compressed``,
    ``model_input`` summaries) -- what ``reference/c/frontend_cli`` prints.
    Stages the file does not carry are skipped.
    """
    doc = json.loads(vectors.read_text())
    mine = [json.loads(line) for line in stages.read_text().splitlines() if line.strip()]
    ok = len(mine) == len(doc["windows"])
    print(f"{stages.name} against {vectors.name}:")
    print(f"  {'ok ' if ok else 'BAD'} {'windows':14} {len(mine)} (expected {len(doc['windows'])})")
    starts = [m["start_sample"] for m in mine] == [w["start_sample"] for w in doc["windows"]]
    print(f"  {'ok ' if starts else 'BAD'} {'window starts':14} {[m['start_sample'] for m in mine]}")
    ok &= starts
    for stage in ("window", "stft_magnitude", "compressed", "model_input"):
        pairs = [(m[stage], w[stage]) for m, w in zip(mine, doc["windows"], strict=False) if stage in m and stage in w]
        if not pairs:
            continue
        worst = max(abs(a["sum"] - b["sum"]) / max(1.0, abs(b["sum"])) for a, b in pairs)
        shapes = all(_squeezed(a["shape"]) == _squeezed(b["shape"]) for a, b in pairs)
        good = worst < 1e-4 and shapes
        ok &= good
        print(f"  {'ok ' if good else 'BAD'} {stage:14} worst relative sum difference {worst:.2e}")
    return ok


def output_step(bundle: Path) -> float:
    """The INT8 model's output grid step, read from its final dequantize."""
    interpreter, _, _, _ = load_bundle(bundle)
    out_index = interpreter.get_output_details()[0]["index"]
    for op in interpreter._get_ops_details():  # noqa: SLF001
        if out_index in op["outputs"] and op["op_name"] == "DEQUANTIZE":
            scale = interpreter.get_tensor_details()[op["inputs"][0]]["quantization"][0]
            return float(scale)
    return 1.0 / 256.0


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bundle", type=Path, help="Directory of an unpacked model bundle")
    parser.add_argument("--audio", type=Path, default=TEST_SIGNAL, help="Recording (default: the test signal)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--explain", action="store_true", help="Print every step's shapes and ranges")
    parser.add_argument("--dump", type=Path, help="Write every stage as .npy files into this directory")
    parser.add_argument("--check-vectors", type=Path, help="Compare against a vectors JSON")
    parser.add_argument(
        "--stages",
        type=Path,
        help="With --check-vectors: compare these stage summaries (JSON lines, e.g. from reference/c) "
        "instead of running this implementation; no bundle needed",
    )
    parser.add_argument("--write-vectors", type=Path, help="Record a vectors JSON for --bundle and --audio")
    parser.add_argument("--make-test-signal", action="store_true", help="(Re)write vectors/test_signal.wav")
    args = parser.parse_args()

    if args.make_test_signal:
        TEST_SIGNAL.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(TEST_SIGNAL), make_test_signal(), 24000, subtype="PCM_16")
        print(f"wrote {TEST_SIGNAL}")
        return
    if args.check_vectors and args.stages:
        raise SystemExit(0 if check_stages(args.check_vectors, args.stages) else 1)
    if args.bundle is None:
        parser.error("--bundle is required")
    if args.check_vectors:
        raise SystemExit(0 if check_vectors(args.bundle, args.check_vectors) else 1)
    if args.write_vectors:
        write_vectors(args.bundle, args.audio, args.write_vectors)
        return

    r = run(args.bundle, args.audio, explain=args.explain)
    if args.dump:
        args.dump.mkdir(parents=True, exist_ok=True)
        for name in ("audio", "windows", "inputs", "outputs", "probabilities", "pooled"):
            np.save(args.dump / f"{name}.npy", r[name])
        print(f"stages written to {args.dump}")
    order = np.argsort(r["pooled"])[::-1][: args.top]
    print(f"\nTop {args.top} for {args.audio.name}:")
    for index in order:
        mark = "  <- detected" if r["pooled"][index] >= args.threshold else ""
        print(f"  {r['labels'][index]:32} {r['pooled'][index]:.3f}{mark}")


if __name__ == "__main__":
    main()
