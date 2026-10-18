"""Post-training quantization (PTQ) conversion from Keras to TFLite.

Provides representative dataset generation and TFLite conversion with
float32 I/O and INT8 internal ops for STM32N6 NPU deployment.
"""

import os
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from birdnet_stm32.audio.io import load_audio_file, peak_normalize
from birdnet_stm32.audio.spectrogram import get_spectrogram_from_audio
from birdnet_stm32.models.frontend import normalize_frontend_name
from birdnet_stm32.models.runners import allocated_interpreter


def calibration_source(data_path_train: str, classes: list[str] | None) -> list[str]:
    """Audio files that INT8 calibration draws from: the training files of the model's classes.

    The draw from them is stratified by class folder and seeded. Calibrating on
    field recordings instead (``--calibration_dir``, 1.5 to 2.1) was measured
    twice and never helped: 1.5 Raw catalog -0.022 at flat WABAD; 2.2's N6-B0
    with a 50/50 focal + field set, WABAD bench 0.4685 -> 0.4619.
    """
    from birdnet_stm32.data.dataset import load_file_paths_from_directory

    paths, _ = load_file_paths_from_directory(data_path_train, classes=classes)
    return paths


def stratified_sample_paths(
    file_paths: list[str],
    num_samples: int,
    *,
    seed: int,
    exclude: set[str] | None = None,
) -> list[str]:
    """Select an exact, deterministic, class-balanced audio manifest."""
    excluded = exclude or set()
    grouped: dict[str, list[str]] = defaultdict(list)
    for path in sorted(set(file_paths)):
        if path not in excluded:
            grouped[os.path.basename(os.path.dirname(path))].append(path)

    rng = random.Random(seed)
    class_names = sorted(grouped)
    rng.shuffle(class_names)
    for paths in grouped.values():
        rng.shuffle(paths)

    selected: list[str] = []
    offsets = {name: 0 for name in class_names}
    while len(selected) < num_samples:
        added = False
        for name in class_names:
            offset = offsets[name]
            paths = grouped[name]
            if offset < len(paths):
                selected.append(paths[offset])
                offsets[name] = offset + 1
                added = True
                if len(selected) == num_samples:
                    break
        if not added:
            break
    return selected


def representative_data_gen(
    file_paths: list[str], cfg: dict, num_samples: int = 100, snr_threshold: float = 0.0
) -> Iterator[list[np.ndarray]]:
    """Build a representative dataset generator for TFLite PTQ calibration.

    Yields one input tensor per iteration in the exact shape expected by the model.
    Includes quiet and nuisance recordings by default because they are part of
    the deployed input distribution and every requested calibration path must
    contribute deterministically. Callers may opt into energy filtering.

    Args:
        file_paths: Audio file paths to sample from.
        cfg: Training config dict (sample_rate, num_mels, spec_width, chunk_duration,
             fft_length, audio_frontend, mag_scale).
        num_samples: Maximum number of samples to draw.
        snr_threshold: Minimum RMS energy for a chunk to be included (0 to disable).

    Yields:
        Single-element list containing the input tensor with batch dimension.
    """
    sr = int(cfg["sample_rate"])
    num_mels = int(cfg["num_mels"])
    spec_width = int(cfg["spec_width"])
    cd = float(cfg["chunk_duration"])
    n_fft = int(cfg["fft_length"])
    frontend = normalize_frontend_name(cfg["audio_frontend"])
    mag_scale = cfg.get("mag_scale", "none")
    compression = cfg.get("input_compression", "none")
    T = int(sr * cd)

    if len(file_paths) == 0:
        raise ValueError("No audio files found for representative dataset generation.")
    # The caller owns sampling.  Keeping this iterator ordered makes calibration
    # reproducible and allows conversion and diagnostics to use the exact same
    # file manifest.
    sampled_paths = file_paths[: min(num_samples, len(file_paths))]
    # Bound calibration read length to a few chunks per file: longer reads waste
    # I/O and only the centre chunk is kept downstream.
    rep_max_duration = float(cfg.get("max_duration", 0)) or max(30.0, cd * 5.0)

    for path in tqdm(sampled_paths, desc="Generating rep. dataset", unit="file", dynamic_ncols=True):
        audio_chunks = load_audio_file(path, sample_rate=sr, max_duration=rep_max_duration, chunk_duration=cd)

        # Pick center chunk to avoid silence-only calibration
        if audio_chunks.shape[0] > 1:
            middle = audio_chunks.shape[0] // 2
            audio_chunks = audio_chunks[middle : middle + 1]

        if frontend == "librosa":
            specs = [
                get_spectrogram_from_audio(
                    ch,
                    sample_rate=sr,
                    n_fft=n_fft,
                    mel_bins=num_mels,
                    spec_width=spec_width,
                    mag_scale=mag_scale,
                    compression=compression,
                )
                for ch in audio_chunks
            ]
            pool = [s for s in specs if s is not None and np.size(s) > 0]
        elif frontend == "hybrid":
            specs = [
                get_spectrogram_from_audio(
                    ch, sample_rate=sr, n_fft=n_fft, mel_bins=-1, spec_width=spec_width, compression=compression
                )
                for ch in audio_chunks
            ]
            pool = [s for s in specs if s is not None and np.size(s) > 0]
        elif frontend == "raw":
            if isinstance(audio_chunks, np.ndarray):
                pool = [audio_chunks[i] for i in range(audio_chunks.shape[0])]
            else:
                pool = list(audio_chunks)
            pool = [c for c in pool if c is not None and np.size(c) > 0]
        else:
            raise ValueError(f"Invalid audio frontend: {frontend}")

        if len(pool) == 0:
            continue

        for sample in pool:
            if frontend == "raw":
                x = sample[:T]
                if x.shape[0] < T:
                    x = np.pad(x, (0, T - x.shape[0]))
                # Skip near-silent chunks
                rms = np.sqrt(np.mean(x**2))
                if snr_threshold > 0 and rms < snr_threshold:
                    continue
                x = peak_normalize(x)
                x = x.astype(np.float32)[None, :, None]
            elif frontend == "hybrid":
                x = sample.astype(np.float32)[None, :, :, None]
                # Skip near-silent spectrograms
                if snr_threshold > 0 and np.mean(np.abs(x)) < snr_threshold:
                    continue
            else:
                x = sample.astype(np.float32)[None, :, :, None]
                # Skip near-silent spectrograms
                if snr_threshold > 0 and np.mean(np.abs(x)) < snr_threshold:
                    continue
            yield [x]


# Activation ranges are set at this percentile of each tensor's values on the
# calibration data, not at their min/max: one rare outlier otherwise stretches a
# range and coarsens the INT8 grid for every other value. Measured on 2.0 Raw over
# 61 WABAD sites, INT8 window AUPRC 0.258 -> 0.286 (float 0.302) and 20% more
# annotated calls found at 0.5, at the same size and speed; the gain is all in the
# audio frontend's tensors. p99.9 clips too far (below min/max), p99.9999 too little.
ACTIVATION_RANGE_PERCENTILE = 99.999
# The raw frontend's tensors (filterbank, band magnitudes, PWL compressor) are cut harder. Their ranges are set by
# loud focal calls while soundscape bins sit a few INT8 steps above zero (N6-B0 on WABAD windows: median level 3-12
# of 255). Measured on 2.2's N6-B0 over 61 WABAD sites, INT8 window AUPRC 0.429 -> 0.437 (float 0.465), recall @0.5
# 0.357 -> 0.368, catalog cMAP 0.7115 -> 0.7150 (float 0.7341); the backbone keeps p99.999 (p99.99 there costs).
FRONTEND_RANGE_PERCENTILE = 99.99
FRONTEND_TENSORS = "audio_frontend"
_RANGE_BINS = 8192


def range_percentile(name: str) -> float:
    """The range percentile for an activation tensor (or QAT boundary) by name."""
    return FRONTEND_RANGE_PERCENTILE if FRONTEND_TENSORS in name else ACTIVATION_RANGE_PERCENTILE


def _binnable(lo: float, hi: float) -> bool:
    """Whether ``[lo, hi]`` splits into ``_RANGE_BINS`` finite bins in float32 tensors' precision.

    A tensor that is nearly constant on the calibration data (a saturated squeeze-and-excite gate, for one)
    can have a range only a few float steps wide; numpy cannot bin it, and there is nothing to clip anyway.
    """
    edges = np.linspace(np.float32(lo), np.float32(hi), _RANGE_BINS + 1, dtype=np.float32)  # numpy's own check
    return bool(hi > lo and np.all(np.diff(edges) > 0))


def _calibrated_ranges(model_content: bytes) -> dict[int, tuple[float, float]]:
    """Tensor index -> (min, max) for every tensor the calibration gave a range."""
    from tensorflow.lite.tools import flatbuffer_utils

    flat = flatbuffer_utils.read_model_from_bytearray(bytearray(model_content))
    ranges = {}
    for index, tensor in enumerate(flat.subgraphs[0].tensors):
        q = tensor.quantization
        if q is not None and q.min is not None and q.max is not None and len(q.min) == 1:
            ranges[index] = (float(q.min[0]), float(q.max[0]))
    return ranges


def clip_activation_ranges(
    calibrated: bytes,
    samples: list[list[np.ndarray]],
    percentile: float | Callable[[str], float] = range_percentile,
) -> bytes:
    """Shrink a calibrated model's activation ranges to a percentile of the calibration values.

    ``calibrated`` is the converter's calibrate-only model, whose tensors carry the
    min/max calibration measured. Each tensor's values on ``samples`` are binned
    between that min and max, and the range is cut to the given percentile at each
    end (the low end only where it is negative); ``percentile`` is one value or a
    function of the tensor name (default ``range_percentile``). The model input and
    outputs keep min/max: clipping the outputs would flatten the top scores. Ranges
    only shrink.
    """
    from tensorflow.lite.tools import flatbuffer_utils

    ranges = _calibrated_ranges(calibrated)
    interpreter = tf.lite.Interpreter(model_content=calibrated, experimental_preserve_all_tensors=True)
    interpreter.allocate_tensors()
    names = {d["index"]: d["name"] for d in interpreter.get_tensor_details()}
    input_index = interpreter.get_input_details()[0]["index"]
    keep = {input_index} | {d["index"] for d in interpreter.get_output_details()}
    watch = [i for i, (lo, hi) in ranges.items() if i not in keep and _binnable(lo, hi)]
    hist = {i: np.zeros(_RANGE_BINS, np.int64) for i in watch}
    for sample in samples:
        interpreter.set_tensor(input_index, np.asarray(sample[0], np.float32))
        interpreter.invoke()
        for i in watch:
            hist[i] += np.histogram(interpreter.get_tensor(i), bins=_RANGE_BINS, range=ranges[i])[0]

    flat = flatbuffer_utils.read_model_from_bytearray(bytearray(calibrated))
    tensors = flat.subgraphs[0].tensors
    for i in watch:
        total = hist[i].sum()
        if total == 0:
            continue
        tail = (100.0 - (percentile(names[i]) if callable(percentile) else percentile)) / 100.0
        lo, hi = ranges[i]
        edges = np.linspace(lo, hi, _RANGE_BINS + 1)
        cdf = np.cumsum(hist[i]) / total
        new_hi = min(hi, float(edges[min(_RANGE_BINS, np.searchsorted(cdf, 1.0 - tail) + 1)]))
        new_lo = max(lo, float(edges[np.searchsorted(cdf, tail)])) if lo < 0 else lo
        if new_hi <= new_lo:
            continue
        tensors[i].quantization.min = np.array([new_lo], np.float32)
        tensors[i].quantization.max = np.array([new_hi], np.float32)
    return bytes(flatbuffer_utils.convert_object_to_bytearray(flat))


def convert_to_tflite(
    model: tf.keras.Model,
    rep_data_gen: Callable[[], Iterable[list[np.ndarray]]],
    output_path: str,
    per_tensor: bool = False,
) -> bytes:
    """Convert a Keras model to quantized TFLite with float32 I/O and INT8 internals.

    Full INT8 post-training quantization: the converter calibrates every
    activation range on ``rep_data_gen``, the ranges are clipped to
    ``range_percentile`` of the same data (``clip_activation_ranges``),
    and the model is quantized with those ranges.

    Args:
        model: Loaded Keras model.
        rep_data_gen: Callable returning an iterable of [input_tensor] for calibration.
        output_path: Path to save the .tflite model.
        per_tensor: If True, use per-tensor instead of per-channel quantization.

    Returns:
        Raw TFLite model bytes.
    """
    samples = list(rep_data_gen())
    if not samples:
        raise ValueError("The representative dataset is empty")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    converter.representative_dataset = lambda: iter(samples)
    converter._experimental_new_quantizer = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if per_tensor:
        converter._experimental_disable_per_channel = True
        print("Using per-tensor quantization (less accurate, use only if per-channel causes issues).")

    # The debugger exposes the calibrate-only model and quantizes from an edited one.
    debugger = tf.lite.experimental.QuantizationDebugger(converter=converter, debug_dataset=lambda: iter(samples[:1]))
    debugger.calibrated_model = clip_activation_ranges(bytes(debugger.calibrated_model), samples)
    tflite_model = bytes(debugger.get_nondebug_quantized_model())

    # Verify the public float32 I/O contract; audio is still quantized internally.
    interpreter = allocated_interpreter(model_content=tflite_model)
    in_dtype = interpreter.get_input_details()[0]["dtype"]
    out_dtype = interpreter.get_output_details()[0]["dtype"]
    if in_dtype != np.float32 or out_dtype != np.float32:
        raise RuntimeError(
            f"Quantized model has non-float32 I/O (input={in_dtype}, output={out_dtype}). "
            "Audio inputs require float32 I/O with INT8 internals only."
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    return tflite_model
