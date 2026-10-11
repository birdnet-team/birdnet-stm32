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


def calibration_source(data_path_train: str, calibration_dir: str, classes: list[str] | None) -> list[str]:
    """Audio files that INT8 calibration draws from.

    By default the training files of the model's classes. ``calibration_dir``
    replaces them with any directory of audio, grouped by subfolder: field
    recordings from the deployment domain calibrate the activation ranges on the
    input density the model will actually see, while training stays on focal
    audio. The draw is stratified by subfolder and seeded, as always.
    """
    from birdnet_stm32.data.dataset import load_file_paths_from_directory

    if calibration_dir:
        paths, _ = load_file_paths_from_directory(calibration_dir)
        if not paths:
            raise ValueError(f"No audio found under --calibration_dir {calibration_dir}")
        return paths
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
_RANGE_BINS = 8192


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
    calibrated: bytes, samples: list[list[np.ndarray]], percentile: float = ACTIVATION_RANGE_PERCENTILE
) -> bytes:
    """Shrink a calibrated model's activation ranges to a percentile of the calibration values.

    ``calibrated`` is the converter's calibrate-only model, whose tensors carry the
    min/max calibration measured. Each tensor's values on ``samples`` are binned
    between that min and max, and the range is cut to the given percentile at each
    end (the low end only where it is negative). The model input and outputs keep
    min/max: clipping the outputs would flatten the top scores. Ranges only shrink.
    """
    from tensorflow.lite.tools import flatbuffer_utils

    ranges = _calibrated_ranges(calibrated)
    interpreter = tf.lite.Interpreter(model_content=calibrated, experimental_preserve_all_tensors=True)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    keep = {input_index} | {d["index"] for d in interpreter.get_output_details()}
    watch = [i for i, (lo, hi) in ranges.items() if i not in keep and hi > lo]
    hist = {i: np.zeros(_RANGE_BINS, np.int64) for i in watch}
    for sample in samples:
        interpreter.set_tensor(input_index, np.asarray(sample[0], np.float32))
        interpreter.invoke()
        for i in watch:
            hist[i] += np.histogram(interpreter.get_tensor(i), bins=_RANGE_BINS, range=ranges[i])[0]

    flat = flatbuffer_utils.read_model_from_bytearray(bytearray(calibrated))
    tensors = flat.subgraphs[0].tensors
    tail = (100.0 - percentile) / 100.0
    for i in watch:
        total = hist[i].sum()
        if total == 0:
            continue
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
    ``ACTIVATION_RANGE_PERCENTILE`` of the same data (``clip_activation_ranges``),
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
