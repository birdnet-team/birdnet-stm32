"""Post-training quantization (PTQ) conversion from Keras to TFLite.

Provides representative dataset generation and TFLite conversion with
float32 I/O and INT8 internal ops for STM32N6 NPU deployment.
"""

import os
import random

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from birdnet_stm32.audio.io import load_audio_file
from birdnet_stm32.audio.spectrogram import get_spectrogram_from_audio


def representative_data_gen(file_paths: list[str], cfg: dict, num_samples: int = 100):
    """Build a representative dataset generator for TFLite PTQ calibration.

    Yields one input tensor per iteration in the exact shape expected by the model.

    Args:
        file_paths: Audio file paths to sample from.
        cfg: Training config dict (sample_rate, num_mels, spec_width, chunk_duration,
             fft_length, audio_frontend, mag_scale).
        num_samples: Maximum number of samples to draw.

    Yields:
        Single-element list containing the input tensor with batch dimension.
    """
    sr = int(cfg["sample_rate"])
    num_mels = int(cfg["num_mels"])
    spec_width = int(cfg["spec_width"])
    cd = float(cfg["chunk_duration"])
    n_fft = int(cfg["fft_length"])
    frontend = cfg["audio_frontend"]
    mag_scale = cfg.get("mag_scale", "none")
    T = int(sr * cd)

    if len(file_paths) == 0:
        raise ValueError("No audio files found for representative dataset generation.")
    sampled_paths = random.sample(file_paths, min(num_samples, len(file_paths)))

    for path in tqdm(sampled_paths, desc="Generating rep. dataset", unit="file", dynamic_ncols=True):
        audio_chunks = load_audio_file(path, sample_rate=sr, max_duration=30, chunk_duration=cd)

        # Pick center chunk to avoid silence-only calibration
        if isinstance(audio_chunks, list) and len(audio_chunks) > 1:
            audio_chunks = [audio_chunks[len(audio_chunks) // 2]]
        elif isinstance(audio_chunks, np.ndarray) and audio_chunks.shape[0] > 1:
            audio_chunks = [audio_chunks[audio_chunks.shape[0] // 2]]

        if frontend in ("precomputed", "librosa"):
            specs = [
                get_spectrogram_from_audio(ch, sample_rate=sr, n_fft=n_fft, mel_bins=num_mels, spec_width=spec_width, mag_scale=mag_scale)
                for ch in audio_chunks
            ]
            pool = [s for s in specs if s is not None and np.size(s) > 0]
        elif frontend == "hybrid":
            specs = [
                get_spectrogram_from_audio(ch, sample_rate=sr, n_fft=n_fft, mel_bins=-1, spec_width=spec_width)
                for ch in audio_chunks
            ]
            pool = [s for s in specs if s is not None and np.size(s) > 0]
        elif frontend in ("tf", "raw"):
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
            if frontend in ("tf", "raw"):
                x = sample[:T]
                if x.shape[0] < T:
                    x = np.pad(x, (0, T - x.shape[0]))
                x = x / (np.max(np.abs(x)) + 1e-6)
                x = x.astype(np.float32)[None, :, None]
            elif frontend == "hybrid":
                x = sample.astype(np.float32)[None, :, :, None]
            else:
                x = sample.astype(np.float32)[None, :, :, None]
            yield [x]


def convert_to_tflite(
    model: tf.keras.Model,
    rep_data_gen,
    output_path: str,
) -> bytes:
    """Convert a Keras model to quantized TFLite with float32 I/O and INT8 internals.

    Args:
        model: Loaded Keras model.
        rep_data_gen: Callable returning an iterable of [input_tensor] for calibration.
        output_path: Path to save the .tflite model.

    Returns:
        Raw TFLite model bytes.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data_gen
    converter._experimental_new_quantizer = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    return tflite_model
