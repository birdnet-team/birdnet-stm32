import argparse
import tensorflow as tf
import numpy as np
import os
import random
import json

from train import load_file_paths_from_directory, AudioFrontendLayer
from utils.audio import load_audio_file, get_spectrogram_from_audio, get_linear_spectrogram_from_audio, sort_by_s2n, pick_random_samples

def representative_data_gen(file_paths, cfg, num_samples=100, reps_per_file=4):
    """
    Representative dataset generator using training config.
    cfg keys used: sample_rate, num_mels, spec_width, chunk_duration, fft_length, audio_frontend, mag_scale
    """
    random.seed(42)
    np.random.seed(42)
    sr = int(cfg["sample_rate"])
    num_mels = int(cfg["num_mels"])
    spec_width = int(cfg["spec_width"])
    cd = int(cfg["chunk_duration"])
    n_fft = int(cfg["fft_length"])
    frontend = cfg["audio_frontend"]
    mag_scale = cfg.get("mag_scale", "none")
    T = int(sr * cd)
    yielded = 0

    for path in file_paths:
        if yielded >= num_samples:
            break

        audio_chunks = load_audio_file(path, sample_rate=sr, max_duration=30, chunk_duration=cd)

        if frontend in ('precomputed', 'librosa'):
            specs = [get_spectrogram_from_audio(ch, sample_rate=sr, n_fft=n_fft, mel_bins=num_mels, spec_width=spec_width, mag_scale=mag_scale) for ch in audio_chunks]
            pool = [s for s in specs if s is not None and np.size(s) > 0]
        elif frontend == 'hybrid':
            specs = [get_linear_spectrogram_from_audio(ch, sample_rate=sr, n_fft=n_fft, spec_width=spec_width, power=2.0) for ch in audio_chunks]
            pool = [s for s in specs if s is not None and np.size(s) > 0]
        elif frontend in ('tf', 'raw'):
            if isinstance(audio_chunks, np.ndarray):
                pool = [audio_chunks[i] for i in range(audio_chunks.shape[0])]
            else:
                pool = list(audio_chunks)
            pool = [c for c in pool if c is not None and np.size(c) > 0]
        else:
            raise ValueError("Invalid audio frontend. Choose 'precomputed', 'hybrid', or 'tf/raw'.")

        if len(pool) == 0:
            continue

        k = min(reps_per_file, len(pool))
        sel_idx = np.random.choice(len(pool), size=k, replace=len(pool) < k)
        for j in sel_idx:
            if yielded >= num_samples:
                break
            sample = pool[j]
            if frontend in ('tf', 'raw'):
                x = sample[:T]
                if x.shape[0] < T:
                    x = np.pad(x, (0, T - x.shape[0]))
                x = x.astype(np.float32)[None, :, None]
            elif frontend == 'hybrid':
                fft_bins = n_fft // 2 + 1
                x = sample.astype(np.float32)[None, :, :, None]
                assert x.shape[1] == fft_bins, f"Expected fft_bins={fft_bins}, got {x.shape[1]}"
            else:
                x = sample.astype(np.float32)[None, :, :, None]
            yield [x]
            yielded += 1

def main():
    """
    Load a .keras model, read its JSON config, build a matching representative dataset, and convert to TFLite.
    """
    parser = argparse.ArgumentParser(description="Convert a trained Keras model (.keras) to TFLite with PTQ (float32 IO).")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to trained .keras model')
    parser.add_argument('--model_config', type=str, default='', help='Path to <checkpoint>_model_config.json. If empty, inferred from checkpoint_path.')
    parser.add_argument('--output_path', type=str, default='', help='Path to save .tflite model. Defaults to <checkpoint>_quantized.tflite')
    parser.add_argument('--data_path_train', type=str, default='', help='Path to training data directory for representative dataset.')
    parser.add_argument('--reps_per_file', type=int, default=4, help='How many representative samples to draw per file.')
    parser.add_argument('--num_samples', type=int, default=1024, help='Number of samples for representative dataset')
    args = parser.parse_args()

    # Infer model_config path if not provided
    if not args.model_config:
        args.model_config = os.path.splitext(args.checkpoint_path)[0] + "_model_config.json"
    if not os.path.isfile(args.model_config):
        raise FileNotFoundError(f"Model config JSON not found: {args.model_config}")
    with open(args.model_config, "r") as f:
        cfg = json.load(f)

    # Load model with custom layers
    model = tf.keras.models.load_model(
        args.checkpoint_path,
        compile=False,
        custom_objects={'AudioFrontendLayer': AudioFrontendLayer}
    )
    print(f"Loaded model from {args.checkpoint_path}")

    # Build representative dataset generator
    if os.path.isdir(args.data_path_train):
        file_paths, _ = load_file_paths_from_directory(args.data_path_train)
        rep_data_gen = lambda: representative_data_gen(
            file_paths, cfg, num_samples=args.num_samples, reps_per_file=args.reps_per_file
        )
    else:
        print(f"No training data directory provided, generating random representative dataset with {args.num_samples} samples.")
        def rep_data_gen():
            sr = int(cfg["sample_rate"]); cd = int(cfg["chunk_duration"]); T = sr * cd
            spec_width = int(cfg["spec_width"]); n_fft = int(cfg["fft_length"])
            frontend = cfg["audio_frontend"]; num_mels = int(cfg["num_mels"])
            fft_bins = n_fft // 2 + 1
            for _ in range(args.num_samples):
                if frontend in ('precomputed', 'librosa'):
                    yield [np.random.rand(1, num_mels, spec_width, 1).astype(np.float32)]
                elif frontend == 'hybrid':
                    yield [np.random.rand(1, fft_bins, spec_width, 1).astype(np.float32)]
                else:  # tf/raw
                    yield [np.random.randn(1, T, 1).astype(np.float32)]

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data_gen
    converter._experimental_new_quantizer = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()

    # Save TFLite
    if len(args.output_path) == 0:
        args.output_path = os.path.splitext(args.checkpoint_path)[0] + "_quantized.tflite"
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {args.output_path}")

    # Optional quick validation if data provided (Keras vs TFLite) could be added here.
    # (Your previous validate_models can be reused with cfg-driven rep_data_gen if needed.)

if __name__ == "__main__":
    main()