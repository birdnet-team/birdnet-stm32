"""CLI entry point for TFLite conversion."""

import argparse
import json
import os
import random

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from birdnet_stm32.audio.activity import pick_random_samples
from birdnet_stm32.conversion.quantize import convert_to_tflite, representative_data_gen
from birdnet_stm32.conversion.validate import validate_models
from birdnet_stm32.data.dataset import load_file_paths_from_directory
from birdnet_stm32.models.frontend import AudioFrontendLayer

random.seed(42)
np.random.seed(42)


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for conversion."""
    parser = argparse.ArgumentParser(
        description="Convert Keras model to quantized TFLite (float32 I/O, INT8 internal)."
    )
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to trained .keras model")
    parser.add_argument("--model_config", type=str, default="", help="Path to model config JSON")
    parser.add_argument("--output_path", type=str, default="", help="Output .tflite path")
    parser.add_argument("--data_path_train", type=str, default="", help="Training data directory for rep. dataset")
    parser.add_argument("--num_samples", type=int, default=1024, help="Representative dataset samples")
    parser.add_argument("--validate_samples", type=int, default=256, help="Validation samples")
    return parser.parse_args()


def main():
    """Convert a trained Keras model to quantized TFLite and validate."""
    args = get_args()

    # Resolve config path
    if not args.model_config:
        args.model_config = os.path.splitext(args.checkpoint_path)[0] + "_model_config.json"
    if not os.path.isfile(args.model_config):
        raise FileNotFoundError(f"Model config JSON not found: {args.model_config}")
    with open(args.model_config) as f:
        cfg = json.load(f)

    # Load model
    model = tf.keras.models.load_model(
        args.checkpoint_path, compile=False, custom_objects={"AudioFrontendLayer": AudioFrontendLayer}
    )
    print(f"Loaded model from {args.checkpoint_path}")

    # Build representative dataset generator
    if os.path.isdir(args.data_path_train):
        file_paths, _ = load_file_paths_from_directory(args.data_path_train)

        def rep_data_gen():
            return representative_data_gen(file_paths, cfg, num_samples=args.num_samples)

        def rep_data_gen_val():
            return representative_data_gen(file_paths, cfg, num_samples=args.validate_samples)
    else:
        print("No training data directory provided; generating random representative dataset.")

        def rep_data_gen(num_samples=args.num_samples):
            sr = int(cfg["sample_rate"])
            cd = cfg["chunk_duration"]
            T = int(sr * cd)
            spec_width = int(cfg["spec_width"])
            n_fft = int(cfg["fft_length"])
            frontend = cfg["audio_frontend"]
            num_mels = int(cfg["num_mels"])
            fft_bins = n_fft // 2 + 1
            for _ in tqdm(range(num_samples), desc="Random samples", unit="sample"):
                if frontend in ("precomputed", "librosa"):
                    yield [np.random.rand(1, num_mels, spec_width, 1).astype(np.float32)]
                elif frontend == "hybrid":
                    yield [np.random.rand(1, fft_bins, spec_width, 1).astype(np.float32)]
                else:
                    yield [np.random.randn(1, T, 1).astype(np.float32)]

        def rep_data_gen_val():
            return rep_data_gen(num_samples=args.validate_samples)

    # Output path
    if not args.output_path:
        args.output_path = os.path.splitext(args.checkpoint_path)[0] + "_quantized.tflite"

    # Convert
    convert_to_tflite(model, rep_data_gen, args.output_path)
    print(f"TFLite model saved to {args.output_path}")

    # Validate
    print("Validating TFLite vs. Keras outputs...")
    validate_models(model, args.output_path, rep_data_gen_val)

    # Save validation data
    validation_data = []
    for sample in rep_data_gen_val():
        validation_data.append(sample[0])
    validation_data = np.array(validation_data)
    if validation_data.shape[0] > 25:
        validation_data = pick_random_samples(validation_data, 25)
    val_path = os.path.splitext(args.output_path)[0] + "_validation_data.npz"
    np.savez_compressed(val_path, data=validation_data)
    print(f"Validation data saved to {val_path}")


if __name__ == "__main__":
    main()
