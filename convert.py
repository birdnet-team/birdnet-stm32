import os

import argparse
import tensorflow as tf
import numpy as np
import random
import json
from tqdm import tqdm

from train import load_file_paths_from_directory, AudioFrontendLayer
from utils.audio import load_audio_file, get_spectrogram_from_audio, sort_by_s2n, pick_random_samples

# Random seed for reproducibility
random.seed(42)
np.random.seed(42)

def representative_data_gen(file_paths, cfg, num_samples=100):
    """
    Build a representative dataset generator from training configuration.

    Uses training-time settings to produce inputs consistent with the model's
    expected frontend and shape for post-training quantization (PTQ).

    Args:
        file_paths (list[str]): Audio file paths for sampling.
        cfg (dict): Training config with keys:
            - sample_rate (int)
            - num_mels (int)
            - spec_width (int)
            - chunk_duration (float)
            - fft_length (int)
            - audio_frontend (str): 'precomputed' | 'librosa' | 'hybrid' | 'raw' | 'tf'
            - mag_scale (str): 'none' | 'pcen' | 'pwl'
        num_samples (int): Maximum number of samples to yield.

    Yields:
        list[np.ndarray]: A single-element list containing one input tensor
            with batch dimension (shape depends on frontend).
    """
    sr = int(cfg["sample_rate"])
    num_mels = int(cfg["num_mels"])
    spec_width = int(cfg["spec_width"])
    cd = float(cfg["chunk_duration"])
    n_fft = int(cfg["fft_length"])
    frontend = cfg["audio_frontend"]
    mag_scale = cfg.get("mag_scale", "none")
    T = int(sr * cd)
    yielded = 0
    
    # Suffle paths and limit to num_samples
    if len(file_paths) == 0:
        raise ValueError("No audio files found for representative dataset generation.")
    file_paths = random.sample(file_paths, num_samples)

    for path in tqdm(file_paths, desc="Generating rep. dataset", unit="file", dynamic_ncols=True):

        audio_chunks = load_audio_file(path, sample_rate=sr, max_duration=30, chunk_duration=cd)
        
        # Assert correct chunk shape
        if isinstance(audio_chunks, np.ndarray):
            assert audio_chunks.ndim == 2 and audio_chunks.shape[1] == T, f"Expected shape (num_chunks, {T}), got {audio_chunks.shape}"
        else:
            for ch in audio_chunks:
                assert ch.ndim == 1 and ch.shape[0] == T, f"Expected shape ({T},), got {ch.shape}"
                
        # Pick 1 random chunk if multiple
        if isinstance(audio_chunks, np.ndarray) and audio_chunks.shape[0] > 1:
            audio_chunks = audio_chunks[random.randint(0, audio_chunks.shape[0] - 1)][None, :]
        elif isinstance(audio_chunks, list) and len(audio_chunks) > 1:
            audio_chunks = [audio_chunks[random.randint(0, len(audio_chunks) - 1)]]

        # Convert to spectrogram if needed (still supports multiple chunks, but we only have 1 for now)
        if frontend in ('precomputed', 'librosa'):
            specs = [get_spectrogram_from_audio(ch, sample_rate=sr, n_fft=n_fft, mel_bins=num_mels, spec_width=spec_width, mag_scale=mag_scale) for ch in audio_chunks]
            pool = [s for s in specs if s is not None and np.size(s) > 0]
        elif frontend == 'hybrid':
            specs = [get_spectrogram_from_audio(ch, sample_rate=sr, n_fft=n_fft, mel_bins=-1, spec_width=spec_width) for ch in audio_chunks]
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

        for i in range(len(pool)):
            sample = pool[i]
            if frontend in ('tf', 'raw'):
                x = sample[:T]
                if x.shape[0] < T:
                    x = np.pad(x, (0, T - x.shape[0]))
                x = x / (np.max(np.abs(x)) + 1e-6)
                x = x.astype(np.float32)[None, :, None]
            elif frontend == 'hybrid':
                fft_bins = n_fft // 2 + 1
                x = sample.astype(np.float32)[None, :, :, None]
                assert x.shape[1] == fft_bins, f"Expected fft_bins={fft_bins}, got {x.shape[1]}"
            else:
                x = sample.astype(np.float32)[None, :, :, None]
            yield [x]
            
def _cosine(a, b, eps=1e-12):
    """
    Compute cosine similarity between two flattened arrays.

    Args:
        a (np.ndarray): Flattened predictions from Keras.
        b (np.ndarray): Flattened predictions from TFLite.
        eps (float): Small value to avoid division by zero.

    Returns:
        float: Cosine similarity in [-1, 1]. If either vector has near-zero norm,
            returns 1.0 when both are near-zero, otherwise 0.0.
    """
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an < eps or bn < eps:
        return 1.0 if an < eps and bn < eps else 0.0
    return float(np.dot(a, b) / (an * bn))

def _pearson(a, b, eps=1e-12):
    """
    Compute Pearson correlation coefficient between two flattened arrays.

    Args:
        a (np.ndarray): Flattened predictions from Keras.
        b (np.ndarray): Flattened predictions from TFLite.
        eps (float): Small value to guard against zero variance.

    Returns:
        float: Pearson's r in [-1, 1]. If both vectors have near-zero variance,
            returns 1.0.
    """
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < eps:
        return 1.0
    return float(np.dot(a, b) / denom)

def validate_models(keras_model, tflite_model_path, rep_data_gen):
    """
    Validate TFLite vs. Keras predictions on representative samples.
    """
    # Create interpreter without delegates (disables XNNPACK explicitly)
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path, 
                                      experimental_delegates=None, 
                                      num_threads=1)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    assert in_det['dtype'] == np.float32, "Input type is not float32"
    assert out_det['dtype'] == np.float32, "Output type is not float32"
    
    print(f"TFLite input shape: {in_det['shape']}, output shape: {out_det['shape']}")

    cos_list, mse_list, mae_list, pcc_list = [], [], [], []
    
    for sample in rep_data_gen():

        # Keras forward
        yk = keras_model(sample[0], training=False).numpy()
        # TFLite forward
        interpreter.set_tensor(in_det['index'], sample[0].astype(np.float32))
        interpreter.invoke()
        yt = interpreter.get_tensor(out_det['index'])

        a = yk.reshape(-1).astype(np.float64)
        b = yt.reshape(-1).astype(np.float64)

        cos_list.append(_cosine(a, b))
        mse_list.append(float(np.mean((a - b) ** 2)))
        mae_list.append(float(np.mean(np.abs(a - b))))
        pcc_list.append(_pearson(a, b))

    def _summ(name, vals):
        if vals:
            print(f"{name}: mean={np.mean(vals):.6f}  std={np.std(vals):.6f}  min={np.min(vals):.6f}  max={np.max(vals):.6f}")

    _summ("cosine", cos_list)
    _summ("mse", mse_list)
    _summ("mae", mae_list)
    _summ("pearson_r", pcc_list)

def main():
    """
    Convert a trained Keras model (.keras) to TFLite with PTQ and optional validation.

    Workflow:
        1) Load model and its JSON config.
        2) Build representative dataset generator from training config.
        3) Convert to TFLite using float32 I/O and default optimizations.
        4) Optionally validate TFLite vs. Keras on representative samples.
        5) Save a small .npz with validation inputs for future checks.

    Args:
        None. Command-line arguments are parsed via argparse.

    Returns:
        None. Writes converted model and validation data to disk.
    """
    parser = argparse.ArgumentParser(description="Convert a trained Keras model (.keras) to TFLite with PTQ (float32 IO).")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to trained .keras model')
    parser.add_argument('--model_config', type=str, default='', help='Path to <checkpoint>_model_config.json. If empty, inferred from checkpoint_path.')
    parser.add_argument('--output_path', type=str, default='', help='Path to save .tflite model. Defaults to <checkpoint>_quantized.tflite')
    parser.add_argument('--data_path_train', type=str, default='', help='Path to training data directory for representative dataset.')
    parser.add_argument('--reps_per_file', type=int, default=4, help='How many representative samples to draw per file.')
    parser.add_argument('--num_samples', type=int, default=256, help='Number of samples for representative dataset')
    parser.add_argument('--validate', action='store_true', default=True, help='Validate TFLite vs. Keras after conversion')
    parser.add_argument('--validate_samples', type=int, default=128, help='Max number of samples to validate')
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
        rep_data_gen = lambda: representative_data_gen(file_paths, cfg, num_samples=args.num_samples)
        rep_data_gen_val = lambda: representative_data_gen(file_paths, cfg, num_samples=args.validate_samples)
    else:
        print(f"No training data directory provided, generating random representative dataset with {args.num_samples} samples.")
        def rep_data_gen(num_samples=args.num_samples):
            sr = int(cfg["sample_rate"]); cd = cfg["chunk_duration"]; T = int(sr * cd)
            spec_width = int(cfg["spec_width"]); n_fft = int(cfg["fft_length"])
            frontend = cfg["audio_frontend"]; num_mels = int(cfg["num_mels"])
            fft_bins = n_fft // 2 + 1
            for _ in tqdm(range(num_samples), desc="Generating random samples", unit="sample", dynamic_ncols=True):
                if frontend in ('precomputed', 'librosa'):
                    yield [np.random.rand(1, num_mels, spec_width, 1).astype(np.float32)]
                elif frontend == 'hybrid':
                    # CONSISTENT: [B, fft_bins, T, 1]
                    yield [np.random.rand(1, fft_bins, spec_width, 1).astype(np.float32)]
                else:  # tf/raw
                    yield [np.random.randn(1, T, 1).astype(np.float32)]
        rep_data_gen_val = lambda: rep_data_gen(num_samples=args.validate_samples)

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data_gen
    converter._experimental_new_quantizer = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
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

    # Optional validation
    if args.validate:
        print("Validating TFLite vs. Keras outputs...")
        validate_models(
            keras_model=model,
            tflite_model_path=args.output_path if args.output_path else os.path.splitext(args.checkpoint_path)[0] + "_quantized.tflite",
            rep_data_gen=rep_data_gen_val
        )
        
    # Always save representative samples as .npz
    validation_data = []
    for sample in rep_data_gen_val():
        validation_data.append(sample[0])
    validation_data = np.array(validation_data)
    if validation_data.shape[0] > 25:
        validation_data = pick_random_samples(validation_data, 25)
    validation_output_path = os.path.splitext(args.output_path)[0] + "_validation_data.npz"
    np.savez_compressed(validation_output_path, data=validation_data)
    print(f"Validation data saved to {validation_output_path}")

if __name__ == "__main__":
    main()