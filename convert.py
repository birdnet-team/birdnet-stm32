"""
TFLite conversion utilities for birdnet-stm32 models.

This script:
- Loads a trained Keras model (.keras) with the custom AudioFrontendLayer.
- Builds a representative dataset generator aligned with training preprocessing.
- Runs TFLite conversion with post-training quantization (PTQ) and float32 I/O.
- Optionally validates TFLite vs. Keras on a subset of samples and saves inputs.

Notes
- Representative dataset shapes match the selected audio_frontend:
  - 'raw'/'tf': [B, T, 1] where T = sample_rate * chunk_duration, normalized to [-1, 1]
  - 'hybrid':   [B, fft_bins, spec_width, 1] with fft_bins = n_fft//2 + 1 (linear spectrogram)
  - 'precomputed'/'librosa': [B, num_mels, spec_width, 1] (mel spectrogram)
- More representative samples can widen activation ranges (due to outliers) and
  reduce cosine similarity after PTQ. Prefer a calibrated but not overly permissive set.
"""

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
    Build a representative dataset generator aligned with training preprocessing.

    This generator yields one input tensor per iteration in the exact shape and
    normalization expected by the model input, suitable for TFLite PTQ calibration.

    Shapes per frontend:
    - 'raw' or 'tf':        [1, T, 1], where T = sample_rate * chunk_duration
                            Each chunk is peak-normalized to [-1, 1].
    - 'hybrid':             [1, fft_bins, spec_width, 1], where fft_bins = n_fft // 2 + 1
                            Values come from a linear-STFT magnitude (no mel).
    - 'precomputed'/'librosa': [1, num_mels, spec_width, 1], mel spectrogram magnitude.

    Args:
        file_paths (list[str]): Audio file paths to sample from. The generator selects
            a center chunk per file to avoid silence-only calibration when possible.
        cfg (dict): Training config keys used here:
            - 'sample_rate' (int)
            - 'num_mels' (int)
            - 'spec_width' (int)
            - 'chunk_duration' (float)
            - 'fft_length' (int)
            - 'audio_frontend' (str): 'precomputed' | 'librosa' | 'hybrid' | 'raw' | 'tf'
            - 'mag_scale' (str): 'none' | 'pcen' | 'pwl' (affects precomputed/librosa)
        num_samples (int): Maximum number of samples to draw across files.

    Yields:
        list[np.ndarray]: Single-element list containing the input tensor, with batch dimension.

    Notes:
        - Using many, highly diverse samples can widen activation ranges seen during PTQ,
          which may reduce cosine similarity. Consider capping extremes/outliers.
        - Ensure the normalization here matches training-time normalization, especially
          for 'raw'/'tf' frontends (peak-normalized to [-1, 1]).
    """
    sr = int(cfg["sample_rate"])
    num_mels = int(cfg["num_mels"])
    spec_width = int(cfg["spec_width"])
    cd = float(cfg["chunk_duration"])
    n_fft = int(cfg["fft_length"])
    frontend = cfg["audio_frontend"]
    mag_scale = cfg.get("mag_scale", "none")
    T = int(sr * cd)
    
    # Shuffle paths and limit to num_samples
    if len(file_paths) == 0:
        raise ValueError("No audio files found for representative dataset generation.")
    file_paths = random.sample(file_paths, num_samples)

    for path in tqdm(file_paths, desc="Generating rep. dataset", unit="file", dynamic_ncols=True):

        audio_chunks = load_audio_file(path, sample_rate=sr, max_duration=30, chunk_duration=cd)
        
        # Pick center chunk if multiple (simple heuristic to avoid very quiet heads/tails)
        if isinstance(audio_chunks, list) and len(audio_chunks) > 1:
            audio_chunks = [audio_chunks[len(audio_chunks) // 2]]
        elif isinstance(audio_chunks, np.ndarray) and audio_chunks.shape[0] > 1:
            audio_chunks = [audio_chunks[audio_chunks.shape[0] // 2]]

        # Convert to spectrogram if needed
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
                # Match training: per-chunk peak-normalization to [-1, 1]
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
    Cosine similarity between two flattened arrays.

    Args:
        a (np.ndarray): Flattened predictions from Keras.
        b (np.ndarray): Flattened predictions from TFLite.
        eps (float): Small constant to avoid division-by-zero.

    Returns:
        float: Cosine similarity in [-1, 1].
               If either vector has near-zero norm, returns 1.0 when both are near-zero,
               else 0.0.
    """
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an < eps or bn < eps:
        return 1.0 if an < eps and bn < eps else 0.0
    return float(np.dot(a, b) / (an * bn))

def _pearson(a, b, eps=1e-12):
    """
    Pearson correlation coefficient between two flattened arrays.

    Args:
        a (np.ndarray): Flattened predictions from Keras.
        b (np.ndarray): Flattened predictions from TFLite.
        eps (float): Small constant to guard against zero variance.

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
    Compare Keras vs. TFLite predictions over a set of samples.

    This runs the TFLite interpreter without delegates (builtin kernels only)
    to minimize numeric differences due to acceleration libraries and prints:
    - cosine similarity (mean/std/min/max)
    - MSE, MAE
    - Pearson correlation

    Args:
        keras_model (tf.keras.Model): Loaded Keras model (compiled=False is fine).
        tflite_model_path (str): Path to the converted .tflite model.
        rep_data_gen (Callable[[], Iterable[List[np.ndarray]]]): A zero-arg callable
            that returns an iterator/generator yielding [input_tensor] shaped exactly
            like the model's input.

    Returns:
        None. Prints summary statistics to stdout.
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
    Convert a trained Keras model (.keras) to TFLite and optionally validate it.

    Workflow:
        1) Load model and its JSON config (infer config path if omitted).
        2) Build a representative dataset generator aligned with training.
        3) Convert to TFLite with PTQ using float32 I/O (internal INT8 ops).
        4) Optionally validate TFLite vs. Keras on a subset of samples.
        5) Save a small .npz with validation inputs for future checks.

    CLI:
        --checkpoint_path  Path to the trained .keras model.
        --model_config     Path to JSON with training config; defaults to <checkpoint>_model_config.json.
        --output_path      Output .tflite path; defaults to <checkpoint>_quantized.tflite.
        --data_path_train  Directory of audio files for representative dataset (recommended).
        --num_samples      Number of representative samples (e.g., 512–2048 typical).
        --validate         Run post-conversion validation (recommended).
        --validate_samples Max samples to use for validation metrics.

    Notes:
        - Conversion uses float32 I/O with INT8 internal ops:
          converter.target_spec.supported_ops = [TFLITE_BUILTINS_INT8]
          converter.inference_input_type = converter.inference_output_type = float32
        - Ensure representative samples reflect deployment data distribution.
    """
    parser = argparse.ArgumentParser(description="Convert a trained Keras model (.keras) to TFLite with PTQ (float32 IO).")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to trained .keras model')
    parser.add_argument('--model_config', type=str, default='', help='Path to <checkpoint>_model_config.json. If empty, inferred from checkpoint_path.')
    parser.add_argument('--output_path', type=str, default='', help='Path to save .tflite model. Defaults to <checkpoint>_quantized.tflite')
    parser.add_argument('--data_path_train', type=str, default='', help='Path to training data directory for representative dataset.')
    parser.add_argument('--reps_per_file', type=int, default=4, help='How many representative samples to draw per file.')
    parser.add_argument('--num_samples', type=int, default=1024, help='Number of samples for representative dataset')
    parser.add_argument('--validate', action='store_true', default=True, help='Validate TFLite vs. Keras after conversion')
    parser.add_argument('--validate_samples', type=int, default=256, help='Max number of samples to validate')
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