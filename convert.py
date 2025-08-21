import argparse
import tensorflow as tf
import numpy as np
import os
import random
from train import load_file_paths_from_directory, AudioFrontendLayer, N6ConvFrontend
from utils.audio import load_audio_file, get_spectrogram_from_audio, sort_by_s2n, pick_random_samples

def representative_data_gen(file_paths, sample_rate, num_mels, spec_width, chunk_duration, audio_frontend, num_samples=100, snr_threshold=None, reps_per_file=4):
    """
    Build a representative dataset generator for TFLite PTQ calibration.

    - snr_threshold=None: no filtering (recommended for PTQ; captures full dynamic range)
    - reps_per_file: how many random samples to draw per file (improves coverage)
    """
    random.seed(42)
    np.random.seed(42)
    T = int(sample_rate * chunk_duration)
    yielded = 0

    for path in file_paths:
        if yielded >= num_samples:
            break

        audio_chunks = load_audio_file(path, sample_rate=sample_rate, max_duration=30, chunk_duration=chunk_duration)

        # Choose source sequences based on frontend and normalize to a list
        if audio_frontend == 'librosa':
            specs = [get_spectrogram_from_audio(ch, sample_rate, mel_bins=num_mels, spec_width=spec_width) for ch in audio_chunks]
            if snr_threshold is not None:
                specs = sort_by_s2n(specs, threshold=snr_threshold)
            # Keep only valid arrays
            pool = [s for s in specs if s is not None and np.size(s) > 0]
        elif audio_frontend == 'tf':
            if snr_threshold is not None:
                audio_chunks = sort_by_s2n(audio_chunks, threshold=snr_threshold)
            # Convert ndarray (N, chunk_len) to list-of-1D arrays
            if isinstance(audio_chunks, np.ndarray):
                pool = [audio_chunks[i] for i in range(audio_chunks.shape[0])]
            else:
                pool = list(audio_chunks)
            pool = [c for c in pool if c is not None and np.size(c) > 0]
        else:
            raise ValueError("Invalid audio frontend. Choose 'librosa' or 'tf'.")

        if len(pool) == 0:
            continue

        # Draw multiple reps per file for better coverage
        k = min(reps_per_file, len(pool))
        sel_idx = np.random.choice(len(pool), size=k, replace=len(pool) < k)
        for j in sel_idx:
            if yielded >= num_samples:
                break
            sample = pool[j]
            # Ensure proper input shape
            if audio_frontend == 'tf':
                x = sample[:T]
                if x.shape[0] < T:
                    x = np.pad(x, (0, T - x.shape[0]))
                x = x.astype(np.float32)[None, :, None]
            else:
                x = sample.astype(np.float32)[None, :, :, None]
            yield [x]
            yielded += 1

def _cosine(a, b, eps=1e-12):
    """
    Compute cosine similarity between two 1D arrays.

    Args:
        a (np.ndarray): Flattened predictions from Keras.
        b (np.ndarray): Flattened predictions from TFLite.
        eps (float): Small value to avoid division by zero.

    Returns:
        float: Cosine similarity in [-1, 1]. Returns 1.0 if both vectors are near-zero,
               else 0.0 if only one vector is near-zero.
    """
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an < eps or bn < eps:
        return 1.0 if an < eps and bn < eps else 0.0
    return float(np.dot(a, b) / (an * bn))

def _pearson(a, b, eps=1e-12):
    """
    Compute Pearson correlation coefficient between two 1D arrays.

    Args:
        a (np.ndarray): Flattened predictions from Keras.
        b (np.ndarray): Flattened predictions from TFLite.
        eps (float): Small value to guard against zero variance.

    Returns:
        float: Pearson's r in [-1, 1]. Returns 1.0 if both vectors have near-zero variance.
    """
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < eps:
        return 1.0
    return float(np.dot(a, b) / denom)

def validate_models(keras_model, tflite_model_path, rep_data_gen, num_samples=50):
    """
    Validate TFLite vs. Keras outputs on representative samples.

    Runs both models on up to num_samples inputs from rep_data_gen() and reports:
      - cosine similarity
      - mean squared error (mse)
      - mean absolute error (mae)
      - Pearson correlation coefficient (pearson_r)

    Assumptions:
      - The TFLite model uses float32 I/O (converter.inference_input/output_type=float32).
      - Internal ops may be int8 (PTQ), which this compares against float32 Keras.

    Args:
        keras_model (tf.keras.Model): Loaded Keras model (.h5 or .keras).
        tflite_model_path (str): Path to the converted .tflite file.
        rep_data_gen (Callable[[], Iterator[list[np.ndarray]]]): Generator factory that
            yields single-element lists with one input batch tensor per iteration.
        num_samples (int): Max number of samples to evaluate.
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    assert in_det['dtype'] == np.float32, "Input type is not float32"
    assert out_det['dtype'] == np.float32, "Output type is not float32"

    cos_list, mse_list, mae_list, pcc_list = [], [], [], []
    cos_sm_list, mse_sm_list, mae_sm_list, pcc_sm_list = [], [], [], []

    gen = rep_data_gen()
    n_eval = 0
    for i in range(num_samples):
        try:
            sample = next(gen)[0]  # (1, ... )
        except StopIteration:
            break

        # Keras forward
        yk = keras_model(sample, training=False).numpy()
        # TFLite forward
        interpreter.set_tensor(in_det['index'], sample.astype(np.float32))
        interpreter.invoke()
        yt = interpreter.get_tensor(out_det['index'])

        a = yk.reshape(-1).astype(np.float64)
        b = yt.reshape(-1).astype(np.float64)

        cos_list.append(_cosine(a, b))
        mse_list.append(float(np.mean((a - b) ** 2)))
        mae_list.append(float(np.mean(np.abs(a - b))))
        pcc_list.append(_pearson(a, b))

        n_eval += 1

    print(f"Validated on {n_eval} samples.")
    if n_eval == 0:
        return

    def _summ(name, vals):
        if vals:
            print(f"{name}: mean={np.mean(vals):.6f}  std={np.std(vals):.6f}  min={np.min(vals):.6f}  max={np.max(vals):.6f}")

    _summ("cosine", cos_list)
    _summ("mse", mse_list)
    _summ("mae", mae_list)
    _summ("pearson_r", pcc_list)

def main():
    """
    CLI entry point: load a trained Keras model, convert to TFLite with PTQ,
    optionally validate fidelity, and save representative samples.

    Steps:
      1) Load the .h5/.keras model (register AudioFrontendLayer if needed).
      2) Build a representative dataset:
           - From --data_path_train (preferred), or random data as fallback.
      3) Convert to TFLite with Optimize.DEFAULT, float32 I/O (int8 internals allowed).
      4) Save the .tflite model to --output_path (or alongside checkpoint).
      5) If --validate, run Keras vs. TFLite comparison on a subset of samples.
      6) Save all representative samples used to <model>_validation_data.npz.
    """
    parser = argparse.ArgumentParser(description="Convert a trained Keras model to TFLite with PTQ (float32 IO, int8 internals).")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to trained .h5 model')
    parser.add_argument('--output_path', type=str, default='', help='Path to save .tflite model. If not provided, will save to same directory as checkpoint.')
    parser.add_argument('--data_path_train', type=str, default='', help='Path to training data for representative dataset. If not provided, will generate random data.')
    parser.add_argument('--snr_threshold', type=float, default=None, help='SNR threshold for representative data (None disables filtering; recommended).')
    parser.add_argument('--reps_per_file', type=int, default=4, help='How many representative samples to draw per file.')
    parser.add_argument('--num_samples', type=int, default=1024, help='Number of samples for representative dataset')
    parser.add_argument('--sample_rate', type=int, default=22050, help='Sample rate for audio files (should match training)')
    parser.add_argument('--num_mels', type=int, default=64, help='Number of mel bins (should match training)')
    parser.add_argument('--spec_width', type=int, default=128, help='Spectrogram width (should match training)')
    parser.add_argument('--chunk_duration', type=int, default=3, help='Audio chunk duration in seconds (should match training)')
    parser.add_argument('--validate', action='store_true', default=True, help='Validate the model after conversion')
    parser.add_argument('--audio_frontend', type=str, choices=['librosa', 'tf'], default='librosa', help='Input type for representative data')
    args = parser.parse_args()

    # Load model
    model = tf.keras.models.load_model(args.checkpoint_path, 
                                       compile=False, 
                                       custom_objects={'N6ConvFrontend': N6ConvFrontend} if args.audio_frontend == 'tf' else None)
    print(f"Loaded model from {args.checkpoint_path}")

    # Representative dataset generator (from files or random)
    if os.path.isdir(args.data_path_train):
        file_paths, _ = load_file_paths_from_directory(args.data_path_train)
        rep_data_gen = lambda: representative_data_gen(
            file_paths, args.sample_rate, args.num_mels, args.spec_width, args.chunk_duration, args.audio_frontend,
            num_samples=args.num_samples, snr_threshold=args.snr_threshold, reps_per_file=args.reps_per_file
        )
    else:
        print(f"No training data directory provided, generating random representative dataset with {args.num_samples} samples.")
        def rep_data_gen():
            sample_rate = args.sample_rate
            target_len = sample_rate * args.chunk_duration
            for _ in range(args.num_samples):
                if args.audio_frontend == 'librosa':
                    yield [np.random.rand(1, args.num_mels, args.spec_width, 1).astype(np.float32)]
                else:
                    yield [np.random.randn(1, target_len, 1).astype(np.float32)]

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data_gen

    # Use new quantizer for stability across checkpoints
    converter._experimental_new_quantizer = True  # uses MLIR quantizer

    # Keep float32 IO; quantize internals
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
    
    # Validation (wrapped)
    if args.validate:
        print("Validating the TFLite model...")
        validate_models(model, args.output_path, rep_data_gen, num_samples=min(args.num_samples, 100))
        
    # Always save representative samples as .npz
    validation_data = []
    gen_for_save = rep_data_gen()
    for _ in range(args.num_samples):
        try:
            sample_input = next(gen_for_save)
        except StopIteration:
            break
        validation_data.append(sample_input[0])
    validation_data = np.array(validation_data)
    validation_output_path = os.path.splitext(args.output_path)[0] + "_validation_data.npz"
    np.savez_compressed(validation_output_path, data=validation_data)
    print(f"Validation data saved to {validation_output_path}")

if __name__ == "__main__":
    main()