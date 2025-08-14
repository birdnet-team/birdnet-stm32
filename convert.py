import argparse
import tensorflow as tf
import numpy as np
import os
from train import load_file_paths_from_directory, AudioFrontendLayer 
from utils.audio import load_audio_file, get_spectrogram_from_audio, sort_by_s2n, pick_random_samples

def representative_data_gen(file_paths, sample_rate, num_mels, spec_width, chunk_duration, audio_frontend, num_samples=100):
    """
    Generator for representative dataset for quantization.
    - librosa: yields (1, num_mels, spec_width, 1)
    - tf: yields (1, chunk_samples, 1)
    """
    count = 0
    for path in file_paths:
        audio_chunks = load_audio_file(path, sample_rate=sample_rate, max_duration=30, chunk_duration=chunk_duration)

        if audio_frontend == 'librosa':
            spectrograms = [get_spectrogram_from_audio(chunk, sample_rate, mel_bins=num_mels, spec_width=spec_width) for chunk in audio_chunks]
            sorted_specs = sort_by_s2n(spectrograms, threshold=0.5)                
            random_spec = pick_random_samples(sorted_specs, num_samples=1)
            sample = random_spec[0] if isinstance(random_spec, list) else random_spec
        elif audio_frontend == 'tf':
            sorted_chunks = sort_by_s2n(audio_chunks, threshold=0.5)
            random_chunk = pick_random_samples(sorted_chunks, num_samples=1)
            sample = random_chunk[0] if isinstance(random_chunk, list) else random_chunk
        else:
            raise ValueError("Invalid audio frontend. Choose 'librosa' or 'tf'.")

        sample = np.expand_dims(sample, axis=-1)
        sample = np.expand_dims(sample, axis=0)  # Add batch dimension
        
        yield [sample.astype(np.float32)]

        count += 1
        if count >= num_samples:
            break

def _cosine(a, b, eps=1e-12):
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an < eps or bn < eps:
        return 1.0 if an < eps and bn < eps else 0.0
    return float(np.dot(a, b) / (an * bn))

def _pearson(a, b, eps=1e-12):
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < eps:
        return 1.0
    return float(np.dot(a, b) / denom)

def validate_models(keras_model, tflite_model_path, rep_data_gen, num_samples=50):
    """
    Compare Keras vs TFLite outputs on rep_data_gen() samples.
    Prints cosine/MSE/MAE/Pearson stats.
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
    parser = argparse.ArgumentParser(description="Convert Keras model to fully quantized TFLite with float32 IO")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to trained .h5 model')
    parser.add_argument('--output_path', type=str, default='', help='Path to save .tflite model. If not provided, will save to same directory as checkpoint.')
    parser.add_argument('--data_path_train', type=str, default='', help='Path to training data for representative dataset. If not provided, will generate random data.')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples for representative dataset')
    parser.add_argument('--num_mels', type=int, default=64, help='Number of mel bins (should match training)')
    parser.add_argument('--spec_width', type=int, default=128, help='Spectrogram width (should match training)')
    parser.add_argument('--chunk_duration', type=int, default=3, help='Audio chunk duration in seconds (should match training)')
    parser.add_argument('--validate', action='store_true', default=True, help='Validate the model after conversion')
    parser.add_argument('--audio_frontend', type=str, choices=['librosa', 'tf'], default='librosa', help='Input type for representative data')
    args = parser.parse_args()

    # Load model
    model = tf.keras.models.load_model(args.checkpoint_path, 
                                       compile=False, 
                                       custom_objects={'AudioFrontendLayer': AudioFrontendLayer} if args.audio_frontend == 'tf' else None)
    print(f"Loaded model from {args.checkpoint_path}")

    # Representative dataset generator (from files or random)
    if os.path.isdir(args.data_path_train):
        file_paths, _ = load_file_paths_from_directory(args.data_path_train)
        rep_data_gen = lambda: representative_data_gen(
            file_paths, 16000, args.num_mels, args.spec_width, args.chunk_duration, args.audio_frontend, num_samples=args.num_samples
        )
    else:
        print(f"No training data directory provided, generating random representative dataset with {args.num_samples} samples.")
        def rep_data_gen():
            sample_rate = 16000
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
        validate_models(model, args.output_path, rep_data_gen, num_samples=min(args.num_samples, 50))
        
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