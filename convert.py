import argparse
import tensorflow as tf
import numpy as np
import os
from train import load_file_paths_from_directory
from utils.audio import load_audio_file, get_spectrogram_from_audio, sort_by_s2n, pick_random_spectrogram

def representative_data_gen(file_paths, num_mels, spec_width, chunk_duration, num_samples=100):
    """
    Generator for representative dataset for quantization.
    Yields spectrograms as input to the model.
    """
    count = 0
    for path in file_paths:
        # Use the same audio preprocessing as in training        
        audio_chunks = load_audio_file(path, sample_rate=16000, max_duration=30, chunk_duration=chunk_duration)
        spectrograms = [get_spectrogram_from_audio(chunk, 16000, mel_bins=num_mels, spec_width=spec_width) for chunk in audio_chunks]
        sorted_specs = sort_by_s2n(spectrograms, threshold=0.33)
        random_spec = pick_random_spectrogram(sorted_specs, num_samples=1)
        spec = random_spec[0] if isinstance(random_spec, list) else random_spec
        spec = np.expand_dims(spec, axis=-1).astype(np.float32)
        spec = np.expand_dims(spec, axis=0)  # add batch dimension     
        yield [spec]
        count += 1
        if count >= num_samples:
            break

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
    args = parser.parse_args()

    # Load model
    model = tf.keras.models.load_model(args.checkpoint_path, compile=False)

    # Load representative dataset file paths or randomly generate if not provided    
    if os.path.isdir(args.data_path_train):
        file_paths, _ = load_file_paths_from_directory(args.data_path_train)
        rep_data_gen = lambda: representative_data_gen(file_paths, args.num_mels, args.spec_width, args.chunk_duration, num_samples=args.num_samples)
    else:
        print(f"No training data directory provided, generating random representative dataset with {args.num_samples} samples.")
        # Generate random data for representative dataset
        np.random.seed(42)  # For reproducibility
        def rep_data_gen():
            for _ in range(args.num_samples):
                yield [np.random.rand(1, args.num_mels, args.spec_width, 1).astype(np.float32)]                

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data_gen
    # Keep float32 input/output, quantize everything else
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()

    # Save
    if len(args.output_path) == 0:
        args.output_path = os.path.splitext(args.checkpoint_path)[0] + "_quantized.tflite"
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {args.output_path}")
    
    # Sanity check
    if args.validate:
        print("Validating the TFLite model...")
        interpreter = tf.lite.Interpreter(model_path=args.output_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Check input/output types
        assert input_details[0]['dtype'] == np.float32, "Input type is not float32"
        assert output_details[0]['dtype'] == np.float32, "Output type is not float32"
        
        # Run inference on a sample
        sample_input = next(rep_data_gen())
        interpreter.set_tensor(input_details[0]['index'], sample_input[0])
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(f"Validation successful!")
        
        print(f"Sample input shape: {sample_input[0].shape}")
        print(f"Model output shape: {output_data.shape}")
        
    # Create validation data to run inference on the STM32N6570-DK
    # We'll generate 100 representative samples and store as .npz file
    if args.data_path_train:
        validation_data = []
        for _ in range(100):
            sample_input = next(rep_data_gen())
            validation_data.append(sample_input[0])
        
        validation_data = np.array(validation_data)
        validation_output_path = os.path.splitext(args.output_path)[0] + "_validation_data.npz"
        np.savez_compressed(validation_output_path, data=validation_data)
        print(f"Validation data saved to {validation_output_path}")

if __name__ == "__main__":
    main()