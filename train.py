import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import math

from utils.audio import load_audio_file, get_spectrogram_from_audio, sort_by_s2n, pick_random_spectrogram, plot_spectrogram

# Mute TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# random seed for reproducibility
np.random.seed(42)

# Enable dynamic GPU memory allocation
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"Could not set memory growth: {e}")
    
def dataset_sanity_check(file_paths, classes, sample_rate=16000, max_duration=30, chunk_duration=3, spec_width=128, mel_bins=64):
    """
    Load and plot spectrograms from a few audio files for sanity checking.

    Args:
        file_paths (list of str): List of audio file paths.
        classes (list of str): List of class names.
        sample_rate (int): Sample rate for audio loading.
        max_duration (int): Maximum duration per file.
        chunk_duration (int): Duration per chunk.
        spec_width (int): Spectrogram width.
        mel_bins (int): Number of mel bins.

    Returns:
        None
    """
    # shuffle file paths
    np.random.shuffle(file_paths)
    for i, path in enumerate(file_paths[:5]):  # Check first 5 files
        audio_chunks = load_audio_file(path, sample_rate, max_duration, chunk_duration)
        if len(audio_chunks) == 0:
            print(f"File {path} has no valid audio chunks.")
            continue
        
        spectrograms = [get_spectrogram_from_audio(chunk, sample_rate, mel_bins=mel_bins, spec_width=spec_width) for chunk in audio_chunks]
        sorted_specs = sort_by_s2n(spectrograms, threshold=0.33)
        
        if len(sorted_specs) == 0:
            print(f"No valid spectrograms found for file {path}.")
            continue
        
        random_spec = pick_random_spectrogram(sorted_specs, num_samples=1)
        plot_spectrogram(random_spec, title=f"Spectrogram for {os.path.basename(path)} - Class: {path.split('/')[-2]}")
    
def load_file_paths_from_directory(directory, classes=None):
    """
    Recursively load all .wav file paths from a directory, optionally filtering by class.

    Args:
        directory (str): Root directory to search.
        classes (list of str, optional): List of class names to filter.

    Returns:
        tuple: (list of file paths, list of classes)
    """
    # Load all audio file paths from the specified directory
    file_paths = []
    classes = tf.io.gfile.listdir(directory) if classes is None else classes
    for root, _, files in tf.io.gfile.walk(directory):
        for file in files:
            if file.endswith('.wav') and (classes is None or tf.io.gfile.join(root, file).split('/')[-2] in classes):
                file_paths.append(tf.io.gfile.join(root, file))
                
    # Shuffle the file paths
    np.random.shuffle(file_paths)            
    
    return file_paths, sorted(classes)

def data_generator(file_paths, classes, batch_size=32, sample_rate=16000, max_duration=30, chunk_duration=3, spec_width=128, mixup_alpha=0.2, mixup_probability=0.25, mel_bins=48):
    """
    Generator that yields batches of spectrograms and one-hot labels, with optional mixup augmentation.

    Args:
        file_paths (list of str): List of audio file paths.
        classes (list of str): List of class names.
        batch_size (int): Number of samples per batch.
        sample_rate (int): Sample rate for audio loading.
        max_duration (int): Maximum duration per file.
        chunk_duration (int): Duration per chunk.
        spec_width (int): Spectrogram width.
        mixup_alpha (float): Mixup beta distribution parameter.
        mixup_probability (float): Fraction of batch to apply mixup.
        mel_bins (int): Number of mel bins.

    Yields:
        tuple: (batch_specs, batch_labels)
    """
    while True:
        idxs = np.random.permutation(len(file_paths))
        for batch_start in range(0, len(idxs), batch_size):
            batch_idxs = idxs[batch_start:batch_start + batch_size]
            batch_specs = []
            batch_labels = []
            for idx in batch_idxs:
                path = file_paths[idx]
                audio_chunks = load_audio_file(path, sample_rate, max_duration, chunk_duration)
                spectrograms = [get_spectrogram_from_audio(chunk, sample_rate, mel_bins=mel_bins, spec_width=spec_width) for chunk in audio_chunks]
                sorted_specs = sort_by_s2n(spectrograms, threshold=0.33)
                random_spec = pick_random_spectrogram(sorted_specs, num_samples=1)
                spec = random_spec[0] if isinstance(random_spec, list) else random_spec

                label_str = path.split('/')[-2]
                one_hot_label = tf.one_hot(classes.index(label_str), depth=len(classes)).numpy()

                spec = np.expand_dims(spec, axis=-1)
                batch_specs.append(spec.astype(np.float32))
                batch_labels.append(one_hot_label.astype(np.float32))

            batch_specs = np.stack(batch_specs)
            batch_labels = np.stack(batch_labels)

            # Batch-level mixup: only a fraction of samples per batch
            if mixup_alpha > 0 and mixup_probability > 0:
                num_mix = int(batch_specs.shape[0] * mixup_probability)
                if num_mix > 0:
                    mix_indices = np.random.choice(batch_specs.shape[0], size=num_mix, replace=False)
                    permuted_indices = np.random.permutation(batch_specs.shape[0])
                    lam = np.random.beta(mixup_alpha, mixup_alpha, size=(num_mix, 1, 1, 1))
                    lam_labels = lam.squeeze(axis=(2, 3))  # shape (num_mix, 1)

                    # Mixup only selected samples
                    batch_specs[mix_indices] = lam * batch_specs[mix_indices] + (1 - lam) * batch_specs[permuted_indices[mix_indices]]
                    batch_labels[mix_indices] = lam_labels * batch_labels[mix_indices] + (1 - lam_labels) * batch_labels[permuted_indices[mix_indices]]

            yield batch_specs, batch_labels

def load_dataset(file_paths, classes, batch_size=32, spec_width=128, mel_bins=48, **kwargs):
    """
    Create a tf.data.Dataset from a data generator for spectrograms and labels.

    Args:
        file_paths (list of str): List of audio file paths.
        classes (list of str): List of class names.
        batch_size (int): Number of samples per batch.
        spec_width (int): Spectrogram width.
        mel_bins (int): Number of mel bins.
        **kwargs: Additional arguments for data_generator.

    Returns:
        tf.data.Dataset: Prefetched, repeated dataset.
    """
    output_signature = (
        tf.TensorSpec(shape=(None, mel_bins, spec_width, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, len(classes)), dtype=tf.float32)
    )
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(file_paths, classes, batch_size=batch_size, spec_width=spec_width, mel_bins=mel_bins, **kwargs),
        output_signature=output_signature
    )
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def build_tiny_model(input_shape=(48, 128, 1), num_classes=100, alpha=0.5, depth_multiplier=2, embeddings_size=256, use_residual=True):
    """
    Build a small CNN model for audio spectrogram classification.

    Args:
        input_shape (tuple): Shape of input spectrograms (mel_bins, spec_width, 1).
        num_classes (int): Number of output classes.
        alpha (float): Scaling factor for model width.
        depth_multiplier (int): Number of block repetitions.
        embeddings_size (int): Size of final embedding layer.
        use_residual (bool): Whether to use residual connections.

    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    def depthwise_separable_block(x, filters, kernel_size=(3, 3), residual=False, strides=1):
        shortcut = x
        x = tf.keras.layers.DepthwiseConv2D(kernel_size, padding='same', use_bias=False, strides=strides)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU(max_value=6)(x)
        x = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU(max_value=6)(x)
        # Only add residual if shapes match and strides==1
        if residual and shortcut.shape[-1] == filters and strides == 1:
            x = tf.keras.layers.Add()([x, shortcut])
        return x

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(int(32 * alpha), (3, 3), strides=2, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6)(x)

    filters = int(64 * alpha)
    for i in range(depth_multiplier):
        x = depthwise_separable_block(x, filters, residual=use_residual, strides=2)
        x = depthwise_separable_block(x, filters, residual=use_residual, strides=1)
        filters = min(filters * 2, int(256 * alpha))

    x = depthwise_separable_block(x, embeddings_size, residual=False, strides=2)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)

    return tf.keras.models.Model(inputs, outputs)

def train_model(model, train_dataset, val_dataset, epochs=50, learning_rate=0.001, batch_size=64, patience=5, checkpoint_path="best_model.h5", steps_per_epoch=None, val_steps=None):
    """
    Train the model with cosine annealing, early stopping, and checkpointing.

    Args:
        model (tf.keras.Model): Model to train.
        train_dataset (tf.data.Dataset): Training dataset.
        val_dataset (tf.data.Dataset): Validation dataset.
        epochs (int): Number of epochs.
        learning_rate (float): Initial learning rate.
        batch_size (int): Batch size.
        patience (int): Early stopping patience.
        checkpoint_path (str): Path to save best model.
        steps_per_epoch (int, optional): Training steps per epoch.
        val_steps (int, optional): Validation steps per epoch.

    Returns:
        History: Keras training history object.
    """

    # Cosine annealing learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=learning_rate,
        first_decay_steps=steps_per_epoch * epochs // 4,
        t_mul=2.0,
        m_mul=1.0,
        alpha=1e-4
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='binary_crossentropy',
        metrics=[
            # Optionally keep AUC or TopKCategoricalAccuracy
            tf.keras.metrics.AUC(multi_label=True, name="auc"),
            # tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")
        ]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc', patience=patience, restore_best_weights=True, mode='max'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor='val_auc', save_best_only=True, mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=patience//2, min_lr=1e-6, verbose=1
        )
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks
    )
    return history

import argparse

def get_args():
    """
    Parse command-line arguments for training configuration.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train iNat-tiny audio classifier")
    parser.add_argument('--data_path_train', type=str, default='/data3/iNatSounds/train_min10_dev', help='Path to train dataset')
    parser.add_argument('--num_mels', type=int, default=64, help='Number of mel bins for spectrogram')
    parser.add_argument('--spec_width', type=int, default=128, help='Spectrogram width')
    parser.add_argument('--chunk_duration', type=int, default=3, help='Audio chunk duration (seconds)')
    parser.add_argument('--max_duration', type=int, default=30, help='Max audio duration (seconds)')
    parser.add_argument('--embeddings_size', type=int, default=256, help='Size of the final embeddings layer')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha for model scaling')
    parser.add_argument('--depth_multiplier', type=int, default=2, help='Depth multiplier for model')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup alpha')
    parser.add_argument('--mixup_probability', type=float, default=0.25, help='Fraction of batch to apply mixup')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/best_model.h5', help='Path to save best model')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Load file paths and classes
    file_paths, classes = load_file_paths_from_directory(args.data_path_train)
    
    # Perform sanity check on the dataset
    dataset_sanity_check(file_paths, classes, sample_rate=16000, max_duration=args.max_duration, chunk_duration=args.chunk_duration, spec_width=args.spec_width, mel_bins=args.num_mels)

    # Split dataset into training and validation sets
    val_split = args.val_split
    split_idx = int(len(file_paths) * (1 - val_split))
    train_paths = file_paths[:split_idx]
    val_paths = file_paths[split_idx:]
    print(f"Training on {len(train_paths)} files, validating on {len(val_paths)} files.")

    # Create training dataset (with mixup)
    train_dataset = load_dataset(
        train_paths, classes,
        sample_rate=16000,
        max_duration=args.max_duration,
        chunk_duration=args.chunk_duration,
        spec_width=args.spec_width,
        batch_size=args.batch_size,
        mixup_alpha=args.mixup_alpha,
        mixup_probability=args.mixup_probability,
        mel_bins=args.num_mels
    )

    # Create validation dataset (without mixup)
    val_dataset = load_dataset(
        val_paths, classes,
        sample_rate=16000,
        max_duration=args.max_duration,
        chunk_duration=args.chunk_duration,
        spec_width=args.spec_width,
        batch_size=args.batch_size,
        mixup_alpha=0.0,
        mixup_probability=0.0,
        mel_bins=args.num_mels
    )

    # Update steps_per_epoch and val_steps
    steps_per_epoch = len(train_paths) // args.batch_size
    val_steps = len(val_paths) // args.batch_size

    # Build model
    print("Building model...")
    model = build_tiny_model(
        input_shape=(args.num_mels, args.spec_width, 1),
        num_classes=len(classes),
        alpha=args.alpha,
        depth_multiplier=args.depth_multiplier,
        use_residual=True,
        embeddings_size=args.embeddings_size
    )
    model.summary()
    print("Model built successfully.")
    
    # Quantize awareness
    print("Applying quantization-aware training...")
    quantize_model = tfmot.quantization.keras.quantize_model
    model = quantize_model(model)
    print("Quantization-aware model created.")

    # Train model
    print("Starting training...")
    history = train_model(
        model, train_dataset, val_dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        patience=args.patience,
        checkpoint_path=args.checkpoint_path,
        steps_per_epoch=steps_per_epoch,
        val_steps=val_steps
    )
    print(f"Training complete. Model saved to '{args.checkpoint_path}'.")