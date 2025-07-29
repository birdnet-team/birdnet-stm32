import librosa
import numpy as np
import matplotlib.pyplot as plt

# random seed for reproducibility
np.random.seed(42)

def load_audio_file(path, sample_rate=16000, max_duration=30, chunk_duration=3):
    """
    Load an audio file, resample to the given sample rate, and split into fixed-length chunks.

    Args:
        path (str): Path to the audio file.
        sample_rate (int): Target sample rate for loading audio.
        max_duration (int): Maximum duration (seconds) to load from the file.
        chunk_duration (int): Duration (seconds) of each chunk.

    Returns:
        np.ndarray: Array of shape (num_chunks, chunk_size) containing audio chunks.
    """
    # Load the audio file
    audio, sr = librosa.load(path, sr=sample_rate, mono=True, duration=max_duration)
    
    # Split into chunks and pad if necessary
    chunk_size = sample_rate * chunk_duration
    num_chunks = len(audio) // chunk_size + (1 if len(audio) % chunk_size > 0 else 0)
    audio = librosa.util.fix_length(audio, size=num_chunks * chunk_size)
    audio = audio[:num_chunks * chunk_size]  # Ensure the audio length is a multiple of chunk size
    chunks = audio.reshape(num_chunks, chunk_size)
        
    return chunks

def get_spectrogram_from_audio(audio, sample_rate=16000, n_fft=1024, mel_bins=48, spec_width=128):
    """
    Compute a normalized mel spectrogram from an audio chunk.

    Args:
        audio (np.ndarray): 1D audio signal.
        sample_rate (int): Sample rate of the audio.
        n_fft (int): FFT window size.
        mel_bins (int): Number of mel frequency bins.
        spec_width (int): Desired spectrogram width (frames).

    Returns:
        np.ndarray: 2D normalized mel spectrogram of shape (mel_bins, spec_width).
    """
    # Determine hop length based on the desired spectrogram width
    hop_length = (len(audio) // spec_width) if spec_width > 0 else n_fft // 2
    
    # Compute the mel spectrogram for audio
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=mel_bins,
        fmax=sample_rate // 2,
        fmin=150
    )
    
    # Convert to decibel scale
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_spectrogram = mel_spectrogram[:, :spec_width]
    
    # Flip the spectrogram vertically
    #mel_spectrogram = np.flipud(mel_spectrogram)
    
    # Normalize the spectrogram
    mel_spectrogram = (mel_spectrogram - mel_spectrogram.min()) / (mel_spectrogram.max() - mel_spectrogram.min() + 1e-10)
    
    return mel_spectrogram

def get_s2n_from_spectrogram(spectrogram):
    """
    Compute a normalized signal-to-noise ratio (SNR) from a spectrogram.

    Args:
        spectrogram (np.ndarray): 2D spectrogram array.

    Returns:
        np.ndarray: 1D array of normalized SNR values.
    """
    # get a simple signal to noise ratio from the spectrogram
    signal = np.mean(spectrogram, axis=1)
    noise = np.std(spectrogram, axis=1)
    s2n = signal / (noise + 1e-10)  # Avoid division by zero
    
    # Normalize SNR to a range of 0 to 1
    s2n = (s2n - s2n.min()) / (s2n.max() - s2n.min() + 1e-10)
    
    return s2n

def sort_by_s2n(specs, threshold=0.1):
    """
    Sort a list of spectrograms by their mean SNR and filter out low-SNR specs.

    Args:
        specs (list of np.ndarray): List of spectrograms.
        threshold (float): Minimum mean SNR to keep a spectrogram.

    Returns:
        list of np.ndarray: Sorted and filtered spectrograms.
    """
    s2n_values = np.array([get_s2n_from_spectrogram(spec).mean() for spec in specs])
    sorted_indices = np.argsort(s2n_values)[::-1]  # Sort in descending order
    sorted_specs = [specs[i] for i in sorted_indices]
    
    # Filter out specs with SNR below the threshold but keep at least one chunk
    filtered_specs = [spec for i, spec in enumerate(sorted_specs) if s2n_values[sorted_indices[i]] >= threshold]
    if not filtered_specs:
        filtered_specs = [sorted_specs[0]]           
    
    return filtered_specs

def pick_random_spectrogram(specs, num_samples=1):
    """
    Randomly select one or more spectrograms from a list.

    Args:
        specs (list of np.ndarray): List of spectrograms.
        num_samples (int): Number of spectrograms to select.

    Returns:
        list or np.ndarray: Selected spectrogram(s).
    """
    # Pick random spectrograms from the list
    if len(specs) == 0:
        return []
    if num_samples > len(specs):
        num_samples = len(specs)   
        
    indices = np.random.choice(len(specs), size=num_samples, replace=False)
    return [specs[i] for i in indices] if num_samples > 1 else specs[indices[0]]

def plot_spectrogram(spectrogram, title='Spectrogram'):
    """
    Plot and save a spectrogram image.

    Args:
        spectrogram (np.ndarray): 2D spectrogram array.
        title (str): Title for the plot and filename.

    Returns:
        None
    """
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.title(title)
    plt.xlabel('Time (frames)')
    plt.ylabel('Frequency (mel bins)')
    plt.tight_layout()
    plt.show()
    # save the plot to a file
    plt.savefig(f"specs/{title.replace(' ', '_').lower()}.png")