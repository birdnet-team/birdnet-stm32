import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# random seed for reproducibility
np.random.seed(42)

def load_audio_file(path, sample_rate=22050, max_duration=30, chunk_duration=3):
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
    
    try: 
        # Load the audio file
        audio, sr = librosa.load(path, sr=sample_rate, mono=True, duration=max_duration)
        
    except:
        print(f"Error loading audio file {path}. Returning random noise.")
        # If loading fails, return random noise
        audio = np.random.randn(sample_rate * chunk_duration)
        sr = sample_rate
        
    # Split into chunks and pad if necessary
    chunk_size = sample_rate * chunk_duration
    num_chunks = len(audio) // chunk_size + (1 if len(audio) % chunk_size > 0 else 0)
    audio = librosa.util.fix_length(audio, size=num_chunks * chunk_size)
    audio = audio[:num_chunks * chunk_size]  # Ensure the audio length is a multiple of chunk size
    chunks = audio.reshape(num_chunks, chunk_size)       
        
    return chunks

def get_spectrogram_from_audio(audio, sample_rate=22050, n_fft=1024, mel_bins=48, spec_width=128):
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
    Compute a signal-to-noise ratio (SNR) from a spectrogram.

    Args:
        spectrogram (np.ndarray): 2D spectrogram array.

    Returns:
        np.ndarray: SNR value for the spectrogram.
    """
    # get a simple signal to noise ratio from the spectrogram
    signal = np.mean(spectrogram)
    noise = np.std(spectrogram)
    s2n = signal / (noise + 1e-10)  # Avoid division by zero
    
    return s2n

def get_s2n_from_audio(audio):
    """
    Compute a signal-to-noise ratio (SNR) from an audio signal.

    Args:
        audio (np.ndarray): 1D audio signal.

    Returns:
        float: SNR value.
    """
    # get a simple signal to noise ratio from the audio
    signal = np.mean(audio)
    noise = np.std(audio)
    s2n = signal / (noise + 1e-10)  # Avoid division by zero
    
    return s2n

def sort_by_s2n(samples, threshold=0.1):
    """
    Sort a list of samples (spectrograms or raw audio) by their mean SNR and filter out low-SNR specs.

    Args:
        samples (list of np.ndarray): List of samples.
        threshold (float): Minimum mean SNR to keep a sample.

    Returns:
        list of np.ndarray: Sorted and filtered samples.
    """
    
    if len(samples[0].shape) == 2:
        s2n_values = np.array([get_s2n_from_spectrogram(spec) for spec in samples])
    elif len(samples[0].shape) == 1:
        s2n_values = np.array([get_s2n_from_audio(audio) for audio in samples])
    else:
        raise ValueError("Samples must be 1D or 2D arrays (raw audio or spectrograms).")
    
    # Normalize SNR values to [0, 1]
    s2n_values /= (s2n_values.max() + 1e-10)  # Avoid division by zero
    
    sorted_indices = np.argsort(s2n_values)[::-1]  # Sort in descending order
    sorted_samples = [samples[i] for i in sorted_indices]
    
    # Filter out samples with SNR below the threshold but keep at least one chunk
    filtered_samples = [sample for sample, s2n in zip(sorted_samples, s2n_values[sorted_indices]) if s2n >= threshold]
    if len(filtered_samples) == 0:
        filtered_samples = [sorted_samples[0]]  # Ensure at least one sample is returned   
    
    return filtered_samples

def pick_random_samples(samples, num_samples=1):
    """
    Randomly select one or more samples from a list.

    Args:
        samples (list of np.ndarray): List of samples.
        num_samples (int): Number of samples to select.

    Returns:
        list or np.ndarray: Selected spectrogram(s).
    """
    
    # Pick random samples from the list
    if len(samples) == 0:
        return []
    if num_samples > len(samples):
        num_samples = len(samples)   
        
    indices = np.random.choice(len(samples), size=num_samples, replace=False)
    return [samples[i] for i in indices] if num_samples > 1 else samples[indices[0]]

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
    plt.savefig(f"samples/{title}.png")
    
    
def save_wav(audio, path, sample_rate=22050):
    """
    Save an audio signal to a WAV file.

    Args:
        audio (np.ndarray): 1D audio signal.
        path (str): Path to save the WAV file.
        sample_rate (int): Sample rate for saving the audio.

    Returns:
        None
    """
    sf.write(path, audio, sample_rate)