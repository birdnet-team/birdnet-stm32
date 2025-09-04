import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# random seed for reproducibility
np.random.seed(42)

def load_audio_file(path, sample_rate=22050, max_duration=30, chunk_duration=3, random_offset=False):
    """
    Load an audio file with soundfile, resample to sample_rate, and split into fixed-length chunks.

    If random_offset is True, a random starting offset up to one chunk_duration (seconds) is used.

    Args:
        path (str): Path to the audio file on disk.
        sample_rate (int): Target sampling rate (Hz).
        max_duration (int): Maximum duration to load (seconds).
        chunk_duration (int): Duration of each chunk (seconds).
        random_offset (bool): Whether to start reading at a random offset.

    Returns:
        np.ndarray: Array of shape (num_chunks, chunk_size) with float32 audio chunks.
                    Returns [] on error.
    """
    try:
        info = sf.info(path)
        sr0 = int(info.samplerate)
        total_frames = int(info.frames)
        if total_frames <= 0 or sr0 <= 0:
            return []

        # Choose start offset (seconds)
        if random_offset:
            offset_sec = float(np.random.uniform(0.0, max(0.0, (total_frames / sr0) / 2) - chunk_duration))
        else:
            offset_sec = 0.0

        # Compute frame window to read
        start_frame = int(offset_sec * sr0)
        if start_frame >= total_frames:
            start_frame = 0
        frames_left = total_frames - start_frame
        frames_to_read = int(min(frames_left, (max_duration * sr0) if max_duration else frames_left))
        if frames_to_read <= 0:
            return []

        # Read only needed frames; force mono
        with sf.SoundFile(path, mode="r") as f:
            f.seek(start_frame)
            y = f.read(frames_to_read, dtype="float32", always_2d=True)
        if y.size == 0:
            return []
        y = y.mean(axis=1)  # mono float32

        # Resample if needed
        if sr0 != sample_rate:
            y = librosa.resample(y, orig_sr=sr0, target_sr=sample_rate, res_type="kaiser_fast")
        else:
            y = y.astype(np.float32, copy=False)

        # Chunking with zero-pad to multiple
        chunk_size = int(sample_rate * chunk_duration)
        if chunk_size <= 0:
            return []
        n = y.shape[0]
        num_chunks = (n + chunk_size - 1) // chunk_size  # ceil
        pad = num_chunks * chunk_size - n
        if pad > 0:
            y = np.pad(y, (0, pad), mode="constant")
        chunks = y.reshape(num_chunks, chunk_size).astype(np.float32, copy=False)
        return chunks
    except Exception:
        # Keep behavior consistent with previous code
        return []

def get_spectrogram_from_audio(audio, sample_rate=22050, n_fft=512, mel_bins=64, spec_width=256, mag_scale: str = "none"):
    """
    Compute a mel magnitude spectrogram with optional magnitude scaling and final normalization.

    Behavior:
        - 'none': Magnitude mel (power=1.0), then normalize to [0, 1].
        - 'pcen': Magnitude mel (power=1.0), scale to 32-bit PCM range, librosa.pcen, then normalize.
        - 'pwl':  Magnitude mel (power=1.0), pre-normalize to [0, 1], piecewise compression, then normalize.
        - 'db':   Magnitude mel (power=1.0), amplitude_to_db(ref=max), then normalize.

    Args:
        audio (np.ndarray): 1D audio array (mono).
        sample_rate (int): Sampling rate (Hz).
        n_fft (int): FFT size for STFT.
        mel_bins (int): Number of mel bands, or <=0 for linear STFT bins (magnitude).
        spec_width (int): Target number of time frames (columns).
        mag_scale (str): 'none' | 'db' | 'pcen' | 'pwl'.

    Returns:
        np.ndarray: Spectrogram array (mel_bins or fft_bins, spec_width), values in [0, 1].
    """
    hop_length = (len(audio) // spec_width) if spec_width > 0 else n_fft // 2
    if mel_bins <= 0:
        S = np.abs(librosa.stft(y=audio, 
                                n_fft=n_fft, 
                                hop_length=hop_length, 
                                win_length=n_fft, 
                                window="hann"))
    else:
        S = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window="hann",
            n_mels=mel_bins,
            power=1.0,
            fmin=150,
            fmax=sample_rate // 2,
            htk=False,
            norm="slaney",
        )
        
    # Ensure fixed width
    S = S[:, :spec_width]    

    if mag_scale == "pcen":
        S = librosa.pcen(S * (2.0**31), sr=sample_rate, hop_length=hop_length, axis=1)

    elif mag_scale == "pwl":
        Smin, Smax = S.min(), S.max()
        Snorm = (S - Smin) / (Smax - Smin + 1e-10)
        t1, t2, t3 = 0.10, 0.35, 0.65
        k0, k1, k2, k3 = 0.40, 0.25, 0.15, 0.08
        relu = lambda z: np.maximum(z, 0.0)
        S = k0 * Snorm + k1 * relu(Snorm - t1) + k2 * relu(Snorm - t2) + k3 * relu(Snorm - t3)

    elif mag_scale == "db":
        S = librosa.amplitude_to_db(S, ref=np.max)

    # Normalize
    S = normalize(S)
    
    return S

def normalize(S):
    """
    Normalize a spectrogram to [0, 1] per sample.

    Args:
        S (np.ndarray): Spectrogram array.

    Returns:
        np.ndarray: Normalized spectrogram, same shape as input.
    """
    S = (S - S.min()) / (S.max() - S.min() + 1e-10)
    return S

def get_s2n_from_spectrogram(spectrogram):
    """
    Compute a simple signal-to-noise proxy from a spectrogram.

    Uses mean/std as a proxy SNR: mean / std.

    Args:
        spectrogram (np.ndarray): 2D spectrogram array.

    Returns:
        float: SNR-like scalar value.
    """
    signal = np.mean(spectrogram)
    noise = np.std(spectrogram)
    s2n = signal / (noise + 1e-10)
    return s2n

def get_s2n_from_audio(audio):
    """
    Compute a simple signal-to-noise proxy from raw audio.

    Uses mean/std as a proxy SNR: mean / std.

    Args:
        audio (np.ndarray): 1D audio signal.

    Returns:
        float: SNR-like scalar value.
    """
    signal = np.mean(audio)
    noise = np.std(audio)
    s2n = signal / (noise + 1e-10)
    return s2n

def sort_by_s2n(samples, threshold=0.1):
    """
    Sort samples by a simple SNR proxy and filter out low-SNR ones.

    Accepts a list of 2D spectrograms or 1D audio arrays. Keeps at least one sample.

    Args:
        samples (list[np.ndarray]): List of 2D spectrograms or 1D audio arrays.
        threshold (float): Minimum normalized SNR to keep a sample (in [0, 1]).

    Returns:
        list[np.ndarray]: Sorted (desc by SNR) and filtered samples.
    """
    if len(samples[0].shape) == 2:
        s2n_values = np.array([get_s2n_from_spectrogram(spec) for spec in samples])
    elif len(samples[0].shape) == 1:
        s2n_values = np.array([get_s2n_from_audio(audio) for audio in samples])
    else:
        raise ValueError("Samples must be 1D or 2D arrays (raw audio or spectrograms).")

    # Normalize SNR values to [0, 1]
    s2n_values /= (s2n_values.max() + 1e-10)

    sorted_indices = np.argsort(s2n_values)[::-1]
    sorted_samples = [samples[i] for i in sorted_indices]

    # Filter out samples with SNR below the threshold but keep at least one chunk
    filtered_samples = [sample for sample, s2n in zip(sorted_samples, s2n_values[sorted_indices]) if s2n >= threshold]
    if len(filtered_samples) == 0:
        filtered_samples = [sorted_samples[0]]
    return filtered_samples

def pick_random_samples(samples, num_samples=1, pick_first=False):
    """
    Randomly select one or more samples from a list.

    Args:
        samples (list[np.ndarray]): List of samples (spectrograms or raw audio).
        num_samples (int): Number of samples to select.
        pick_first (bool): If True, always pick the first sample instead of random.

    Returns:
        list[np.ndarray] | np.ndarray: Selected samples. Returns a list if num_samples > 1,
            otherwise returns a single np.ndarray.
    """
    if len(samples) == 0:
        return []
    if num_samples > len(samples):
        num_samples = len(samples)

    indices = np.random.choice(len(samples), size=num_samples, replace=False)
    return [samples[i] for i in indices] if num_samples > 1 and not pick_first else samples[0]

def plot_spectrogram(spectrogram, title='Spectrogram'):
    """
    Plot and save a spectrogram image.

    Args:
        spectrogram (np.ndarray): 2D spectrogram.
        title (str): Plot title and filename stem for saved image.

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
    plt.savefig(f"samples/{title}.png")

def save_wav(audio, path, sample_rate=22050):
    """
    Save an audio signal to a WAV file.

    Args:
        audio (np.ndarray): 1D audio array (mono).
        path (str): Output file path (.wav).
        sample_rate (int): Sampling rate (Hz).

    Returns:
        None
    """
    sf.write(path, audio, sample_rate)