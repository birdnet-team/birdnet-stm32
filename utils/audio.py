import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# random seed for reproducibility
np.random.seed(42)


def load_audio_file(path, sample_rate=22050, max_duration=30, chunk_duration=3, random_offset=False):
    """
    Load an audio file, resample to sample_rate, and split into fixed-length chunks.

    If random_offset is True, a random starting offset up to half the file duration is used.

    Args:
        path (str): Path to the audio file on disk.
        sample_rate (int): Target sampling rate (Hz).
        max_duration (int): Maximum duration to load (seconds).
        chunk_duration (int): Duration of each chunk (seconds).
        random_offset (bool): Whether to start reading at a random offset.

    Returns:
        np.ndarray: Array of shape (num_chunks, chunk_size) with float32 audio chunks.
    """
    if random_offset:
        offset = np.random.uniform(0, librosa.get_duration(filename=path) / 2)
    else:
        offset = 0.0

    try:
        audio, sr = librosa.load(path, sr=sample_rate, mono=True, duration=max_duration, offset=offset)
    except Exception:
        print(f"Error loading audio file {path}. Returning random noise.")
        audio = np.random.randn(sample_rate * chunk_duration).astype(np.float32)
        sr = sample_rate

    # Split into chunks, pad/truncate to a multiple of chunk_size
    chunk_size = sample_rate * chunk_duration
    num_chunks = len(audio) // chunk_size + (1 if len(audio) % chunk_size > 0 else 0)
    audio = librosa.util.fix_length(audio, size=num_chunks * chunk_size)
    audio = audio[:num_chunks * chunk_size]
    chunks = audio.reshape(num_chunks, chunk_size)
    return chunks


def mel_power_spectrogram(audio, sample_rate=22050, n_fft=1024, mel_bins=48, spec_width=128, fmin=150, fmax=None):
    """
    Compute a mel power spectrogram without compression or normalization.

    Args:
        audio (np.ndarray): 1D audio array (mono).
        sample_rate (int): Sampling rate (Hz).
        n_fft (int): FFT size for STFT.
        mel_bins (int): Number of mel bands.
        spec_width (int): Target number of time frames (columns).
        fmin (float): Minimum frequency for mel filterbank (Hz).
        fmax (float | None): Maximum frequency (Hz); defaults to sample_rate//2.

    Returns:
        np.ndarray: Mel power spectrogram of shape (mel_bins, spec_width).
    """
    fmax = fmax or (sample_rate // 2)
    hop_length = (len(audio) // spec_width) if spec_width > 0 else n_fft // 2
    S = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate,
        n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window="hann",
        n_mels=mel_bins, fmin=fmin, fmax=fmax, power=2.0, htk=False, norm='slaney'
    )
    return S[:, :spec_width]


def linear_power_spectrogram(audio, sample_rate=22050, n_fft=512, spec_width=128, power=2.0):
    """
    Compute a linear-frequency power spectrogram (|STFT|**power), unnormalized.

    Args:
        audio (np.ndarray): 1D audio array (mono).
        sample_rate (int): Sampling rate (Hz).
        n_fft (int): FFT size for STFT.
        spec_width (int): Target number of time frames (columns).
        power (float): Power exponent for magnitude (e.g., 2.0 => power).

    Returns:
        np.ndarray: Linear power spectrogram of shape (n_fft//2+1, spec_width).
    """
    hop_length = (len(audio) // spec_width) if spec_width > 0 else n_fft // 2
    S = np.abs(librosa.stft(
        y=audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window="hann"
    )) ** float(power)
    return S[:, :spec_width]


def get_spectrogram_from_audio(audio, sample_rate=22050, n_fft=1024, mel_bins=48, spec_width=128, mag_scale: str = "none"):
    """
    Compute a mel power spectrogram with optional magnitude scaling and final normalization.

    Behavior:
        - 'none': No compression, then normalize to [0, 1].
        - 'pcen': Apply librosa.pcen on power spectrogram, then normalize to [0, 1].
        - 'pwl':  Pre-normalize to [0, 1], apply piecewise-linear compression, then normalize.

    Args:
        audio (np.ndarray): 1D audio array (mono).
        sample_rate (int): Sampling rate (Hz).
        n_fft (int): FFT size for STFT.
        mel_bins (int): Number of mel bands.
        spec_width (int): Target number of time frames (columns).
        mag_scale (str): Magnitude scaling: 'none' | 'pcen' | 'pwl'.

    Returns:
        np.ndarray: Mel spectrogram-like array of shape (mel_bins, spec_width), values in [0, 1].
    """
    hop_length = (len(audio) // spec_width) if spec_width > 0 else n_fft // 2
    S = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window="hann",
        n_mels=mel_bins,
        power=2.0,
        fmin=150,
        fmax=sample_rate // 2,
        htk=False,
        norm="slaney",
    )
    S = S[:, :spec_width]

    if mag_scale == "pcen":
        S = librosa.pcen(S, sr=sample_rate, hop_length=hop_length, axis=1)

    elif mag_scale == "pwl":
        Smin, Smax = S.min(), S.max()
        Snorm = (S - Smin) / (Smax - Smin + 1e-10)
        t1, t2, t3 = 0.10, 0.35, 0.65
        k0, k1, k2, k3 = 0.40, 0.25, 0.15, 0.08
        relu = lambda z: np.maximum(z, 0.0)
        S = k0 * Snorm + k1 * relu(Snorm - t1) + k2 * relu(Snorm - t2) + k3 * relu(Snorm - t3)

    # Normalize AFTER mag_scale for consistent visualization
    Smin, Smax = S.min(), S.max()
    S = (S - Smin) / (Smax - Smin + 1e-10)
    return S


def to_db(S_power):
    """
    Convert a power spectrogram to dB for visualization and normalize to [0, 1].

    Args:
        S_power (np.ndarray): Power-like spectrogram (mel or linear).

    Returns:
        np.ndarray: dB-scaled spectrogram normalized to [0, 1], same shape as input.
    """
    S_db = librosa.power_to_db(S_power, ref=np.max)
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-10)
    return S_db


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


def get_linear_spectrogram_from_audio(audio, sample_rate=22050, n_fft=512, spec_width=128, power=2.0):
    """
    Compute a normalized linear-frequency spectrogram (|STFT|**power).

    Args:
        audio (np.ndarray): 1D audio array (mono).
        sample_rate (int): Sampling rate (Hz).
        n_fft (int): FFT size for STFT.
        spec_width (int): Target number of time frames (columns).
        power (float): Power exponent for magnitude (e.g., 2.0 => power).

    Returns:
        np.ndarray: Linear spectrogram of shape (n_fft//2+1, spec_width), normalized to [0, 1].
    """
    hop_length = (len(audio) // spec_width) if spec_width > 0 else n_fft // 2
    S = np.abs(librosa.stft(
        y=audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window="hann"
    )) ** float(power)
    S = S[:, :spec_width]
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


def pick_random_samples(samples, num_samples=1):
    """
    Randomly select one or more samples from a list.

    Args:
        samples (list[np.ndarray]): List of samples (spectrograms or raw audio).
        num_samples (int): Number of samples to select.

    Returns:
        list[np.ndarray] | np.ndarray: Selected samples. Returns a list if num_samples > 1,
            otherwise returns a single np.ndarray.
    """
    if len(samples) == 0:
        return []
    if num_samples > len(samples):
        num_samples = len(samples)

    indices = np.random.choice(len(samples), size=num_samples, replace=False)
    return [samples[i] for i in indices] if num_samples > 1 else samples[indices[0]]


def plot_spectrogram(spectrogram, title='Spectrogram'):
    """
    Plot and save a spectrogram image.

    Input is expected in power scale; it is converted to dB only for visualization.

    Args:
        spectrogram (np.ndarray): 2D power-like spectrogram.
        title (str): Plot title and filename stem for saved image.

    Returns:
        None
    """
    vis = to_db(spectrogram)
    plt.figure(figsize=(10, 4))
    plt.imshow(vis, aspect='auto', origin='lower', cmap='viridis')
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