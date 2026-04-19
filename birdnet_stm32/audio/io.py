"""Audio file loading, resampling, chunking, and saving.

This module handles all I/O operations for audio files: loading with soundfile,
resampling via scipy, splitting into fixed-length chunks, and writing WAV files.
"""

from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def fast_resample(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample audio using scipy.signal.resample_poly.

    Args:
        y: 1D float32 audio signal.
        sr_in: Original sample rate (Hz).
        sr_out: Target sample rate (Hz).

    Returns:
        Resampled audio as float32.
    """
    if sr_in == sr_out:
        return y.astype(np.float32, copy=False)
    g = gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    return resample_poly(y, up, down).astype(np.float32, copy=False)


def load_audio_file(
    path: str,
    sample_rate: int = 24000,
    max_duration: int = 30,
    chunk_duration: float = 3.0,
    chunk_overlap: float = 0.0,
    random_offset: bool = False,
) -> np.ndarray:
    """Load an audio file, resample, normalize, and split into fixed-length chunks.

    Args:
        path: Path to the audio file on disk.
        sample_rate: Target sampling rate (Hz).
        max_duration: Maximum duration to load (seconds).
        chunk_duration: Duration of each chunk (seconds).
        chunk_overlap: Overlap between chunks (seconds, 0 <= overlap < chunk_duration).
        random_offset: Whether to start reading at a random offset.

    Returns:
        Array of shape (num_chunks, chunk_size) with float32 audio chunks.
        Returns empty list on error.
    """
    try:
        info = sf.info(path)
        sr0 = int(info.samplerate)
        total_frames = int(info.frames)
        if total_frames <= 0 or sr0 <= 0:
            return []

        # Choose start offset (seconds)
        if random_offset:
            offset_sec = float(np.random.uniform(0.0, max(0.0, (total_frames / sr0) / 2 - chunk_duration)))
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
        y = fast_resample(y, sr0, sample_rate) if sr0 != sample_rate else y.astype(np.float32, copy=False)

        chunk_size = int(sample_rate * chunk_duration)
        if chunk_size <= 0:
            return []

        # Normalize to -1.0 to 1.0
        y = y / (np.max(np.abs(y)) + 1e-10)

        # Interpret chunk_overlap as seconds, clamp to [0, chunk_duration - 0.1]
        max_overlap = max(0.0, min(chunk_overlap, chunk_duration - 0.1))
        step_size = int(sample_rate * (chunk_duration - max_overlap))
        if step_size < 1:
            step_size = 1

        n = y.shape[0]
        starts = np.arange(0, n - chunk_size + 1, step_size)
        if len(starts) == 0 or (starts[-1] + chunk_size < n):
            starts = np.append(starts, n - chunk_size)
        starts = starts.astype(int)

        chunks = np.stack([y[s : s + chunk_size] for s in starts])

        # Pad last chunk if shorter than half chunk_duration
        if chunks.shape[0] > 0 and chunks[-1].shape[0] < chunk_size // 2:
            pad_width = chunk_size - chunks[-1].shape[0]
            chunks[-1] = np.pad(chunks[-1], (0, pad_width), mode="constant", constant_values=0)

        return chunks.astype(np.float32, copy=False)
    except Exception:
        return []


def save_wav(audio: np.ndarray, path: str, sample_rate: int = 24000) -> None:
    """Save an audio signal to a WAV file.

    Args:
        audio: 1D audio array (mono).
        path: Output file path (.wav).
        sample_rate: Sampling rate (Hz).
    """
    sf.write(path, audio, sample_rate)
