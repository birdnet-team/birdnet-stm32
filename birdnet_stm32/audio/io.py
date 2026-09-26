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
    return np.asarray(resample_poly(y, up, down), dtype=np.float32)


def peak_normalize(x: np.ndarray) -> np.ndarray:
    """Divide raw waveform windows by their peak, along the last axis.

    The ``raw`` models' input normalization, per window: ``x / (max|x| + 1e-6)``.
    Training, calibration and evaluation all go through here, and the firmware
    and ``reference/`` implement the same expression.

    Args:
        x: One window ``[T]`` or a batch ``[N, T]``.

    Returns:
        The scaled window(s), same shape.
    """
    return x / (np.max(np.abs(x), axis=-1, keepdims=True) + 1e-6)


def estimate_num_chunks(
    num_samples: int,
    sample_rate: int,
    chunk_duration: float,
    chunk_overlap: float = 0.0,
) -> int:
    """Estimate how many fixed-length chunks an audio window will produce.

    Args:
        num_samples: Number of audio samples in the waveform.
        sample_rate: Sampling rate of the waveform.
        chunk_duration: Duration of each chunk in seconds.
        chunk_overlap: Overlap between adjacent chunks in seconds.

    Returns:
        Number of chunks that :func:`split_audio_into_chunks` would emit.
    """
    chunk_size = int(sample_rate * chunk_duration)
    if num_samples <= 0 or chunk_size <= 0:
        return 0
    if num_samples <= chunk_size:
        return 1

    max_overlap = max(0.0, min(chunk_overlap, chunk_duration - 0.1))
    step_size = max(1, int(sample_rate * (chunk_duration - max_overlap)))
    n_full = 1 + max(0, (num_samples - chunk_size) // step_size)
    has_tail = (num_samples - chunk_size) % step_size != 0
    return int(n_full + int(has_tail))


def load_audio_window(
    path: str,
    sample_rate: int = 24000,
    max_duration: float | None = 30,
    chunk_duration: float = 3.0,
    random_offset: bool = False,
    return_offset: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Load one contiguous mono waveform window from disk.

    The window is read directly from the source file, resampled, and peak
    normalized to ``[-1, 1]``. This is the lowest-overhead path for callers
    that want to perform their own chunk selection after reading a file once.

    Args:
        path: Path to the audio file on disk.
        sample_rate: Target sampling rate in Hz.
        max_duration: Maximum duration to read in seconds. ``None`` or ``0``
            reads the remainder of the file.
        chunk_duration: Reference chunk duration in seconds, used when choosing
            a random start offset.
        random_offset: Whether to read from a random offset instead of the
            beginning of the file.
        return_offset: Also return where in the file the window starts, in
            seconds. Callers that attach per-window targets (teacher scores)
            need it to place a chunk on the file's own timeline.

    Returns:
        Mono float32 waveform, or ``(waveform, offset_seconds)`` with
        ``return_offset``. Returns an empty array (offset 0.0) on error.
    """
    empty = np.empty((0,), dtype=np.float32)
    try:
        info = sf.info(path)
        sr0 = int(info.samplerate)
        total_frames = int(info.frames)
        if total_frames <= 0 or sr0 <= 0:
            return (empty, 0.0) if return_offset else empty

        total_duration = total_frames / float(sr0)
        if max_duration and max_duration > 0:
            read_duration = min(float(max_duration), total_duration)
        else:
            read_duration = total_duration

        if random_offset:
            max_start_sec = max(0.0, total_duration - max(chunk_duration, read_duration))
            offset_sec = float(np.random.uniform(0.0, max_start_sec)) if max_start_sec > 0 else 0.0
        else:
            offset_sec = 0.0

        start_frame = min(int(offset_sec * sr0), total_frames)
        frames_left = max(0, total_frames - start_frame)
        frames_to_read = int(min(frames_left, read_duration * sr0))
        if frames_to_read <= 0:
            return (empty, 0.0) if return_offset else empty

        with sf.SoundFile(path, mode="r") as f:
            f.seek(start_frame)
            y = f.read(frames_to_read, dtype="float32", always_2d=True)
        if y.size == 0:
            return (empty, 0.0) if return_offset else empty

        y = y.mean(axis=1).astype(np.float32, copy=False)
        if sr0 != sample_rate:
            y = fast_resample(y, sr0, sample_rate)

        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 0.0:
            y = y / peak

        y = np.asarray(y, dtype=np.float32)
        # The position actually read, not the float draw it was rounded from.
        return (y, start_frame / float(sr0)) if return_offset else y
    except Exception:
        return (empty, 0.0) if return_offset else empty


def chunk_start_samples(
    num_samples: int,
    sample_rate: int,
    chunk_duration: float,
    chunk_overlap: float = 0.0,
) -> np.ndarray:
    """Start sample of every chunk :func:`split_audio_into_chunks` emits.

    Chunks step by ``chunk_duration - chunk_overlap``, and a final chunk is
    right-aligned to the end of the waveform when the steps do not land on it
    exactly. A waveform no longer than one chunk yields one chunk at 0.

    Args:
        num_samples: Length of the waveform in samples.
        sample_rate: Sampling rate in Hz.
        chunk_duration: Duration of each chunk in seconds.
        chunk_overlap: Overlap between chunks in seconds.

    Returns:
        int64 array of chunk start positions, in samples.
    """
    chunk_size = int(sample_rate * chunk_duration)
    if num_samples <= 0 or chunk_size <= 0:
        return np.empty((0,), dtype=np.int64)
    if num_samples <= chunk_size:
        return np.zeros((1,), dtype=np.int64)
    max_overlap = max(0.0, min(chunk_overlap, chunk_duration - 0.1))
    step_size = max(1, int(sample_rate * (chunk_duration - max_overlap)))
    starts = np.arange(0, num_samples - chunk_size + 1, step_size, dtype=np.int64)
    if starts.size == 0 or (starts[-1] + chunk_size < num_samples):
        starts = np.append(starts, num_samples - chunk_size)
    return starts


def split_audio_into_chunks(
    audio: np.ndarray,
    sample_rate: int = 24000,
    chunk_duration: float = 3.0,
    chunk_overlap: float = 0.0,
) -> np.ndarray:
    """Split a waveform into fixed-length chunks, padding short files once.

    Args:
        audio: Mono float32 waveform.
        sample_rate: Sampling rate in Hz.
        chunk_duration: Duration of each chunk in seconds.
        chunk_overlap: Overlap between chunks in seconds.

    Returns:
        Array of shape ``(num_chunks, chunk_size)`` with float32 chunks.
        If the waveform is shorter than one chunk, the single returned chunk is
        right-padded with zeros.
    """
    chunk_size = int(sample_rate * chunk_duration)
    if audio.size == 0 or chunk_size <= 0:
        return np.empty((0, max(chunk_size, 0)), dtype=np.float32)

    y = np.asarray(audio, dtype=np.float32)
    if y.ndim != 1:
        y = y.reshape(-1)

    if y.shape[0] <= chunk_size:
        padded = np.pad(y, (0, chunk_size - y.shape[0]), mode="constant")
        return padded[np.newaxis, :].astype(np.float32, copy=False)

    starts = chunk_start_samples(y.shape[0], sample_rate, chunk_duration, chunk_overlap)
    chunks = np.empty((starts.size, chunk_size), dtype=np.float32)
    for i, start in enumerate(starts):
        chunks[i] = y[start : start + chunk_size]
    return chunks


def load_audio_file(
    path: str,
    sample_rate: int = 24000,
    max_duration: float | None = 30,
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
        Returns an empty array on error.
    """
    audio = load_audio_window(
        path,
        sample_rate=sample_rate,
        max_duration=max_duration,
        chunk_duration=chunk_duration,
        random_offset=random_offset,
    )
    if audio.size == 0:
        chunk_size = max(0, int(sample_rate * chunk_duration))
        return np.empty((0, chunk_size), dtype=np.float32)
    return split_audio_into_chunks(
        audio,
        sample_rate=sample_rate,
        chunk_duration=chunk_duration,
        chunk_overlap=chunk_overlap,
    )
