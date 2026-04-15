"""Shared test fixtures and configuration for pytest."""

import os
import sys

import numpy as np
import pytest

# Ensure the package is importable from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def sample_rate():
    """Default sample rate for tests."""
    return 22050


@pytest.fixture
def chunk_duration():
    """Default chunk duration in seconds."""
    return 3


@pytest.fixture
def mel_bins():
    """Default number of mel bins."""
    return 64


@pytest.fixture
def spec_width():
    """Default spectrogram width in frames."""
    return 256


@pytest.fixture
def fft_length():
    """Default FFT length."""
    return 512


@pytest.fixture
def num_classes():
    """Default number of classes for test models."""
    return 10


@pytest.fixture
def sine_wave(sample_rate, chunk_duration):
    """Generate a 1 kHz sine wave for testing.

    Returns:
        1-D float32 numpy array of length sample_rate * chunk_duration.
    """
    T = int(sample_rate * chunk_duration)
    t = np.linspace(0, chunk_duration, T, endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * 1000 * t)


@pytest.fixture
def silence(sample_rate, chunk_duration):
    """Generate a silence signal (all zeros).

    Returns:
        1-D float32 numpy array of length sample_rate * chunk_duration.
    """
    T = int(sample_rate * chunk_duration)
    return np.zeros(T, dtype=np.float32)


@pytest.fixture
def white_noise(sample_rate, chunk_duration):
    """Generate white noise for testing.

    Returns:
        1-D float32 numpy array of length sample_rate * chunk_duration.
    """
    rng = np.random.default_rng(42)
    T = int(sample_rate * chunk_duration)
    return rng.standard_normal(T).astype(np.float32)


@pytest.fixture
def tmp_dataset(tmp_path, sine_wave, sample_rate):
    """Create a minimal class-structured dataset in a temp directory.

    Layout:
        tmp_path/class_a/sample_0.wav
        tmp_path/class_b/sample_0.wav

    Returns:
        Tuple of (dataset root path, list of class names).
    """
    import soundfile as sf

    classes = ["class_a", "class_b"]
    for cls in classes:
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        sf.write(str(cls_dir / "sample_0.wav"), sine_wave, sample_rate)
    return str(tmp_path), classes
