"""Generate synthetic audio fixtures for testing.

Creates short WAV files (sine waves, chirps, noise) organized into a
class-structured dataset for unit and integration tests.

Usage:
    python -m tests.fixtures.generate_fixtures [--output tests/fixtures/data]
"""

import argparse
import os

import numpy as np
import soundfile as sf


def make_sine(sr: int = 22050, duration: float = 3.0, freq: float = 1000.0) -> np.ndarray:
    """Generate a pure sine wave."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return 0.8 * np.sin(2 * np.pi * freq * t)


def make_chirp(sr: int = 22050, duration: float = 3.0, f0: float = 500.0, f1: float = 4000.0) -> np.ndarray:
    """Generate a linear chirp (swept sine)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return 0.8 * np.sin(2 * np.pi * (f0 + (f1 - f0) / (2 * duration) * t) * t)


def make_noise(sr: int = 22050, duration: float = 3.0, seed: int = 42) -> np.ndarray:
    """Generate white noise."""
    rng = np.random.default_rng(seed)
    return 0.3 * rng.standard_normal(int(sr * duration)).astype(np.float32)


def generate_fixtures(output_dir: str, sr: int = 22050, duration: float = 3.0):
    """Create a minimal class-structured dataset with synthetic audio.

    Layout:
        output_dir/train/bird_a/sine_1000.wav
        output_dir/train/bird_a/chirp_500_4000.wav
        output_dir/train/bird_b/sine_2000.wav
        output_dir/train/bird_b/chirp_1000_6000.wav
        output_dir/train/noise/noise_0.wav
        output_dir/test/bird_a/sine_1500.wav
        output_dir/test/bird_b/sine_2500.wav
    """
    classes = {
        "bird_a": [
            ("sine_1000.wav", make_sine(sr, duration, 1000.0)),
            ("chirp_500_4000.wav", make_chirp(sr, duration, 500.0, 4000.0)),
            ("sine_800.wav", make_sine(sr, duration, 800.0)),
        ],
        "bird_b": [
            ("sine_2000.wav", make_sine(sr, duration, 2000.0)),
            ("chirp_1000_6000.wav", make_chirp(sr, duration, 1000.0, 6000.0)),
            ("sine_3000.wav", make_sine(sr, duration, 3000.0)),
        ],
        "noise": [
            ("noise_0.wav", make_noise(sr, duration, seed=0)),
            ("noise_1.wav", make_noise(sr, duration, seed=1)),
        ],
    }

    for split in ("train", "test"):
        for cls_name, files in classes.items():
            cls_dir = os.path.join(output_dir, split, cls_name)
            os.makedirs(cls_dir, exist_ok=True)
            subset = files if split == "train" else files[:1]
            for fname, audio in subset:
                sf.write(os.path.join(cls_dir, fname), audio, sr)

    print(f"Fixtures written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test fixtures")
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "data"))
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--duration", type=float, default=3.0)
    args = parser.parse_args()
    generate_fixtures(args.output, args.sr, args.duration)
