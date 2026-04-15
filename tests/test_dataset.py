"""Unit tests for dataset loading utilities."""

import os

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow required for dataset tests")

from birdnet_stm32.data.dataset import load_file_paths_from_directory, upsample_minority_classes


class TestLoadFilePaths:
    """Tests for load_file_paths_from_directory."""

    def test_finds_wav_files(self, tmp_dataset):
        """Should find .wav files in a class-structured directory."""
        root, classes = tmp_dataset
        paths, found_classes = load_file_paths_from_directory(root)
        assert len(paths) == 2
        assert set(found_classes) == set(classes)

    def test_class_filter(self, tmp_dataset):
        """Should restrict to specified classes."""
        root, _classes = tmp_dataset
        paths, found_classes = load_file_paths_from_directory(root, classes=["class_a"])
        assert len(paths) == 1
        assert found_classes == ["class_a"]


class TestUpsampleMinority:
    """Tests for upsample_minority_classes."""

    def test_upsamples(self, tmp_path):
        """Minority class should be upsampled toward target size."""
        # Create fake paths
        majority = [str(tmp_path / "big" / f"{i}.wav") for i in range(100)]
        minority = [str(tmp_path / "small" / f"{i}.wav") for i in range(10)]
        all_paths = majority + minority
        classes = ["big", "small"]

        # Create dirs so os.path.dirname works
        (tmp_path / "big").mkdir()
        (tmp_path / "small").mkdir()

        result = upsample_minority_classes(all_paths, classes, ratio=0.5)
        # Small class should grow to ~50
        small_count = sum(1 for p in result if "small" in p)
        assert small_count >= 40  # Allow some variance
