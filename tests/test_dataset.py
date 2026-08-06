"""Unit tests for dataset loading utilities."""

import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow required for dataset tests")

from birdnet_stm32.data.dataset import load_classes_file, load_file_paths_from_directory, upsample_minority_classes


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

    def test_explicit_order_and_noise(self, tmp_path):
        """An explicit output order should retain all-zero noise files."""
        for class_name in ("class_a", "class_b", "noise"):
            directory = tmp_path / class_name
            directory.mkdir()
            (directory / "sample.wav").touch()
        paths, found_classes = load_file_paths_from_directory(tmp_path, classes=["class_b", "class_a"])
        assert found_classes == ["class_b", "class_a"]
        assert len(paths) == 3

    def test_classes_file(self, tmp_path):
        """Class files preserve order and reject duplicate outputs."""
        labels = tmp_path / "labels.txt"
        labels.write_text("class_b\n# comment\nclass_a\n")
        assert load_classes_file(str(labels)) == ["class_b", "class_a"]
        labels.write_text("class_a\nclass_a\n")
        with pytest.raises(ValueError, match="duplicate"):
            load_classes_file(str(labels))


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

    def test_preserves_noise_examples(self, tmp_path):
        """Balancing output classes must retain all-zero noise examples."""
        class_paths = [str(tmp_path / "bird" / f"{i}.wav") for i in range(4)]
        noise_paths = [str(tmp_path / "noise" / f"{i}.wav") for i in range(3)]
        result = upsample_minority_classes(class_paths + noise_paths, ["bird"], ratio=0.5)
        assert set(noise_paths).issubset(result)
        assert len(result) == len(class_paths) + len(noise_paths)
