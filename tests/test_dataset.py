"""Unit tests for dataset loading utilities."""

import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow required for dataset tests")

from birdnet_stm32.data.dataset import load_classes_file, load_file_paths_from_directory


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

    def test_symlinked_class_folders_and_files_are_found(self, tmp_path):
        """Link trees symlink class folders and audio files; both must be listed, nested folders too."""
        store = tmp_path / "store"
        (store / "a").mkdir(parents=True)
        (store / "a" / "x.flac").touch()
        (store / "b.flac").touch()
        root = tmp_path / "train"
        (root / "class_b" / "deeper").mkdir(parents=True)
        (root / "class_a").symlink_to(store / "a", target_is_directory=True)
        (root / "class_b" / "y.flac").symlink_to(store / "b.flac")
        (root / "class_b" / "deeper" / "z.wav").touch()
        (root / "class_b" / "notes.txt").touch()
        paths, found = load_file_paths_from_directory(str(root))
        assert found == ["class_a", "class_b", "deeper"]
        assert sorted(p.replace(str(root) + "/", "") for p in paths) == [
            "class_a/x.flac",
            "class_b/deeper/z.wav",
            "class_b/y.flac",
        ]

    def test_classes_file(self, tmp_path):
        """Class files preserve order and reject duplicate outputs."""
        labels = tmp_path / "labels.txt"
        labels.write_text("class_b\n# comment\nclass_a\n")
        assert load_classes_file(str(labels)) == ["class_b", "class_a"]
        labels.write_text("class_a\nclass_a\n")
        with pytest.raises(ValueError, match="duplicate"):
            load_classes_file(str(labels))
