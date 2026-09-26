"""--calibration_dir: where INT8 calibration draws its audio from."""

import numpy as np
import pytest
import soundfile as sf

from birdnet_stm32.conversion.quantize import calibration_source, stratified_sample_paths


def _tree(root, layout):
    for folder, count in layout.items():
        (root / folder).mkdir(parents=True)
        for i in range(count):
            sf.write(str(root / folder / f"{i}.wav"), np.zeros(240, np.float32), 24000)


def test_defaults_to_the_training_files_of_the_models_classes(tmp_path):
    _tree(tmp_path / "train", {"robin": 3, "wren": 2, "gull": 4})
    paths = calibration_source(str(tmp_path / "train"), "", ["robin", "wren"])
    assert len(paths) == 5
    assert all("gull" not in p for p in paths)


def test_a_calibration_dir_replaces_them_whatever_its_folders_are_called(tmp_path):
    _tree(tmp_path / "train", {"robin": 3})
    _tree(tmp_path / "field", {"site_a": 4, "site_b": 2})
    paths = calibration_source(str(tmp_path / "train"), str(tmp_path / "field"), ["robin"])
    assert len(paths) == 6
    assert all("/field/" in p for p in paths)
    # Stratified by subfolder, so both sites contribute to a small draw.
    drawn = stratified_sample_paths(paths, 2, seed=42)
    assert {p.split("/")[-2] for p in drawn} == {"site_a", "site_b"}


def test_an_empty_calibration_dir_is_refused(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="calibration_dir"):
        calibration_source("", str(tmp_path / "empty"), None)
