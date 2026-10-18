"""Where INT8 calibration draws its audio from: the training files of the model's classes."""

import numpy as np
import soundfile as sf

from birdnet_stm32.conversion.quantize import calibration_source, stratified_sample_paths


def _tree(root, layout):
    for folder, count in layout.items():
        (root / folder).mkdir(parents=True)
        for i in range(count):
            sf.write(str(root / folder / f"{i}.wav"), np.zeros(240, np.float32), 24000)


def test_draws_from_the_training_files_of_the_models_classes(tmp_path):
    _tree(tmp_path / "train", {"robin": 3, "wren": 2, "gull": 4})
    paths = calibration_source(str(tmp_path / "train"), ["robin", "wren"])
    assert len(paths) == 5
    assert all("gull" not in p for p in paths)
    # Stratified by class folder, so both classes contribute to a small draw.
    drawn = stratified_sample_paths(paths, 2, seed=42)
    assert {p.split("/")[-2] for p in drawn} == {"robin", "wren"}
