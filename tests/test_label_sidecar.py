"""Per-file label additions: parsing, extra positives, and confirmed absences the teacher may not raise."""

import numpy as np
from test_teacher_targets import CLASSES, TestWorkerIntegration, write_cache

from birdnet_stm32.data.sidecar import load_label_sidecar


def write_sidecar(path, rows):
    path.write_text("sample_id,positives,negatives\n" + "".join(f"{s},{p},{n}\n" for s, p, n in rows))
    return path


class TestLoadLabelSidecar:
    def test_maps_names_to_indices_and_drops_unknown_names(self, tmp_path):
        side = write_sidecar(tmp_path / "s.csv", [("a", f"{CLASSES[1]};nope", CLASSES[2]), ("b", "", "")])
        assert load_label_sidecar(side, CLASSES) == {"a": ((1,), (2,))}

    def test_a_class_both_present_and_absent_counts_as_present(self, tmp_path):
        side = write_sidecar(tmp_path / "s.csv", [("a", CLASSES[1], f"{CLASSES[1]};{CLASSES[2]}")])
        assert load_label_sidecar(side, CLASSES) == {"a": ((1,), (2,))}

    def test_rows_for_the_same_file_are_merged(self, tmp_path):
        side = write_sidecar(tmp_path / "s.csv", [("a", CLASSES[1], ""), ("a", "", CLASSES[2])])
        assert load_label_sidecar(side, CLASSES) == {"a": ((1,), (2,))}

    def test_negatives_column_is_optional(self, tmp_path):
        side = tmp_path / "s.csv"
        side.write_text(f"sample_id,positives\na,{CLASSES[2]}\n")
        assert load_label_sidecar(side, CLASSES) == {"a": ((2,), ())}


class TestWorkerSidecar:
    def _run(self, tmp_path, cache, weight, sidecar):
        from birdnet_stm32.data import worker

        path = tmp_path / "audio" / "a" / "rec1.wav"
        if not path.exists():
            TestWorkerIntegration()._write_audio(tmp_path)
        cfg = TestWorkerIntegration()._cfg(cache, weight)
        cfg["label_sidecar"] = sidecar
        worker._init_worker(cfg)
        try:
            ((_, target),) = worker._process_file(str(path))
        finally:
            worker._init_worker(TestWorkerIntegration()._cfg(None, 0.0))
        return target

    def test_extra_positives_join_the_folder_label(self, tmp_path):
        target = self._run(tmp_path, None, 0.0, {"rec1": ((2,), ())})
        np.testing.assert_array_equal(target, [1.0, 0.0, 1.0])

    def test_a_confirmed_absence_is_not_raised_by_the_teacher(self, tmp_path):
        cache = write_cache(
            tmp_path / "cache", recordings={"rec1": ([0.0], [[0.9, 0.0, 0.6]])}, mask=(True, True, True)
        )
        blended = self._run(tmp_path, cache, 0.5, {})
        assert blended[2] > 0.25  # the teacher alone lifts class 2
        target = self._run(tmp_path, cache, 0.5, {"rec1": ((), (2,))})
        assert target[2] == 0.0
        np.testing.assert_allclose(target[:2], blended[:2], atol=1e-6)  # other classes still blended

    def test_files_without_an_entry_are_unchanged(self, tmp_path):
        cache = write_cache(tmp_path / "cache")
        np.testing.assert_array_equal(
            self._run(tmp_path, cache, 0.5, {"other": ((1,), (2,))}), self._run(tmp_path, cache, 0.5, {})
        )
