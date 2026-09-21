"""Per-window soft targets from a cached teacher model.

A teacher (a larger audio classifier) is run once, offline, over every
training recording on a fixed window grid, and its scores are stored on disk.
Training then looks up the teacher window that best matches each chunk and
blends its scores into the chunk's label. The teacher never runs during
training, so this costs nothing per step.

Why: training labels are weak. A recording carries one species label, so every
chunk drawn from it trains as that species alone, whether the chunk holds the
call, silence, or a different bird calling over it. A per-window teacher score
says which of those it is.

Kept free of TensorFlow: the loader's worker processes import it.

Cache format (a directory):

- ``meta.json``: ``classes`` (must match the training class order),
  ``teacher_mask`` (one bool per class; ``False`` where the teacher has no
  corresponding output, and such classes keep their hard label), and
  ``teacher_window_s`` (teacher window length in seconds).
- ``mapped.npy``: float16 ``[n_windows, n_classes]`` teacher probabilities.
- ``starts.npy``: float32 ``[n_windows]`` window start, in seconds from the
  start of the recording.
- ``index.npz``: ``sample_id`` (recording file stem), ``row_offset`` and
  ``n_windows``, locating each recording's rows.

The large arrays are memory-mapped, so every worker shares one copy through
the page cache.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class TeacherTargets:
    """Look up cached teacher scores for a chunk of a training recording.

    Args:
        cache_dir: Directory in the format described in the module docstring.
        classes: The training class order. Must equal the cache's, so a cache
            built for another schema cannot silently mislabel classes.
    """

    def __init__(self, cache_dir: str | Path, classes: list[str]):
        root = Path(cache_dir)
        meta = json.loads((root / "meta.json").read_text())
        if list(meta["classes"]) != list(classes):
            raise ValueError(
                f"teacher cache {root} was built for a different class list "
                f"({len(meta['classes'])} classes) than training ({len(classes)})"
            )
        self.mask = np.asarray(meta["teacher_mask"], dtype=bool)
        if self.mask.shape != (len(classes),):
            raise ValueError(f"teacher_mask has shape {self.mask.shape}, expected ({len(classes)},)")
        self.window_s = float(meta["teacher_window_s"])
        self.mapped = np.load(root / "mapped.npy", mmap_mode="r")
        self.starts = np.load(root / "starts.npy", mmap_mode="r")
        index = np.load(root / "index.npz")
        self.index: dict[str, tuple[int, int]] = {
            str(sid): (int(off), int(n))
            for sid, off, n in zip(index["sample_id"], index["row_offset"], index["n_windows"], strict=True)
        }

    def __contains__(self, sample_id: str) -> bool:
        return sample_id in self.index

    def lookup(self, sample_id: str, chunk_start_s: float, chunk_duration_s: float) -> np.ndarray | None:
        """Teacher scores for the window whose centre is nearest the chunk's.

        Args:
            sample_id: Recording identifier (the file stem).
            chunk_start_s: Where the chunk starts in the recording, in seconds.
            chunk_duration_s: Chunk length in seconds.

        Returns:
            float32 ``[n_classes]`` teacher scores, or ``None`` when the
            recording is not in the cache or no teacher window lies within one
            window length of the chunk. Callers fall back to the hard label.
        """
        entry = self.index.get(sample_id)
        if entry is None:
            return None
        offset, n = entry
        if n <= 0:
            return None
        centres = np.asarray(self.starts[offset : offset + n], dtype=np.float64) + self.window_s / 2.0
        chunk_centre = chunk_start_s + chunk_duration_s / 2.0
        distance = np.abs(centres - chunk_centre)
        i = int(np.argmin(distance))
        if distance[i] > self.window_s:
            return None
        return np.asarray(self.mapped[offset + i], dtype=np.float32)


def blend_targets(hard: np.ndarray, teacher: np.ndarray, mask: np.ndarray, weight: float) -> np.ndarray:
    """Mix a hard label with teacher scores on the classes the teacher covers.

    ``(1 - weight) * hard + weight * teacher`` where ``mask`` is set, and the
    hard label unchanged elsewhere. Every entry stays in ``[0, 1]``, so the
    result is a valid soft target for binary cross-entropy.

    With ``weight`` 0.5, a labelled class the teacher also hears stays near 1,
    a labelled class on a chunk where the teacher hears nothing drops to 0.5,
    and an unlabelled species the teacher hears clearly rises toward 0.5.

    Args:
        hard: float32 ``[n_classes]`` hard label (0/1, or all zero for noise).
        teacher: float32 ``[n_classes]`` teacher scores.
        mask: bool ``[n_classes]``, where the teacher has a corresponding output.
        weight: Teacher share in ``[0, 1]``. 0 returns the hard label.

    Returns:
        float32 ``[n_classes]`` blended target.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"teacher weight must be in [0, 1], got {weight}")
    out = np.asarray(hard, dtype=np.float32).copy()
    t = np.clip(np.asarray(teacher, dtype=np.float32), 0.0, 1.0)
    out[mask] = (1.0 - weight) * out[mask] + weight * t[mask]
    return out
