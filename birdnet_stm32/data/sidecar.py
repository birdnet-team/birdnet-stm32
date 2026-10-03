"""Per-file label additions read from a CSV next to the folder labels.

A training file's folder names one class. Some recordings carry more: an annotated
soundscape segment can hold several species, and a validated clip can state that a
species is *not* there. The sidecar adds both, by sample id (the file's stem):

    sample_id,positives,negatives
    3f2a...,american_robin;song_sparrow,
    9c41...,,blue_jay;house_finch

* ``positives``: classes present besides the folder's, set to 1 in the hard label;
* ``negatives``: classes confirmed absent: their hard target is 0 and the teacher's soft
  target is not blended in for them, whatever the teacher heard.

Either column may be missing or empty. Labels not in the class list are ignored.
"""

from __future__ import annotations

import csv
from pathlib import Path


def load_label_sidecar(path: str | Path, classes: list[str]) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Map sample id -> (positive class indices, negative class indices)."""
    index = {c: i for i, c in enumerate(classes)}

    def parse(cell: str | None) -> tuple[int, ...]:
        return tuple(sorted({index[c] for c in (cell or "").split(";") if c in index}))

    out: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            pos, neg = parse(row.get("positives")), parse(row.get("negatives"))
            neg = tuple(k for k in neg if k not in pos)
            if pos or neg:
                old_pos, old_neg = out.get(row["sample_id"], ((), ()))
                out[row["sample_id"]] = (tuple(sorted(set(old_pos) | set(pos))), tuple(sorted(set(old_neg) | set(neg))))
    return out
