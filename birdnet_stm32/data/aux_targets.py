"""Where and when each training recording was made, as targets for a training-only auxiliary head.

The head (``birdnet_stm32.training.embedding_distillation``) predicts, from the pooled embedding, the
recording's location as a unit vector on the sphere (x, y, z) and its day of year as (sin, cos). Learning
these pushes the embedding to encode the species composition of a place and season, which a single-label
classification loss does not ask for.

The table lives in a folder of memory-mapped arrays, so every loader worker shares one copy:

    keys.npy     uint64, sorted: an 8-byte BLAKE2b hash of each sample id (the audio file's stem)
    targets.npy  float16 [N, 5]: x, y, z, sin(day), cos(day)
    valid.npy    uint8 [N, 2]: location known, date known
    meta.json    counts and the build inputs

Build it from a dataset manifest (``sample_id``, ``latitude``, ``longitude``, ``event_date``, ``source``)
with ``python -m birdnet_stm32.data.aux_targets <manifest.csv> <out_dir> [--site_coords sites.json]``;
``sites.json`` maps a ``source`` to ``[lat, lon]`` for collections recorded at one known site whose rows
carry no coordinates (e.g. a soundscape set).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

AUX_DIM = 5


def sample_key(sample_id: str) -> int:
    return int.from_bytes(hashlib.blake2b(sample_id.encode(), digest_size=8).digest(), "little")


def geo_unit(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """[N, 3] unit vectors for latitude / longitude in degrees."""
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=1)


def day_angle(day_of_year: np.ndarray) -> np.ndarray:
    """[N, 2] (sin, cos) of the day of year on a 365.25-day circle."""
    a = 2 * np.pi * (np.asarray(day_of_year, np.float64) - 1) / 365.25
    return np.stack([np.sin(a), np.cos(a)], axis=1)


def build(manifest_csv: str | Path, out_dir: str | Path, site_coords: dict | None = None) -> dict:
    import pandas as pd

    m = pd.read_csv(
        manifest_csv, usecols=["sample_id", "latitude", "longitude", "event_date", "source"], low_memory=False
    )
    m = m.drop_duplicates("sample_id")
    lat = pd.to_numeric(m.latitude, errors="coerce").to_numpy(np.float64, copy=True)
    lon = pd.to_numeric(m.longitude, errors="coerce").to_numpy(np.float64, copy=True)
    for source, (s_lat, s_lon) in (site_coords or {}).items():
        rows = (m.source == source).to_numpy() & ~np.isfinite(lat)
        lat[rows], lon[rows] = s_lat, s_lon
    geo_ok = (
        np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180) & ~((lat == 0) & (lon == 0))
    )
    date = pd.to_datetime(m.event_date, errors="coerce", format="mixed", utc=True)
    date_ok = date.notna().to_numpy() & (date.dt.year.fillna(0).to_numpy() >= 1900)
    targets = np.zeros((len(m), AUX_DIM), np.float32)
    targets[geo_ok, :3] = geo_unit(lat[geo_ok], lon[geo_ok])
    doy = date.dt.dayofyear.fillna(1).to_numpy()
    targets[date_ok, 3:] = day_angle(doy[date_ok])
    keys = np.array([sample_key(s) for s in m.sample_id.astype(str)], np.uint64)
    order = np.argsort(keys)
    if (np.diff(keys[order]) == 0).any():
        raise ValueError("sample id hash collision; use a wider key")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "keys.npy", keys[order])
    np.save(out / "targets.npy", targets[order].astype(np.float16))
    np.save(out / "valid.npy", np.stack([geo_ok, date_ok], axis=1)[order].astype(np.uint8))
    meta = {
        "manifest": str(manifest_csv),
        "rows": int(len(m)),
        "location": int(geo_ok.sum()),
        "date": int(date_ok.sum()),
        "both": int((geo_ok & date_ok).sum()),
        "site_coords": site_coords or {},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


class AuxTargets:
    """Memory-mapped lookup of a recording's auxiliary targets by sample id."""

    def __init__(self, folder: str | Path):
        d = Path(folder)
        self.keys = np.load(d / "keys.npy", mmap_mode="r")
        self.targets = np.load(d / "targets.npy", mmap_mode="r")
        self.valid = np.load(d / "valid.npy", mmap_mode="r")

    def lookup(self, sample_id: str) -> tuple[np.ndarray, np.ndarray]:
        """(targets float32 [5], valid float32 [2]); zeros when the recording is not in the table."""
        k = np.uint64(sample_key(sample_id))
        i = int(np.searchsorted(self.keys, k))
        if i < len(self.keys) and self.keys[i] == k:
            return np.asarray(self.targets[i], np.float32), np.asarray(self.valid[i], np.float32)
        return np.zeros(AUX_DIM, np.float32), np.zeros(2, np.float32)

    def __contains__(self, sample_id: str) -> bool:
        k = np.uint64(sample_key(sample_id))
        i = int(np.searchsorted(self.keys, k))
        return i < len(self.keys) and self.keys[i] == k


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the auxiliary location/date target table from a manifest")
    ap.add_argument("manifest")
    ap.add_argument("out_dir")
    ap.add_argument("--site_coords", default="", help="JSON {source: [lat, lon]} for single-site collections")
    a = ap.parse_args()
    sites = json.loads(Path(a.site_coords).read_text()) if a.site_coords else None
    print(json.dumps(build(a.manifest, a.out_dir, sites), indent=1))


if __name__ == "__main__":
    main()
