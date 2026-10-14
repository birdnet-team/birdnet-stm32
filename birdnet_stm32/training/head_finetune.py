"""Fine-tune the classifier head on a frozen backbone, on every training file.

A post-training step: after ``train``, the backbone's pooled embedding of every
training file is computed once (one chunk per file, cut and targeted as the
training loader does: crop policy, teacher targets, label sidecar) and cached on
disk. Then only the classifier (``pred``) is refitted, starting from its trained
weights: each epoch draws the cached files class-capped (``ClassCappedPasses``,
with the field table, as training draws), fits sigmoid cross-entropy against the
cached targets, and keeps the epoch with the best validation score (class-mean
AP plus micro AP, one chunk per validation file, hard labels). The backbone, the
frontend and the config are untouched. With the backbone frozen every output is
an independent logistic regression, so an epoch over two million cached files
takes seconds to minutes.

Optional co-occurrence weighting: each output's negatives from species it never
shares a region with count ``far_weight`` (regions from a CSV of ``label,group``
rows; labels not listed, such as non-bird outputs, co-occur with everything).
With a range filter at inference those species never compete; the head then
spends its capacity on the confusions that remain, and without a filter it
costs nothing.

Measured on 2.1 Raw (1,013 outputs; INT8; 61 WABAD sites, no species mask):
micro window AP +3.3% for the refit alone, +3.9% with co-occurrence weighting
0.3; with each region's species mask, +10% (North America, Europe) and +16% to
+19% (Neotropics) over the unrefitted head.
"""

from __future__ import annotations

import csv
import json
import os
import random
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import tensorflow as tf

from birdnet_stm32.data.feeder import ClassCappedPasses, _create_worker_pool, _terminate_worker_pool
from birdnet_stm32.data.generator import _worker_config
from birdnet_stm32.data.species import NOISE_CLASSES
from birdnet_stm32.data.worker import _init_worker, _process_file

EMBED_BATCH = 256
FIT_BATCH = 1024


# ----------------------------------------------------------------------------------------------- metrics


def micro_ap(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision over all (row, output) pairs pooled; NaN without positives."""
    y, s = np.asarray(y).ravel() > 0.5, np.asarray(s, dtype=np.float32).ravel()
    if not y.any():
        return float("nan")
    hits = y[np.argsort(-s, kind="stable")]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision[hits].sum() / hits.sum())


def class_mean_ap(y: np.ndarray, s: np.ndarray) -> float:
    """Mean over outputs with positives of each output's average precision."""
    aps = [micro_ap(y[:, k], s[:, k]) for k in range(y.shape[1]) if (np.asarray(y[:, k]) > 0.5).any()]
    return float(np.mean(aps)) if aps else float("nan")


# ----------------------------------------------------------------------------------------------- embedding


def embedding_model(model: tf.keras.Model) -> tf.keras.Model:
    """The model up to the input of ``pred`` (the pooled embedding; dropout is inactive at inference)."""
    return tf.keras.Model(model.input, model.get_layer("pred").input)


def _embedder(model: tf.keras.Model) -> Callable[[np.ndarray], np.ndarray]:
    sub = embedding_model(model)
    fn = tf.function(lambda x: sub(x, training=False), reduce_retracing=True)

    def run(x: np.ndarray) -> np.ndarray:
        return np.concatenate([fn(tf.constant(x[i : i + EMBED_BATCH])).numpy() for i in range(0, len(x), EMBED_BATCH)])

    return run


def class_index(file_paths: Sequence[str], classes: Sequence[str]) -> np.ndarray:
    """Folder class index per file; -1 for noise folders (all-zero negatives)."""
    idx = {c: i for i, c in enumerate(classes)}
    out = []
    for p in file_paths:
        folder = p.replace("\\", "/").split("/")[-2]
        if folder.lower() in NOISE_CLASSES:
            out.append(-1)
        elif folder in idx:
            out.append(idx[folder])
        else:
            raise ValueError(f"{p}: folder '{folder}' is neither an output class nor a noise folder")
    return np.array(out, dtype=np.int32)


def embed_files(
    model: tf.keras.Model,
    file_paths: Sequence[str],
    classes: Sequence[str],
    cache_dir: str | Path,
    chunk_kwargs: dict,
    num_workers: int = 8,
    block: int = 4096,
    log: Callable[[str], None] = print,
) -> Path:
    """Embed one chunk per file into ``cache_dir`` (memory-mapped, resumable); returns ``cache_dir``.

    Writes ``files.txt``, ``cls.npy`` (folder class, -1 noise), ``emb.npy`` [files, dim] and ``y.npy``
    [files, classes] (float16), ``ok.npy`` (0 to do, 1 done, 2 failed to load) and ``meta.json``. A cache
    made for other files, classes or chunk settings is refused rather than silently reused.
    """
    from numpy.lib.format import open_memmap

    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    files = list(file_paths)
    meta = {
        "files": len(files),
        "classes": list(classes),
        "chunk": {
            k: (v if isinstance(v, (str, int, float, bool, type(None))) else repr(type(v)))
            for k, v in chunk_kwargs.items()
        },
    }
    if (d / "meta.json").exists():
        old = json.loads((d / "meta.json").read_text())
        old_files = (d / "files.txt").read_text().splitlines() if (d / "files.txt").exists() else []
        if old != meta or old_files != files:
            raise ValueError(
                f"{d} holds a cache for other files, classes or chunk settings; remove it or pass another --cache_dir"
            )
    else:
        (d / "files.txt").write_text("\n".join(files))
        np.save(d / "cls.npy", class_index(files, classes))
        (d / "meta.json").write_text(json.dumps(meta))
    run = _embedder(model)
    dim = int(embedding_model(model).output.shape[-1])
    n = len(files)
    mode = "r+" if (d / "ok.npy").exists() else "w+"
    emb = open_memmap(d / "emb.npy", mode=mode, dtype=np.float16, shape=(n, dim))
    ys = open_memmap(d / "y.npy", mode=mode, dtype=np.float16, shape=(n, len(classes)))
    ok = open_memmap(d / "ok.npy", mode=mode, dtype=np.uint8, shape=(n,))
    if ok.all():
        log(f"Embedding cache complete: {n} files ({int((ok == 2).sum())} unreadable)")
        return d
    kw = dict(chunk_kwargs)
    frontend, spec_width, mel_bins = kw.pop("audio_frontend"), kw.pop("spec_width"), kw.pop("mel_bins")
    worker_cfg, _ = _worker_config(list(classes), frontend, spec_width, mel_bins, 1, **kw)
    if num_workers > 0:
        pool = _create_worker_pool(num_workers, worker_cfg)
        process = lambda paths: pool.map(_process_file, paths, chunksize=16)  # noqa: E731
    else:  # in this process, as sample_chunks does
        pool = None
        _init_worker(worker_cfg)
        process = lambda paths: [_process_file(p) for p in paths]  # noqa: E731
    try:
        for s0 in range(0, n, block):
            if ok[s0 : s0 + block].all():
                continue
            part = range(s0, min(s0 + block, n))
            results = process([files[i] for i in part])
            idx, xs, yy = [], [], []
            for i, r in zip(part, results, strict=True):
                if r:
                    idx.append(i)
                    xs.append(r[0][0])
                    yy.append(r[0][1])
                else:
                    ok[i] = 2
            if idx:
                emb[idx] = run(np.stack(xs).astype(np.float32)).astype(np.float16)
                ys[idx] = np.stack(yy).astype(np.float16)
                ok[idx] = 1
            emb.flush(), ys.flush(), ok.flush()
            log(f"  embedded {min(s0 + block, n)}/{n} files ({int((ok == 2).sum())} unreadable)")
    finally:
        if pool is not None:
            _terminate_worker_pool(pool)
    return d


# ----------------------------------------------------------------------------------------------- weighting


def load_cooccurrence(path: str | Path, classes: Sequence[str]) -> np.ndarray:
    """[classes, classes] bool from a ``label,group`` CSV: True where two labels share a group.

    A label may appear in several groups. Labels not in the CSV (e.g. non-bird outputs) co-occur with everything.
    """
    idx = {c: i for i, c in enumerate(classes)}
    members: dict[str, set[int]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("label") in idx and row.get("group"):
                members.setdefault(row["group"], set()).add(idx[row["label"]])
    n = len(classes)
    co = np.zeros((n, n), dtype=bool)
    for m in members.values():
        m = np.array(sorted(m))
        co[np.ix_(m, m)] = True
    listed = np.zeros(n, dtype=bool)
    for m in members.values():
        listed[list(m)] = True
    co[~listed, :] = True
    co[:, ~listed] = True
    np.fill_diagonal(co, True)
    return co


def negative_weights(cooccurrence: np.ndarray, far_weight: float) -> np.ndarray:
    """Per (sample class, output) weight: 1 where the two co-occur, ``far_weight`` elsewhere."""
    return np.where(cooccurrence, 1.0, float(far_weight)).astype(np.float32)


# ----------------------------------------------------------------------------------------------- fitting


def fit_head(
    emb,
    targets,
    cls: np.ndarray,
    epoch_rows: Callable[[], np.ndarray],
    val_emb: np.ndarray,
    val_targets: np.ndarray,
    w0: np.ndarray,
    b0: np.ndarray,
    *,
    learning_rate: float = 3e-4,
    epochs: int = 30,
    patience: int = 5,
    weights: np.ndarray | None = None,
    seed: int = 42,
    log: Callable[[str], None] = print,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Refit a dense sigmoid head from ``(w0, b0)``; returns the best weights by validation and the history.

    ``emb``/``targets`` are indexable by row arrays (memory maps work); ``epoch_rows()`` returns one epoch's rows;
    ``weights`` [classes, classes] scales each sample's negatives by its class (rows of class -1, noise, and
    every positive > 0.5 keep weight 1).
    """
    tf.keras.utils.set_random_seed(seed)
    w = tf.Variable(np.asarray(w0, np.float32))
    b = tf.Variable(np.asarray(b0, np.float32))
    opt = tf.keras.optimizers.Adam(learning_rate)

    @tf.function
    def step(x, t, wt):
        with tf.GradientTape() as tape:
            bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=t, logits=tf.matmul(x, w) + b)
            loss = tf.reduce_sum(bce * wt) / tf.reduce_sum(wt)
        opt.apply_gradients(zip(tape.gradient(loss, [w, b]), [w, b], strict=True))
        return loss

    val_emb = np.asarray(val_emb, np.float32)
    best, bad, history = None, 0, []
    for ep in range(epochs):
        rows = epoch_rows()
        losses = []
        for i in range(0, len(rows), FIT_BATCH):
            r = np.sort(rows[i : i + FIT_BATCH])
            t = np.asarray(targets[r], np.float32)
            if weights is None:
                wt = np.ones_like(t)
            else:
                k = cls[r]
                wt = np.where(k[:, None] >= 0, weights[np.maximum(k, 0)], 1.0).astype(np.float32)
                wt = np.maximum(wt, (t > 0.5).astype(np.float32))
            losses.append(float(step(tf.constant(np.asarray(emb[r], np.float32)), tf.constant(t), tf.constant(wt))))
        wn, bn = w.numpy(), b.numpy()
        z = val_emb @ wn + bn
        cap, mic = class_mean_ap(val_targets, z), micro_ap(val_targets, z)
        score = cap + mic
        history.append(
            {
                "epoch": ep + 1,
                "loss": float(np.mean(losses)) if losses else float("nan"),
                "val_class_ap": round(cap, 4),
                "val_micro_ap": round(mic, 4),
                "rows": int(len(rows)),
            }
        )
        log(f"  epoch {ep + 1}/{epochs}: loss {history[-1]['loss']:.5f}, val class AP {cap:.4f}, micro AP {mic:.4f}")
        if best is None or score > best[0]:
            best, bad = (score, wn.copy(), bn.copy(), ep + 1), 0
        else:
            bad += 1
            if bad >= patience:
                log(f"  no improvement for {patience} epochs: stopping")
                break
    _, wb, bb, ep = best
    for h in history:
        h["selected"] = h["epoch"] == ep
    return wb, bb, history


def capped_epochs(
    cache_files: Sequence[str], ok: np.ndarray, cap: int, field_kwargs: dict | None = None, seed: int = 42
) -> Callable[[], np.ndarray]:
    """An ``epoch_rows`` callable: each call one class-capped pass (``ClassCappedPasses``) over the cached files."""
    usable = [p for p, o in zip(cache_files, ok, strict=True) if o == 1]
    row_of = {p: i for i, p in enumerate(cache_files)}
    passes = ClassCappedPasses(usable, cap, rng=random.Random(seed), **(field_kwargs or {})) if cap > 0 else None
    rng = np.random.default_rng(seed)

    def epoch_rows() -> np.ndarray:
        files = passes.next_pass() if passes is not None else usable
        return rng.permutation(np.array([row_of[p] for p in files], dtype=np.int64))

    return epoch_rows


def replace_head(model: tf.keras.Model, w: np.ndarray, b: np.ndarray) -> tf.keras.Model:
    """Set ``pred``'s weights in place (shapes must match) and return the model."""
    pred = model.get_layer("pred")
    old = pred.get_weights()
    if old[0].shape != w.shape or old[1].shape != b.shape:
        raise ValueError(f"head shape {w.shape}/{b.shape} does not match pred {old[0].shape}/{old[1].shape}")
    pred.set_weights([w.astype(old[0].dtype), b.astype(old[1].dtype)])
    return model


def save_report(path: str | os.PathLike, report: dict) -> None:
    Path(path).write_text(json.dumps(report, indent=2))
