"""Tests for the post-training head fine-tune (birdnet_stm32.training.head_finetune, cli.finetune_head)."""

import json
import sys

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow required")

from birdnet_stm32.training import head_finetune as hf  # noqa: E402


def test_cooccurrence_groups_and_unlisted_labels(tmp_path):
    classes = ["a", "b", "c", "d", "human"]
    csv = tmp_path / "groups.csv"
    csv.write_text("label,group\na,north\nb,north\nb,south\nc,south\nd,east\n")
    co = hf.load_cooccurrence(csv, classes)
    assert co.shape == (5, 5) and (co == co.T).all() and co.diagonal().all()
    assert co[0, 1] and co[1, 2] and not co[0, 2]  # a-b share north, b-c share south, a-c share nothing
    assert co[3, :4].tolist() == [False, False, False, True]  # d is alone in east
    assert co[4].all() and co[:, 4].all()  # an unlisted label (non-bird) co-occurs with everything


def test_negative_weights():
    co = np.array([[True, False], [False, True]])
    w = hf.negative_weights(co, 0.3)
    assert w.dtype == np.float32
    assert np.allclose(w, [[1.0, 0.3], [0.3, 1.0]])


def test_class_index_noise_and_unknown(tmp_path):
    paths = ["/d/a/1.wav", "/d/noise/2.wav", "/d/b/3.wav"]
    assert hf.class_index(paths, ["a", "b"]).tolist() == [0, -1, 1]
    with pytest.raises(ValueError):
        hf.class_index(["/d/zzz/1.wav"], ["a", "b"])


def test_capped_epochs_draw_and_rotation():
    files = (
        [f"/d/a/a{i}.wav" for i in range(10)]
        + [f"/d/b/b{i}.wav" for i in range(3)]
        + [f"/d/noise/n{i}.wav" for i in range(4)]
    )
    ok = np.ones(len(files), np.uint8)
    ok[1] = 2  # unreadable: never drawn
    rows = hf.capped_epochs(files, ok, cap=5, seed=0)
    seen_a = set()
    for _ in range(3):
        r = rows()
        drawn = [files[i] for i in r]
        assert sum("/a/" in p for p in drawn) == 5
        assert sum("/b/" in p for p in drawn) == 3  # smaller than the cap: all of them
        assert sum("/noise/" in p for p in drawn) == 4  # noise is never capped
        assert files[1] not in drawn
        seen_a |= {p for p in drawn if "/a/" in p}
    assert len(seen_a) == 9  # successive passes rotate through every usable file of the capped class


def _clusters(n_per=200, dim=16, k=3, seed=0):
    centers = np.random.default_rng(123).normal(0, 3, (k, dim)).astype(np.float32)  # shared by every draw
    rng = np.random.default_rng(seed)
    x = np.concatenate([centers[c] + rng.normal(0, 1, (n_per, dim)) for c in range(k)]).astype(np.float32)
    y = np.repeat(np.eye(k, dtype=np.float32), n_per, axis=0)
    cls = np.repeat(np.arange(k), n_per).astype(np.int32)
    return x, y, cls


def test_fit_head_improves_a_degraded_head_and_marks_the_selection():
    x, y, cls = _clusters()
    xv, yv, _ = _clusters(seed=1)
    rng = np.random.default_rng(0)
    w0 = rng.normal(0, 0.01, (x.shape[1], 3)).astype(np.float32)
    b0 = np.full(3, -3.0, np.float32)
    before = hf.class_mean_ap(yv, xv @ w0 + b0)
    w, b, hist = hf.fit_head(
        x,
        y,
        cls,
        lambda: rng.permutation(len(x)),
        xv,
        yv,
        w0,
        b0,
        learning_rate=0.05,
        epochs=10,
        patience=3,
        log=lambda *_: None,
    )
    assert w.shape == w0.shape and b.shape == b0.shape
    assert hf.class_mean_ap(yv, xv @ w + b) > before + 0.2
    assert sum(h["selected"] for h in hist) == 1


def test_fit_head_unit_weights_equal_no_weights():
    x, y, cls = _clusters(n_per=50)
    xv, yv, _ = _clusters(n_per=20, seed=1)
    w0 = np.zeros((x.shape[1], 3), np.float32)
    b0 = np.zeros(3, np.float32)
    order = np.random.default_rng(0).permutation(len(x))
    kw = dict(learning_rate=0.01, epochs=2, patience=5, log=lambda *_: None)
    w1, b1, _ = hf.fit_head(x, y, cls, lambda: order, xv, yv, w0, b0, **kw)
    w2, b2, _ = hf.fit_head(x, y, cls, lambda: order, xv, yv, w0, b0, weights=np.ones((3, 3), np.float32), **kw)
    assert np.allclose(w1, w2, atol=1e-5) and np.allclose(b1, b2, atol=1e-5)


def test_replace_head_checks_shape():
    inp = tf.keras.Input((4,))
    model = tf.keras.Model(inp, tf.keras.layers.Dense(2, activation="sigmoid", name="pred")(inp))
    hf.replace_head(model, np.ones((4, 2), np.float32), np.zeros(2, np.float32))
    assert np.allclose(model.get_layer("pred").get_weights()[0], 1.0)
    with pytest.raises(ValueError):
        hf.replace_head(model, np.ones((4, 3), np.float32), np.zeros(3, np.float32))


@pytest.mark.integration
@pytest.mark.slow
def test_cli_end_to_end(tmp_path, monkeypatch):
    """A tiny trained model + synthetic audio: only pred changes, the config is copied, the cache is reused."""
    import soundfile as sf

    from birdnet_stm32.cli import finetune_head
    from birdnet_stm32.models.dscnn import build_dscnn_model
    from birdnet_stm32.models.runners import load_keras_model

    sr, cd, classes = 22050, 3, ["class_a", "class_b"]
    for split in ("train", "val"):
        for k, cls in enumerate(classes + ["noise"]):
            d = tmp_path / split / cls
            d.mkdir(parents=True)
            for i in range(4):
                t = np.linspace(0, cd, sr * cd, endpoint=False, dtype=np.float32)
                freq = (1000.0, 3000.0, 0.0)[k] + 150 * i
                audio = (
                    0.5 * np.sin(2 * np.pi * freq * t) if freq else 0.01 * np.random.default_rng(i).normal(size=t.size)
                )
                sf.write(str(d / f"{split}_{cls}_{i}.wav"), audio.astype(np.float32), sr)
    model = build_dscnn_model(
        num_mels=32,
        spec_width=64,
        sample_rate=sr,
        chunk_duration=cd,
        embeddings_size=32,
        num_classes=2,
        audio_frontend="librosa",
        alpha=0.25,
        mag_scale="none",
    )
    ckpt = tmp_path / "model.keras"
    model.save(ckpt)
    cfg = {
        "sample_rate": sr,
        "num_mels": 32,
        "spec_width": 64,
        "fft_length": 512,
        "chunk_duration": cd,
        "audio_frontend": "librosa",
        "mag_scale": "none",
        "embeddings_size": 32,
        "alpha": 0.25,
        "num_classes": 2,
        "class_names": classes,
        "extra_key_kept": 1,
    }
    (tmp_path / "model_model_config.json").write_text(json.dumps(cfg))
    out = tmp_path / "ft" / "model_ft.keras"
    argv = [
        "finetune-head",
        "--checkpoint_path",
        str(ckpt),
        "--data_path_train",
        str(tmp_path / "train"),
        "--data_path_val",
        str(tmp_path / "val"),
        "--output_path",
        str(out),
        "--num_workers",
        "0",
        "--epochs",
        "3",
        "--learning_rate",
        "0.01",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    finetune_head.main()
    ft = load_keras_model(str(out))
    before, after = model.get_weights(), ft.get_weights()
    names = [w.path for w in model.weights]  # Keras 3: name is "kernel", path is "pred/kernel"
    changed = [n for n, a, b in zip(names, before, after, strict=True) if not np.array_equal(a, b)]
    assert changed and all(n.startswith("pred/") for n in changed)
    assert json.loads((out.parent / "model_ft_model_config.json").read_text()) == cfg  # copied verbatim
    report = json.loads((out.parent / "model_ft_head_finetune.json").read_text())
    assert report["training_files"] == 12 and report["selected_epoch"] >= 1
    # a second run reuses the complete cache
    out2 = tmp_path / "ft" / "model_ft2.keras"
    argv2 = [a if a != str(out) else str(out2) for a in argv]
    argv2[argv2.index("--epochs") + 1] = "1"
    monkeypatch.setattr(sys, "argv", argv2 + ["--cache_dir", str(out.parent / "head_cache")])
    finetune_head.main()
    assert out2.exists()
