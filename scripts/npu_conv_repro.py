"""Minimal reproduction of the STM32N6 raw-filterbank convolution defect.

Builds a single-Conv2D INT8 TFLite model with the same geometry as the raw
frontend's learned filterbank (112 input channels, 1x4 kernel, stride 2, 64
filters) and lets the weights and the calibration source be varied
independently. Validating each variant with `stedgeai validate --mode target`
isolates what the NPU actually mishandles.

Measured 2026-09-09 (cosine, target vs host):

    random Gaussian weights ............ 0.99999  pass
    uniform weights (denser, larger) ... 0.99997  pass
    random weights, real zero pattern .. 0.99999  pass
    10x per-channel scale spread ....... 0.99999  pass
    real trained filterbank ............ 0.756    FAIL
    real filterbank, values shuffled ... 0.820    FAIL
    real filterbank, top 25% of taps ... 0.99981  pass
    real filterbank, 4-bit weights ..... 0.967

Only the trained weights fail, and shuffling them keeps it failing, so it is
the value distribution rather than the geometry, the calibration, the bias, the
sparsity or the scale spread.

The frontend now avoids it by splitting the filterbank into channel-group
convolutions and summing them (``RAW_SPLIT`` in birdnet_stm32/models/frontend.py),
which with the real weights measures 0.864 at 2 groups, 0.99956 at 4 and
0.99950 at 8. This script keeps the undivided geometry on purpose: it is the
reproduction, for reporting upstream. See docs/dev/audio-frontends.md.

Usage:
    python scripts/npu_conv_repro.py <weights> <calib> <out.tflite>

    weights: random | uniform | randzb | real | shuffled | maskrand
             | realtop | realbits<N> | spread<RATIO>
    calib:   random | audio | headroom<FACTOR>
"""

import glob
import os
import sys

import numpy as np
import soundfile as sf
import tensorflow as tf

W, C, K, S, F = 535, 112, 4, 2, 64
weights_src = sys.argv[1]  # "random" | "real"
calib_src = sys.argv[2]  # "random" | "audio"
out = sys.argv[3]
# Source of the trained filterbank and of the audio used for calibration. The
# "real"/"shuffled"/"maskrand"/"realtop"/"realbits*" modes read
# <run>/int8/smoke25.tflite; the "audio" calibration reads <run>/sdcard/audio/*.WAV.
run = os.environ.get("NPU_REPRO_RUN", "/data3/ssw_magpie_rt_model/checks/smoke25/run")
rng = np.random.default_rng(0)


def folded_audio(n):
    """Real waveforms, peak-normalised and folded exactly as the graph does."""
    files = sorted(glob.glob(f"{run}/sdcard/audio/*.WAV"))
    out = []
    for f in files[:n]:
        a, _ = sf.read(f, dtype="float32")
        a = a / (float(np.max(np.abs(a))) + 1e-6)
        a = a[: (len(a) // C) * C][: W * C]
        out.append(a.reshape(1, 1, W, C))
    return out


if weights_src in ("real", "shuffled"):
    # Dequantised weights of the first frontend conv, taken from the INT8 model.
    it = tf.lite.Interpreter(model_path=f"{run}/int8/smoke25.tflite")
    it.allocate_tensors()
    cand = [
        d
        for d in it.get_tensor_details()
        if len(d["shape"]) == 4 and tuple(d["shape"][1:]) == (1, K, C) and d["shape"][0] == F
    ]
    d = cand[0]
    q = it.get_tensor(d["index"]).astype("float32")
    sc = d["quantization_parameters"]["scales"]
    kern = (q * sc.reshape(-1, 1, 1, 1)).transpose(1, 2, 3, 0)  # -> (1,K,C,F)
    bias = np.zeros((F,), "float32")
    print(f"real weights from tensor '{d['name']}' shape={q.shape} scales={sc.min():.6g}..{sc.max():.6g}")
    if weights_src == "shuffled":
        # Same values and per-filter scales, structure destroyed.
        flat = kern.reshape(-1, F)
        for j in range(F):
            rng.shuffle(flat[:, j])
        kern = flat.reshape(1, K, C, F)
        print("weights shuffled within each output filter")
elif weights_src.startswith("realbits"):
    # Real weights coarsened to N bits: if the device already computes at that
    # precision, handing it pre-coarsened weights should make it match the host.
    bits = int(weights_src[len("realbits") :])
    it = tf.lite.Interpreter(model_path=f"{run}/int8/smoke25.tflite")
    it.allocate_tensors()
    d = [
        x
        for x in it.get_tensor_details()
        if len(x["shape"]) == 4 and tuple(x["shape"][1:]) == (1, K, C) and x["shape"][0] == F
    ][0]
    q = it.get_tensor(d["index"]).astype("float32")
    sc = d["quantization_parameters"]["scales"]
    step = 2 ** (8 - bits)
    q = np.round(q / step) * step
    kern = (q * sc.reshape(-1, 1, 1, 1)).transpose(1, 2, 3, 0)
    bias = np.zeros((F,), "float32")
    print(f"realbits{bits}: step={step}, distinct |q| levels={len(np.unique(np.abs(q)))}")
elif weights_src == "realtop":
    # Real weights, but only the largest 25% of taps per filter: same values,
    # far fewer accumulated products. Separates "which values" from "how many".
    it = tf.lite.Interpreter(model_path=f"{run}/int8/smoke25.tflite")
    it.allocate_tensors()
    d = [
        x
        for x in it.get_tensor_details()
        if len(x["shape"]) == 4 and tuple(x["shape"][1:]) == (1, K, C) and x["shape"][0] == F
    ][0]
    q = it.get_tensor(d["index"]).astype("float32")
    sc = d["quantization_parameters"]["scales"]
    kern = (q * sc.reshape(-1, 1, 1, 1)).transpose(1, 2, 3, 0)
    flat = kern.reshape(-1, F)
    for j in range(F):
        col = flat[:, j]
        keep = int(0.25 * np.count_nonzero(col))
        if keep < 1:
            continue
        thr = np.sort(np.abs(col))[-keep]
        col[np.abs(col) < thr] = 0.0
    kern = flat.reshape(1, K, C, F)
    bias = np.zeros((F,), "float32")
    print(f"realtop: nonzeros now {int(np.count_nonzero(kern))}/{kern.size}")
elif weights_src.startswith("spread"):
    # Random weights with a deliberate per-output-channel scale spread, to test
    # whether a wide spread of per-channel requant multipliers is what breaks it.
    ratio = float(weights_src[len("spread") :])
    kern = (rng.standard_normal((1, K, C, F)) * 0.05).astype("float32")
    gains = ratio ** (np.arange(F) / (F - 1))
    kern = (kern * gains.reshape(1, 1, 1, F)).astype("float32")
    bias = np.zeros((F,), "float32")
    print(f"spread ratio target={ratio}")
elif weights_src == "maskrand":
    # Random values wearing the real filterbank's zero pattern: tests whether the
    # sparsity (47% exact zeros), rather than the values, is what breaks the NPU.
    it = tf.lite.Interpreter(model_path=f"{run}/int8/smoke25.tflite")
    it.allocate_tensors()
    d = [
        x
        for x in it.get_tensor_details()
        if len(x["shape"]) == 4 and tuple(x["shape"][1:]) == (1, K, C) and x["shape"][0] == F
    ][0]
    q = it.get_tensor(d["index"])
    mask = (q != 0).astype("float32").transpose(1, 2, 3, 0)
    kern = (rng.standard_normal((1, K, C, F)) * 0.05).astype("float32") * mask
    bias = np.zeros((F,), "float32")
    print(f"maskrand: zeros={int((mask == 0).sum())}/{mask.size}")
elif weights_src == "randzb":
    # Random weights but a ZERO bias, matching how the real filterbank is built.
    kern = (rng.standard_normal((1, K, C, F)) * 0.05).astype("float32")
    bias = np.zeros((F,), "float32")
elif weights_src == "uniform":
    # Fills the int8 range the way the trained filterbank does, but unstructured.
    kern = (rng.uniform(-1.0, 1.0, (1, K, C, F)) * 0.05).astype("float32")
    bias = (rng.standard_normal((F,)) * 0.01).astype("float32")
else:
    kern = (rng.standard_normal((1, K, C, F)) * 0.05).astype("float32")
    bias = (rng.standard_normal((F,)) * 0.01).astype("float32")

inp = tf.keras.Input(shape=(1, W, C), name="x")
y = tf.keras.layers.Conv2D(F, (1, K), strides=(1, S), padding="valid", use_bias=True)(inp)
m = tf.keras.Model(inp, y)
m.layers[-1].set_weights([kern, bias])

if calib_src.startswith("headroom"):
    # Calibrate with deliberate headroom: the input scale is set by a range the
    # real signal never reaches, so live audio occupies a smaller int8 span and
    # the first conv's accumulator stays inside the NPU's 16-bit limit.
    hr = float(calib_src[len("headroom") :])
    samples = folded_audio(25) + [folded_audio(1)[0] * hr]
elif calib_src == "audio":
    samples = folded_audio(25)
else:
    samples = [rng.standard_normal((1, 1, W, C)).astype("float32") for _ in range(25)]


def rep():
    for s in samples:
        yield [s.astype("float32")]


conv = tf.lite.TFLiteConverter.from_keras_model(m)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.float32
conv.inference_output_type = tf.float32
with open(out, "wb") as fh:
    fh.write(conv.convert())
print(f"wrote {out} weights={weights_src} calib={calib_src}")
