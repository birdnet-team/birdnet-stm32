"""Superseded: the frontend fix is ``RAW_SPLIT`` (exact, no precision cost). Kept
as the experiment that first confirmed the defect scales with the accumulated
signal.

Widen the raw-waveform quantisation scale of a birdnet-stm32 INT8 model.

The N6 NPU's convolution accumulator saturates at signed 16 bits. The learned
raw filterbank convolves 448 taps (fold 112 x kernel 4) that, on real audio,
accumulate coherently to ~133k -- four times over the limit -- so the low
filters collapse on device while the host is exact.

Widening the scale of the int8 waveform tensors feeding the filterbank shrinks
the input codes, and with them the accumulator, by the same factor. The stored
scales still describe the same real-world values, so the convolution's
requantisation multiplier compensates exactly and no downstream scale changes.
The only cost is waveform quantisation precision.
"""

import sys

import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema

src, dst, factor = sys.argv[1], sys.argv[2], float(sys.argv[3])
with open(src, "rb") as fh:
    buf = bytearray(fh.read())
model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(buf, 0))
g = model.subgraphs[0]

# The waveform carried as int8: everything from the input QUANTIZE through the
# slice and the polyphase reshape, i.e. every int8 tensor whose last dimension
# is the fold (or 1) and whose element count matches the chunk length.
patched = []
for i, t in enumerate(g.tensors):
    if (
        t.type != schema.TensorType.INT8
        or t.quantization is None
        or t.quantization.scale is None
        or len(t.quantization.scale) == 0
    ):
        continue
    shape = list(t.shape)
    n = 1
    for d in shape:
        n *= d
    # Only the waveform itself: the int8 chunk as [1,T,1], the slice that trims
    # it to a whole number of folds, and the polyphase view [1,1,frames,fold].
    is_wave = (len(shape) == 3 and shape[0] == 1 and shape[2] == 1 and n >= 50000) or (
        len(shape) == 4 and shape[0] == 1 and shape[1] == 1 and n >= 50000
    )
    if is_wave and len(t.quantization.scale) == 1:
        old = t.quantization.scale[0]
        t.quantization.scale = [old * factor]
        patched.append((i, t.name.decode() if isinstance(t.name, bytes) else t.name, shape, old, old * factor))

for i, name, shape, o, n in patched:
    print(f"  tensor {i:3d} {name:28s} {shape} scale {o:.8f} -> {n:.8f}")
if not patched:
    raise SystemExit("no waveform tensors found")

b = flatbuffers.Builder(1024)
b.Finish(model.Pack(b), b"TFL3")
with open(dst, "wb") as fh:
    fh.write(b.Output())
print(f"patched {len(patched)} tensor(s) -> {dst}")
