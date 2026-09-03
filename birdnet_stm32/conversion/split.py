"""Split a trained model into a fixed backbone and a swappable classifier head.

The backbone maps audio to an embedding vector and is flashed once. The head
maps that embedding to per-class probabilities and is the only part that has to
change when the species list changes, so it is the part that travels over a
narrowband satellite link. Keeping it a separate artifact means an update costs
kilobytes instead of the whole model.

The split point is the pooling layer that produces the embedding vector. In a
DS-CNN everything after it is a short chain (dropout, then the classifier
``Dense``), which is rebuilt onto a clean ``embeddings`` input so the head is a
standalone model with its own weights.
"""

import gzip
import hashlib
import json
import os
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Layers that collapse the feature map into the embedding vector. The custom
# AttentionPooling layer is matched by name because it is not a Keras builtin.
POOLING_LAYER_TYPES = (
    layers.GlobalAveragePooling2D,
    layers.GlobalMaxPooling2D,
    layers.Flatten,
)
POOLING_LAYER_NAMES = ("attn_pool", "gap")


def find_embedding_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    """Return the layer whose output is the model's embedding vector.

    Args:
        model: Functional classifier model.

    Returns:
        The last pooling layer in the graph.

    Raises:
        ValueError: If no pooling layer is present.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, POOLING_LAYER_TYPES):
            return layer
        if any(token in layer.name for token in POOLING_LAYER_NAMES):
            return layer
    raise ValueError("Could not locate an embedding layer (global pooling, flatten, or attention pooling) in the model")


def _head_layers(model: tf.keras.Model, embedding_layer: tf.keras.layers.Layer) -> list[tf.keras.layers.Layer]:
    """Return the layers after the embedding layer, in graph order.

    Raises:
        ValueError: If the head is not a single chain of one-input layers.
    """
    index = model.layers.index(embedding_layer)
    head = model.layers[index + 1 :]
    if not head:
        raise ValueError("Model has no classifier head after its embedding layer")
    for layer in head:
        inbound = layer._inbound_nodes  # noqa: SLF001
        if len(inbound) != 1 or len(inbound[0].input_tensors) != 1:
            raise ValueError(
                f"Classifier head layer '{layer.name}' is not part of a single chain; "
                "splitting supports heads that apply one layer after another"
            )
    if head[-1].output is not model.outputs[0]:
        raise ValueError("The last head layer does not produce the model output; the head may be branched")
    return head


def split_model(model: tf.keras.Model) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Split *model* into a backbone and an independent classifier head.

    The backbone shares layers with *model*; the head is rebuilt from cloned
    layers so neither model mutates the source graph.

    Args:
        model: Trained functional classifier.

    Returns:
        Tuple of ``(backbone, classifier)``. The backbone maps the model input
        to the embedding vector; the classifier maps an ``embeddings`` input to
        the class probabilities.

    Raises:
        ValueError: If the model has no embedding layer or a branched head.
    """
    embedding_layer = find_embedding_layer(model)
    head = _head_layers(model, embedding_layer)

    backbone = tf.keras.Model(
        model.inputs,
        embedding_layer.output,
        name=f"{model.name}_backbone",
    )

    head_input = tf.keras.Input(
        shape=tuple(embedding_layer.output.shape[1:]),
        dtype=embedding_layer.output.dtype,
        name="embeddings",
    )
    x = head_input
    for layer in head:
        clone = layer.__class__.from_config(layer.get_config())
        x = clone(x)
        if layer.weights:
            clone.set_weights(layer.get_weights())
    classifier = tf.keras.Model(head_input, x, name=f"{model.name}_classifier")
    return backbone, classifier


def embedding_dimension(backbone: tf.keras.Model) -> int:
    """Return the width of the embedding vector a backbone emits."""
    return int(backbone.output_shape[-1])


def backbone_fingerprint(backbone: tf.keras.Model) -> str:
    """Return a SHA-256 over a backbone's float weights in graph order.

    A head shipped over the air only works if the receiver's flashed backbone
    is the one the head was calibrated against. Comparing this fingerprint is
    how a later head-only conversion proves the backbone did not move: weight
    values are hashed together with each tensor's name and shape, so a
    reordered or resized graph cannot collide with the original.
    """
    digest = hashlib.sha256()
    for weight in backbone.weights:
        digest.update(weight.name.encode("utf-8"))
        value = np.ascontiguousarray(weight.numpy())
        digest.update(str(value.shape).encode("utf-8"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def file_sha256(path: str) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_fingerprint(backbone_path: str, fingerprint: str, embedding_dim: int) -> str:
    """Record a backbone's fingerprint beside the artifact it identifies."""
    path = backbone_path + ".fingerprint.json"
    payload = {
        "backbone": os.path.basename(backbone_path),
        "weights_sha256": fingerprint,
        "artifact_sha256": file_sha256(backbone_path),
        "embedding_dim": embedding_dim,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


def read_fingerprint(backbone_path: str) -> dict | None:
    """Return the recorded fingerprint for a backbone, or None if absent."""
    path = backbone_path + ".fingerprint.json"
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def count_parameters(model: tf.keras.Model) -> int:
    """Return the total number of parameters in a model."""
    return int(sum(int(np.prod(weight.shape)) for weight in model.weights))


def gzip_file(source_path: str, output_path: str = "") -> str:
    """Write a deterministic maximum-compression gzip copy of a file.

    The timestamp is zeroed and the source name is left out of the header, so
    repeated compressions of identical bytes produce identical archives and an
    over-the-air update stays diffable.

    Args:
        source_path: File to compress.
        output_path: Destination; defaults to ``source_path + '.gz'``.

    Returns:
        The path written.
    """
    output_path = output_path or source_path + ".gz"
    with (
        open(source_path, "rb") as source,
        open(output_path, "wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as destination,
    ):
        shutil.copyfileobj(source, destination)
    return output_path


def weight_sparsity(model_path: str, min_tensor_size: int = 256) -> dict[str, float | int]:
    """Measure the fraction of zero INT8 weights inside a TFLite model.

    A pruned head compresses well precisely because these bytes are zero, so
    the split report records the number the compression ratio follows from.

    Args:
        model_path: Path to a .tflite model.
        min_tensor_size: Ignore tensors smaller than this (biases, shapes).

    Returns:
        Dict with the counted weights, the zeros among them, and their ratio.
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    total = 0
    zeros = 0
    for detail in interpreter.get_tensor_details():
        if detail["dtype"] != np.int8:
            continue
        try:
            tensor = interpreter.get_tensor(detail["index"])
        except ValueError:
            # Activation tensors have no stored buffer to read.
            continue
        if tensor.size < min_tensor_size:
            continue
        total += int(tensor.size)
        zeros += int(tensor.size - np.count_nonzero(tensor))
    return {
        "int8_weights": total,
        "int8_zeros": zeros,
        "int8_sparsity": zeros / float(total) if total else 0.0,
    }


def size_record(model_path: str) -> dict:
    """Return raw and gzipped sizes plus INT8 sparsity for a TFLite artifact."""
    gzip_path = gzip_file(model_path)
    raw_bytes = os.path.getsize(model_path)
    gzip_bytes = os.path.getsize(gzip_path)
    record = {
        "path": os.path.basename(model_path),
        "gzip_path": os.path.basename(gzip_path),
        "size_bytes": raw_bytes,
        "gzip_bytes": gzip_bytes,
        "gzip_ratio": raw_bytes / max(gzip_bytes, 1),
    }
    record.update(weight_sparsity(model_path))
    return record
