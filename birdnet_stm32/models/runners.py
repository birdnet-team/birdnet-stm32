"""Inference runners for Keras and TFLite models.

Provides a uniform predict(x_batch) interface for both model formats,
enabling the evaluation pipeline to be agnostic to the model type.
"""

import os
import warnings

import numpy as np
import tensorflow as tf

from birdnet_stm32.models.frontend import AudioFrontendLayer
from birdnet_stm32.models.magnitude import MagnitudeScalingLayer

_KERAS_CUSTOM_OBJECTS = {
    "AudioFrontendLayer": AudioFrontendLayer,
    "MagnitudeScalingLayer": MagnitudeScalingLayer,
}


def load_keras_model(model_path: str) -> tf.keras.Model:
    """Load a project checkpoint with the canonical custom-layer registry."""
    return tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects=_KERAS_CUSTOM_OBJECTS,
    )


def allocated_interpreter(
    model_path: str | None = None, model_content: bytes | None = None, num_threads: int | None = None
) -> tf.lite.Interpreter:
    """Return a TFLite interpreter with its tensors allocated.

    TFLite applies the XNNPACK delegate by default. XNNPACK refuses some valid
    INT8 graphs outright, for example a requantization scale of 256 or more,
    which PTQ produces when a branch calibrates to an almost empty range (a PWL
    hinge that never fires). The builtin reference kernels run those graphs, so
    the interpreter falls back to them with a warning instead of failing.
    Graphs XNNPACK accepts keep running on it, so their outputs are unchanged.

    Args:
        model_path: Path to a .tflite file.
        model_content: Serialized model, instead of a path.
        num_threads: Interpreter threads.

    Returns:
        Interpreter with tensors allocated.
    """
    kwargs = {"model_path": model_path, "model_content": model_content, "num_threads": num_threads}
    interpreter = tf.lite.Interpreter(**kwargs, experimental_delegates=[])
    try:
        interpreter.allocate_tensors()
        return interpreter
    except RuntimeError as exc:
        if "XNNPACK" not in str(exc):
            raise
        warnings.warn(
            f"XNNPACK cannot run {model_path or 'this model'} ({exc}); using the builtin reference kernels.",
            RuntimeWarning,
            stacklevel=2,
        )
    interpreter = tf.lite.Interpreter(
        **kwargs,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interpreter.allocate_tensors()
    return interpreter


class KerasRunner:
    """Thin wrapper for a Keras model to standardize batch prediction."""

    def __init__(self, model: tf.keras.Model):
        """Initialize with a loaded Keras model.

        Args:
            model: Loaded Keras model (compiled=False is fine).
        """
        self.model = model
        # One traced graph per input shape instead of op-by-op eager dispatch,
        # which dominated evaluation time. The graph reads the model's variables,
        # so weights updated after construction are still used.
        self._forward = tf.function(lambda x: self.model(x, training=False), reduce_retracing=True)

    def predict(self, x_batch: np.ndarray) -> np.ndarray:
        """Run a forward pass on a batch.

        Args:
            x_batch: Input batch in the model's expected shape and dtype.

        Returns:
            Model outputs [B, C] as float32.
        """
        x_batch = x_batch.astype(np.float32, copy=False)
        return np.asarray(self._forward(tf.convert_to_tensor(x_batch)), dtype=np.float32)


class TFLiteRunner:
    """TFLite model runner (XNNPACK, or the reference kernels where XNNPACK refuses the graph)."""

    def __init__(self, model_path: str, num_threads: int | None = None):
        """Initialize with a TFLite model file.

        Args:
            model_path: Path to a .tflite model file.
            num_threads: Interpreter threads (default: up to 8).
        """
        threads = num_threads if num_threads is not None else min(8, os.cpu_count() or 1)
        self.interpreter = allocated_interpreter(model_path=model_path, num_threads=threads)
        self.input_index = None
        self.output_index = None
        self._allocate()

    def _allocate(self):
        """Allocate tensors and cache input/output tensor indices."""
        self.interpreter.allocate_tensors()
        in_det = self.interpreter.get_input_details()[0]
        out_det = self.interpreter.get_output_details()[0]
        self.input_index = in_det["index"]
        self.output_index = out_det["index"]

    def _ensure_shape(self, shape: tuple):
        """Resize the interpreter input tensor to match the batch shape if needed.

        Args:
            shape: Desired input tensor shape.
        """
        in_det = self.interpreter.get_input_details()[0]
        cur = in_det["shape"]
        if list(cur) != list(shape):
            self.interpreter.resize_tensor_input(self.input_index, shape)
            self._allocate()

    def predict(self, x_batch: np.ndarray) -> np.ndarray:
        """Run a forward pass on a batch.

        Args:
            x_batch: Input batch in the model's expected shape and dtype.

        Returns:
            Model outputs [B, C] as float32.
        """
        x_batch = x_batch.astype(np.float32, copy=False)
        self._ensure_shape(x_batch.shape)
        self.interpreter.set_tensor(self.input_index, x_batch)
        self.interpreter.invoke()
        return np.asarray(self.interpreter.get_tensor(self.output_index), dtype=np.float32)


class ChainedTFLiteRunner:
    """Run a split backbone and classifier head as one model.

    Conversion can emit the classifier head as its own artifact so it can be
    updated over a narrowband link without reflashing the backbone. Evaluation
    and deployment then need the two halves to behave like the model they came
    from, which is what this runner provides.
    """

    def __init__(self, backbone_path: str, classifier_path: str):
        """Initialize with the two halves of a split model.

        Args:
            backbone_path: Path to the .tflite backbone (audio -> embeddings).
            classifier_path: Path to the .tflite head (embeddings -> scores).
        """
        self.backbone = TFLiteRunner(backbone_path)
        self.classifier = TFLiteRunner(classifier_path)

    def predict(self, x_batch: np.ndarray) -> np.ndarray:
        """Run the backbone and feed its embeddings to the classifier head.

        Args:
            x_batch: Input batch in the backbone's expected shape and dtype.

        Returns:
            Model outputs [B, C] as float32.
        """
        embeddings = self.backbone.predict(x_batch)
        return self.classifier.predict(np.asarray(embeddings, dtype=np.float32))


class SigmoidRunner:
    """Wrap a runner whose model emits logits, and return probabilities.

    A model converted with ``--output_activation logit`` leaves the sigmoid to
    its caller, which keeps its output off the INT8 1/256 probability grid.
    Everything downstream — evaluation, the board comparison, the operational
    gate — expects probabilities, so this puts the sigmoid back. It is 100
    values per window; the cost is not measurable.
    """

    def __init__(self, runner):
        self.runner = runner

    def predict(self, x):
        return 1.0 / (1.0 + np.exp(-np.asarray(self.runner.predict(x), dtype=np.float32)))


def runner_for_config(runner, config: dict | None):
    """Wrap *runner* in the sigmoid when *config* says the model emits logits."""
    if config and config.get("output_activation", "sigmoid") == "logit":
        return SigmoidRunner(runner)
    return runner


def load_model_runner(
    model_path: str,
    classifier_path: str = "",
    config: dict | None = None,
) -> KerasRunner | TFLiteRunner | ChainedTFLiteRunner | SigmoidRunner:
    """Load a .keras or .tflite model and return a runner with predict().

    Args:
        model_path: Path to a saved model (.keras or .tflite). When
            ``classifier_path`` is given, this is the .tflite backbone.
        classifier_path: Optional .tflite classifier head. Supplying it runs
            the split pair as one chained model.
        config: Optional model config. When it says ``output_activation:
            "logit"``, the runner is wrapped so it returns probabilities.

    Returns:
        KerasRunner, TFLiteRunner, or ChainedTFLiteRunner instance.

    Raises:
        ValueError: If a classifier head is paired with a non-TFLite backbone.
    """
    if classifier_path:
        if not model_path.lower().endswith(".tflite"):
            raise ValueError("A classifier head can only be chained onto a .tflite backbone")
        return runner_for_config(ChainedTFLiteRunner(model_path, classifier_path), config)
    if model_path.lower().endswith(".tflite"):
        return runner_for_config(TFLiteRunner(model_path), config)
    return runner_for_config(KerasRunner(load_keras_model(model_path)), config)
