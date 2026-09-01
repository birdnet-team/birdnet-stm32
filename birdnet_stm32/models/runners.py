"""Inference runners for Keras and TFLite models.

Provides a uniform predict(x_batch) interface for both model formats,
enabling the evaluation pipeline to be agnostic to the model type.
"""

import numpy as np
import tensorflow as tf

from birdnet_stm32.models.frontend import AudioFrontendLayer
from birdnet_stm32.models.magnitude import MagnitudeScalingLayer


class KerasRunner:
    """Thin wrapper for a Keras model to standardize batch prediction."""

    def __init__(self, model: tf.keras.Model):
        """Initialize with a loaded Keras model.

        Args:
            model: Loaded Keras model (compiled=False is fine).
        """
        self.model = model
        try:
            self.input_names = [t.name.split(":")[0] for t in self.model.inputs]
        except Exception:
            self.input_names = []

    def predict(self, x_batch: np.ndarray) -> np.ndarray:
        """Run a forward pass on a batch.

        Args:
            x_batch: Input batch in the model's expected shape and dtype.

        Returns:
            Model outputs [B, C] as float32.
        """
        x_batch = x_batch.astype(np.float32, copy=False)
        if getattr(self, "input_names", None) and len(self.input_names) == 1:
            feed = {self.input_names[0]: x_batch}
            try:
                return self.model(feed, training=False).numpy()
            except Exception:
                pass
        return self.model(x_batch, training=False).numpy()


class TFLiteRunner:
    """TFLite model runner using the builtin interpreter (no delegates)."""

    def __init__(self, model_path: str):
        """Initialize with a TFLite model file.

        Args:
            model_path: Path to a .tflite model file.
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path, experimental_delegates=[])
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
        return self.interpreter.get_tensor(self.output_index)


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


def load_model_runner(
    model_path: str,
    classifier_path: str = "",
) -> KerasRunner | TFLiteRunner | ChainedTFLiteRunner:
    """Load a .keras or .tflite model and return a runner with predict().

    Args:
        model_path: Path to a saved model (.keras or .tflite). When
            ``classifier_path`` is given, this is the .tflite backbone.
        classifier_path: Optional .tflite classifier head. Supplying it runs
            the split pair as one chained model.

    Returns:
        KerasRunner, TFLiteRunner, or ChainedTFLiteRunner instance.

    Raises:
        ValueError: If a classifier head is paired with a non-TFLite backbone.
    """
    if classifier_path:
        if not model_path.lower().endswith(".tflite"):
            raise ValueError("A classifier head can only be chained onto a .tflite backbone")
        return ChainedTFLiteRunner(model_path, classifier_path)
    if model_path.lower().endswith(".tflite"):
        return TFLiteRunner(model_path)
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={"AudioFrontendLayer": AudioFrontendLayer, "MagnitudeScalingLayer": MagnitudeScalingLayer},
    )
    return KerasRunner(model)
