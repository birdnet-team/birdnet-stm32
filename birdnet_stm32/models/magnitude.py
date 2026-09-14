"""MagnitudeScalingLayer: composable magnitude scaling for spectrograms.

Supports two modes:
- 'none': Pass-through, the ablation baseline.
- 'pwl': Learned piecewise-linear scaling via 1x1 depthwise branches + ReLU + Add.

PCEN and dB were removed. dB's log op produces a dynamic range INT8 cannot
hold, which is the failure this frontend exists to avoid, and PCEN was never
used by any release.
"""

import tensorflow as tf
from tensorflow.keras import layers

from birdnet_stm32.models.quantization import clip_activation, validate_bounds

VALID_MAG_SCALES = ("none", "pwl", "cpwl")


@tf.keras.utils.register_keras_serializable(package="birdnet_stm32")
class NonPositive(tf.keras.constraints.Constraint):
    """Constrain weights to be <= 0 (the compressive PWL's hinge slopes)."""

    def __call__(self, w):
        return -tf.nn.relu(-w)

    def get_config(self):
        return {}


class MagnitudeScalingLayer(layers.Layer):
    """Channel-wise magnitude scaling as a standalone Keras layer.

    Accepts 4-D tensors [B, H, W, C] and applies the selected scaling
    independently per channel. All sub-layers use 1x1 depthwise convolutions
    so the layer is NPU-friendly.

    Args:
        method: 'none' | 'pwl'.
        channels: Number of input channels (typically mel_bins).
        is_trainable: Whether sub-layer weights are trainable.
        name: Layer name.
    """

    def __init__(
        self,
        method: str = "none",
        channels: int = 64,
        is_trainable: bool = False,
        activation_bounds: dict | None = None,
        name: str = "mag_scale",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if method not in VALID_MAG_SCALES:
            raise ValueError(f"Invalid mag_scale: '{method}'. Valid options: {VALID_MAG_SCALES}")
        self.method = method
        self.channels = int(channels)
        self.is_trainable = bool(is_trainable)
        self.activation_bounds = validate_bounds(activation_bounds)
        self._quantization_hook = None

        # PWL sublayers. "pwl" is the learned hinge sum with unconstrained slopes;
        # it initializes expansive (cumulative slope 0.40 -> 0.88) and stays so
        # after training, which gives its output a heavy upper tail: on the V12
        # raw model 50/90/99% of values used 1/3/14 of the 255 INT8 codes, the
        # rare peaks setting the range. "cpwl" is the same hinge sum held
        # compressive -- k0 >= 0, hinge input weights >= 0, hinge slopes <= 0 --
        # and initialized log-like (slopes 1.0 -> 0.55 -> 0.25 -> 0.10), so the
        # tail is squeezed and typical values keep their INT8 resolution. Same
        # ops, same names; only the constraints and the init differ.
        if self.method in ("pwl", "cpwl"):
            compressive = self.method == "cpwl"
            k0_init = 1.0 if compressive else 0.40
            k_inits = (-0.45, -0.30, -0.15) if compressive else (0.25, 0.15, 0.08)
            non_neg = tf.keras.constraints.NonNeg() if compressive else None
            non_pos = NonPositive() if compressive else None
            self._pwl_k0_dw = layers.DepthwiseConv2D(
                (1, 1),
                use_bias=False,
                depthwise_initializer=tf.keras.initializers.Constant(k0_init),
                depthwise_constraint=non_neg,
                padding="same",
                name=f"{name}_pwl_k0_dw",
                trainable=self.is_trainable,
            )
            self._pwl_shift_dws = [
                layers.DepthwiseConv2D(
                    (1, 1),
                    use_bias=True,
                    depthwise_initializer=tf.keras.initializers.Ones(),
                    depthwise_constraint=tf.keras.constraints.NonNeg() if compressive else None,
                    bias_initializer=tf.keras.initializers.Constant(-t),
                    padding="same",
                    name=f"{name}_pwl_shift{i + 1}_dw",
                    trainable=self.is_trainable,
                )
                for i, t in enumerate((0.10, 0.35, 0.65))
            ]
            self._pwl_k_dws = [
                layers.DepthwiseConv2D(
                    (1, 1),
                    use_bias=False,
                    depthwise_initializer=tf.keras.initializers.Constant(k),
                    depthwise_constraint=non_pos,
                    padding="same",
                    name=f"{name}_pwl_k{i + 1}_dw",
                    trainable=self.is_trainable,
                )
                for i, k in enumerate(k_inits)
            ]
        else:
            self._pwl_k0_dw = None
            self._pwl_shift_dws = []
            self._pwl_k_dws = []

    def build(self, input_shape):
        """Build magnitude scaling sub-layers for the given input shape."""
        if self.method in ("pwl", "cpwl"):
            if self._pwl_k0_dw is not None and not self._pwl_k0_dw.built:
                self._pwl_k0_dw.build(input_shape)
            for s in self._pwl_shift_dws:
                if not s.built:
                    s.build(input_shape)
            for k in self._pwl_k_dws:
                if not k.built:
                    k.build(input_shape)
        super().build(input_shape)

    def call(self, x, training=None):
        """Apply magnitude scaling to a 4-D tensor [B, H, W, C]."""
        if self.method in ("pwl", "cpwl"):
            return self._apply_pwl(x)
        return x

    def set_quantization_hook(self, hook) -> None:
        """Install or remove a training-only internal quantization hook."""
        self._quantization_hook = hook

    def _quantized_call(self, layer, inputs):
        """Call a kernel layer through the QAT hook when one is installed."""
        if self._quantization_hook is None:
            return layer(inputs)
        return self._quantization_hook.kernel(layer, inputs)

    def _quantized_activation(self, name: str, inputs):
        """Mark an internal tensor as an INT8 activation boundary for QAT."""
        inputs = clip_activation(inputs, self.activation_bounds, name)
        if self._quantization_hook is None:
            return inputs
        return self._quantization_hook.activation(name, inputs)

    def _apply_pwl(self, x):
        """Learned hinge sum; slopes are not constrained to be compressive."""
        branches = []
        if self._pwl_k0_dw is not None:
            branch = self._quantized_call(self._pwl_k0_dw, x)
            branches.append(self._quantized_activation(self._pwl_k0_dw.name, branch))
        for shift_dw, k_dw in zip(self._pwl_shift_dws, self._pwl_k_dws, strict=True):
            shifted = self._quantized_call(shift_dw, x)
            relu = self._quantized_activation(f"{shift_dw.name}_relu", tf.nn.relu(shifted))
            branch = self._quantized_call(k_dw, relu)
            branches.append(self._quantized_activation(k_dw.name, branch))
        if not branches:
            return x
        y = branches[0]
        for j, b in enumerate(branches[1:], start=1):
            name = f"{self.name}_pwl_add_{j}"
            y = self._quantized_activation(name, tf.add(y, b, name=name))
        return y

    def compute_output_shape(self, input_shape):
        """Output shape is identical to input shape."""
        return input_shape

    # Constructor arguments retired in 1.2.0 along with the features behind
    # them. Checkpoints saved before that still carry them in their serialized
    # layer config, so they are dropped on load rather than rejected: removing
    # a training option must not make existing models unreadable.
    _RETIRED_CONFIG_KEYS = ("pcen_K", "pcen_pool_width")

    @classmethod
    def from_config(cls, config):
        """Build from a serialized config, ignoring retired arguments."""
        return cls(**{key: value for key, value in config.items() if key not in cls._RETIRED_CONFIG_KEYS})

    def get_config(self):
        """Return a serializable configuration dict."""
        cfg = super().get_config()
        cfg.update(
            {
                "method": self.method,
                "channels": self.channels,
                "is_trainable": self.is_trainable,
                "activation_bounds": self.activation_bounds,
            }
        )
        return cfg
