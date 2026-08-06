"""MagnitudeScalingLayer: composable magnitude scaling for spectrograms.

Supports four modes:
- 'none': Pass-through.
- 'pwl': Piecewise-linear compression via 1x1 depthwise branches + ReLU + Add.
- 'pcen': PCEN-like compression (pool/conv/ReLU/Add approximation).
- 'db': Log compression (10*log10) — avoid for PTQ deployment.
"""

import tensorflow as tf
from tensorflow.keras import layers

VALID_MAG_SCALES = ("none", "pwl", "pcen", "db")


class MagnitudeScalingLayer(layers.Layer):
    """Channel-wise magnitude scaling as a standalone Keras layer.

    Accepts 4-D tensors [B, H, W, C] and applies the selected scaling
    independently per channel. All sub-layers use 1x1 depthwise convolutions
    so the layer is NPU-friendly.

    Args:
        method: 'none' | 'pwl' | 'pcen' | 'db'.
        channels: Number of input channels (typically mel_bins).
        pcen_K: Number of average-pooling stages for PCEN smoothing.
        pcen_pool_width: Width (in time frames) of each PCEN smoothing stage.
            Stacking ``pcen_K`` stages of this width gives an effective
            smoothing support of ``pcen_K * (pcen_pool_width - 1) + 1`` frames,
            which is what stands in for PCEN's IIR envelope follower.
        is_trainable: Whether sub-layer weights are trainable.
        name: Layer name.
    """

    def __init__(
        self,
        method: str = "none",
        channels: int = 64,
        pcen_K: int = 8,
        pcen_pool_width: int = 3,
        is_trainable: bool = False,
        name: str = "mag_scale",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if method not in VALID_MAG_SCALES:
            raise ValueError(f"Invalid mag_scale: '{method}'. Valid options: {VALID_MAG_SCALES}")
        self.method = method
        self.channels = int(channels)
        self.pcen_K = int(pcen_K)
        self.pcen_pool_width = max(1, int(pcen_pool_width))
        self.is_trainable = bool(is_trainable)
        self._quantization_hook = None

        # DB constants
        self._db_eps = 1e-6
        self._db_ref = 1.0

        # PCEN sublayers
        if self.method == "pcen":
            # Smoothing runs along the time axis (W). The frontend feeds this
            # layer as [B, 1, T, C], so pooling over W is pooling over frames;
            # a unit pool would make the whole AGC branch an identity.
            self._pcen_pools = [
                layers.AveragePooling2D(
                    pool_size=(1, self.pcen_pool_width),
                    strides=(1, 1),
                    padding="same",
                    name=f"{name}_pcen_ema{k}",
                )
                for k in range(self.pcen_K)
            ]
            self._pcen_agc_dw = layers.DepthwiseConv2D(
                (1, 1),
                use_bias=False,
                depthwise_initializer=tf.keras.initializers.Constant(0.6),
                padding="same",
                name=f"{name}_pcen_agc_dw",
                trainable=self.is_trainable,
            )
            self._pcen_k1_dw = layers.DepthwiseConv2D(
                (1, 1),
                use_bias=False,
                depthwise_initializer=tf.keras.initializers.Constant(0.15),
                padding="same",
                name=f"{name}_pcen_k1_dw",
                trainable=self.is_trainable,
            )
            self._pcen_shift_dw = layers.DepthwiseConv2D(
                (1, 1),
                use_bias=True,
                depthwise_initializer=tf.keras.initializers.Ones(),
                bias_initializer=tf.keras.initializers.Constant(-0.2),
                padding="same",
                name=f"{name}_pcen_shift_dw",
                trainable=self.is_trainable,
            )
            self._pcen_k2mk1_dw = layers.DepthwiseConv2D(
                (1, 1),
                use_bias=False,
                depthwise_initializer=tf.keras.initializers.Constant(0.45),
                padding="same",
                name=f"{name}_pcen_k2mk1_dw",
                trainable=self.is_trainable,
            )
        else:
            self._pcen_pools = []
            self._pcen_agc_dw = None
            self._pcen_k1_dw = None
            self._pcen_shift_dw = None
            self._pcen_k2mk1_dw = None

        # PWL sublayers
        if self.method == "pwl":
            self._pwl_k0_dw = layers.DepthwiseConv2D(
                (1, 1),
                use_bias=False,
                depthwise_initializer=tf.keras.initializers.Constant(0.40),
                padding="same",
                name=f"{name}_pwl_k0_dw",
                trainable=self.is_trainable,
            )
            self._pwl_shift_dws = [
                layers.DepthwiseConv2D(
                    (1, 1),
                    use_bias=True,
                    depthwise_initializer=tf.keras.initializers.Ones(),
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
                    padding="same",
                    name=f"{name}_pwl_k{i + 1}_dw",
                    trainable=self.is_trainable,
                )
                for i, k in enumerate((0.25, 0.15, 0.08))
            ]
        else:
            self._pwl_k0_dw = None
            self._pwl_shift_dws = []
            self._pwl_k_dws = []

    def build(self, input_shape):
        """Build magnitude scaling sub-layers for the given input shape."""
        if self.method == "pcen":
            for pool in self._pcen_pools:
                if not pool.built:
                    pool.build(input_shape)
            for dw in (self._pcen_agc_dw, self._pcen_k1_dw, self._pcen_shift_dw, self._pcen_k2mk1_dw):
                if dw is not None and not dw.built:
                    dw.build(input_shape)
        if self.method == "pwl":
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
        if self.method == "pcen":
            return self._apply_pcen(x)
        if self.method == "pwl":
            return self._apply_pwl(x)
        if self.method == "db":
            return self._apply_db(x)
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
        if self._quantization_hook is None:
            return inputs
        return self._quantization_hook.activation(name, inputs)

    def _apply_pcen(self, x):
        """PCEN-like compression using only pool/conv/ReLU/Add ops."""
        m = x
        for pool in self._pcen_pools:
            m = pool(m)
        agc = self._quantized_call(self._pcen_agc_dw, m) if self._pcen_agc_dw is not None else m
        agc = self._quantized_activation(f"{self.name}_pcen_agc", agc)
        y0 = self._quantized_activation(f"{self.name}_pcen_relu0", tf.nn.relu(x - agc))
        b1 = self._quantized_call(self._pcen_k1_dw, y0) if self._pcen_k1_dw is not None else y0
        b1 = self._quantized_activation(f"{self.name}_pcen_branch1", b1)
        y_shift = self._quantized_call(self._pcen_shift_dw, y0) if self._pcen_shift_dw is not None else y0
        relu = self._quantized_activation(f"{self.name}_pcen_relu1", tf.nn.relu(y_shift))
        b2 = self._quantized_call(self._pcen_k2mk1_dw, relu) if self._pcen_k2mk1_dw is not None else relu
        b2 = self._quantized_activation(f"{self.name}_pcen_branch2", b2)
        return self._quantized_activation(f"{self.name}_pcen_output", tf.nn.relu(b1 + b2))

    def _apply_pwl(self, x):
        """Piecewise-linear compression via 1x1 depthwise branches."""
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

    def _apply_db(self, x):
        """Apply dB log compression (10 * log10)."""
        eps = tf.cast(self._db_eps, x.dtype)
        ref = tf.cast(self._db_ref, x.dtype)
        log10 = tf.math.log(tf.cast(10.0, x.dtype))
        safe = tf.maximum(x, eps)
        return tf.cast(10.0, x.dtype) * tf.math.log(safe / ref) / log10

    def compute_output_shape(self, input_shape):
        """Output shape is identical to input shape."""
        return input_shape

    def get_config(self):
        """Return a serializable configuration dict."""
        cfg = super().get_config()
        cfg.update(
            {
                "method": self.method,
                "channels": self.channels,
                "pcen_K": self.pcen_K,
                "pcen_pool_width": self.pcen_pool_width,
                "is_trainable": self.is_trainable,
            }
        )
        return cfg
