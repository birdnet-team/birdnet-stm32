"""Model architectures, audio frontend layer, magnitude scaling, and inference runners.

Use :func:`build_model` to create a model by name::

    from birdnet_stm32.models import build_model
    model = build_model("dscnn", num_mels=64, spec_width=256, ...)
"""

from __future__ import annotations

from typing import Any, Callable

import tensorflow as tf

# Model registry: name -> builder function
_MODEL_REGISTRY: dict[str, Callable[..., tf.keras.Model]] = {}


def register_model(name: str):
    """Decorator to register a model builder function.

    The decorated function must accept keyword arguments and return an
    uncompiled ``tf.keras.Model``.

    Args:
        name: Canonical model name (e.g. "dscnn").
    """
    def decorator(fn: Callable[..., tf.keras.Model]) -> Callable[..., tf.keras.Model]:
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered.")
        _MODEL_REGISTRY[name] = fn
        return fn
    return decorator


def build_model(name: str, **kwargs: Any) -> tf.keras.Model:
    """Build a model by registered name.

    Args:
        name: Model architecture name (e.g. "dscnn").
        **kwargs: Forwarded to the model builder.

    Returns:
        Uncompiled Keras model.

    Raises:
        KeyError: If no model with the given name is registered.
    """
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model: '{name}'. Available: {list_models()}")
    return _MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """Return all registered model names."""
    return sorted(_MODEL_REGISTRY.keys())


# Register built-in models on import
from birdnet_stm32.models.dscnn import build_dscnn_model as _build_dscnn  # noqa: E402

_MODEL_REGISTRY["dscnn"] = _build_dscnn

