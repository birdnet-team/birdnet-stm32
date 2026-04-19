"""Frontend registry: register and discover audio frontend configurations.

Each registered frontend declares its name, whether it's a precomputed
(host-side) or in-model mode, and its N6 NPU compatibility constraints.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrontendInfo:
    """Metadata for a registered audio frontend.

    Attributes:
        name: Canonical frontend name.
        mode: Internal AudioFrontendLayer mode ('precomputed', 'hybrid', 'raw').
        precomputed: Whether spectrogram is computed on the host (True) or in-model (False).
        n6_compatible: Whether this frontend is compatible with STM32N6 NPU deployment.
        description: Short human-readable description.
    """

    name: str
    mode: str
    precomputed: bool
    n6_compatible: bool
    description: str = ""


# Global registry
_REGISTRY: dict[str, FrontendInfo] = {}


def register_frontend(info: FrontendInfo) -> None:
    """Register a frontend configuration.

    Args:
        info: FrontendInfo describing the frontend.

    Raises:
        ValueError: If a frontend with the same name is already registered.
    """
    if info.name in _REGISTRY:
        raise ValueError(f"Frontend '{info.name}' is already registered.")
    _REGISTRY[info.name] = info


def get_frontend_info(name: str) -> FrontendInfo:
    """Look up a registered frontend by name.

    Args:
        name: Canonical frontend name.

    Returns:
        FrontendInfo for the frontend.

    Raises:
        KeyError: If the frontend is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Frontend '{name}' is not registered. Available: {list_frontends()}")
    return _REGISTRY[name]


def list_frontends() -> list[str]:
    """Return all registered frontend names."""
    return sorted(_REGISTRY.keys())


def is_precomputed(name: str) -> bool:
    """Check whether a frontend uses host-side precomputed spectrograms.

    Args:
        name: Canonical frontend name.

    Returns:
        True if the frontend computes spectrograms on the host.
    """
    return get_frontend_info(name).precomputed


def is_n6_compatible(name: str) -> bool:
    """Check whether a frontend is compatible with the STM32N6 NPU.

    Args:
        name: Canonical frontend name.

    Returns:
        True if the frontend can be deployed on the N6 NPU.
    """
    return get_frontend_info(name).n6_compatible


# Built-in frontend registrations
register_frontend(
    FrontendInfo(
        name="librosa",
        mode="precomputed",
        precomputed=True,
        n6_compatible=True,
        description="Host-side mel spectrogram (librosa). Pass-through in model.",
    )
)
register_frontend(
    FrontendInfo(
        name="hybrid",
        mode="hybrid",
        precomputed=False,
        n6_compatible=True,
        description="Offline STFT + in-model 1x1 Conv2D mel mixer.",
    )
)
register_frontend(
    FrontendInfo(
        name="raw",
        mode="raw",
        precomputed=False,
        n6_compatible=True,
        description="Raw waveform -> learned Conv2D filterbank (requires T < 65536).",
    )
)
register_frontend(
    FrontendInfo(
        name="mfcc",
        mode="precomputed",
        precomputed=True,
        n6_compatible=True,
        description="Host-side MFCC (mel -> DCT -> truncate). Pass-through in model.",
    )
)
register_frontend(
    FrontendInfo(
        name="log_mel",
        mode="precomputed",
        precomputed=True,
        n6_compatible=True,
        description="Host-side log-mel spectrogram (log1p). Quantization-friendly.",
    )
)
