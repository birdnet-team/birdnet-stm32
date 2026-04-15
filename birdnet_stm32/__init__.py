"""BirdNet-STM32: bird sound classification for edge deployment on STM32N6570-DK."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("birdnet-stm32")
except PackageNotFoundError:
    __version__ = "0.2.0"  # fallback when not installed
