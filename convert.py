"""Backward-compatible wrapper — delegates to the birdnet_stm32 package.

Usage: python convert.py --checkpoint_path MODEL.keras --model_config CONFIG.json [options]
Prefer: python -m birdnet_stm32 convert --checkpoint_path MODEL.keras --model_config CONFIG.json [options]
"""

import warnings

warnings.warn(
    "Running convert.py directly is deprecated. Use 'python -m birdnet_stm32 convert' instead.",
    DeprecationWarning,
    stacklevel=1,
)

from birdnet_stm32.cli.convert import main  # noqa: E402

if __name__ == "__main__":
    main()
