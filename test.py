"""Backward-compatible wrapper — delegates to the birdnet_stm32 package.

Usage: python test.py --model_path MODEL --data_path_test data/test [options]
Prefer: python -m birdnet_stm32 evaluate --model_path MODEL --data_path_test data/test [options]
"""

import warnings

warnings.warn(
    "Running test.py directly is deprecated. Use 'python -m birdnet_stm32 evaluate' instead.",
    DeprecationWarning,
    stacklevel=1,
)

from birdnet_stm32.cli.evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
