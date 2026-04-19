"""Backward-compatible wrapper — delegates to the birdnet_stm32 package.

Usage: python train.py --data_path_train data/train [options]
Prefer: python -m birdnet_stm32 train --data_path_train data/train [options]
"""

import warnings

warnings.warn(
    "Running train.py directly is deprecated. Use 'python -m birdnet_stm32 train' instead.",
    DeprecationWarning,
    stacklevel=1,
)

from birdnet_stm32.cli.train import main  # noqa: E402

if __name__ == "__main__":
    main()
