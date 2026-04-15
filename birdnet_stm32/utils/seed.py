"""Global random seed management for reproducibility."""

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Set random seeds for numpy, random, and TensorFlow for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass
