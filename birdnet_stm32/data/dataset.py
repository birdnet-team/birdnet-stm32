"""Dataset loading, file path discovery, and class balancing utilities.

Functions for walking a class-structured audio directory, collecting file paths,
and performing minority-class upsampling.
"""

import os

import numpy as np
import tensorflow as tf

# Supported audio filename extensions (lowercase)
SUPPORTED_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def get_classes_with_most_samples(
    directory: str,
    n_classes: int = 25,
    include_noise: bool = False,
    exts: tuple = SUPPORTED_AUDIO_EXTS,
) -> list[str]:
    """Collect the most frequent class labels from a dataset root.

    Args:
        directory: Root dataset directory (class-subfolders).
        n_classes: Number of top classes to return (upper bound).
        include_noise: If False, exclude noise-like labels.
        exts: Accepted audio file extensions (case-insensitive).

    Returns:
        Up to n_classes class names, sorted by descending sample count.
    """
    classes: dict[str, int] = {}
    noise_classes = {"noise", "silence", "background", "other"}

    for root, _dirs, files in os.walk(directory):
        for fname in files:
            if not fname.lower().endswith(exts):
                continue
            class_name = os.path.basename(root)
            if not include_noise and class_name.lower() in noise_classes:
                continue
            classes[class_name] = classes.get(class_name, 0) + 1

    sorted_classes = sorted(classes.items(), key=lambda x: x[1], reverse=True)
    return [cls for cls, _ in sorted_classes[:n_classes]]


def load_file_paths_from_directory(
    directory: str,
    classes: list[str] | None = None,
    max_samples: int | None = None,
    exts: tuple = SUPPORTED_AUDIO_EXTS,
) -> tuple[list[str], list[str]]:
    """Recursively gather audio files from a class-structured directory.

    Expected layout::

        root/
          class_a/*.(wav|mp3|flac|ogg|m4a)
          class_b/*.(wav|mp3|flac|ogg|m4a)

    Args:
        directory: Dataset root directory.
        classes: If given, restrict to these class names only.
        max_samples: Cap the number of files per class (uniform random).
        exts: Accepted audio file extensions (case-insensitive).

    Returns:
        Tuple of (shuffled file paths, sorted class names). Noise-like names
        are excluded from the class list but their files are still included.
    """
    per_class: dict[str, list[str]] = {}

    for root, _, files in tf.io.gfile.walk(directory):
        for fname in files:
            if not fname.lower().endswith(exts):
                continue
            full_path = tf.io.gfile.join(root, fname)
            parent_class = os.path.basename(os.path.dirname(full_path))

            if classes is not None and parent_class not in classes:
                continue

            per_class.setdefault(parent_class, []).append(full_path)

    all_paths: list[str] = []
    for _cls, paths in per_class.items():
        if max_samples is not None and max_samples > 0 and len(paths) > max_samples:
            idx = np.random.permutation(len(paths))[:max_samples]
            paths = [paths[i] for i in idx]
        all_paths.extend(paths)

    np.random.shuffle(all_paths)

    noise_classes = {"noise", "silence", "background", "other"}
    classes_out = sorted(c for c in per_class if c.lower() not in noise_classes)

    return all_paths, classes_out


def upsample_minority_classes(
    file_paths: list[str],
    classes: list[str],
    ratio: float = 0.25,
) -> list[str]:
    """Upsample minority classes to approach the largest class size via repetition.

    Args:
        file_paths: List of audio file paths.
        classes: Ordered class names.
        ratio: Target fraction of the largest class size (0 < ratio <= 1).

    Returns:
        Augmented list of file paths with upsampled minority classes.
    """
    assert 0 < ratio <= 1, "Ratio must be in (0, 1]."
    class_to_paths: dict[str, list[str]] = {cls: [] for cls in classes}

    for path in file_paths:
        class_name = os.path.basename(os.path.dirname(path))
        if class_name in class_to_paths:
            class_to_paths[class_name].append(path)

    max_size = max(len(paths) for paths in class_to_paths.values())
    target_size = int(max_size * ratio)

    augmented_paths: list[str] = []
    for _cls, paths in class_to_paths.items():
        current_size = len(paths)
        if current_size < target_size:
            num_to_add = target_size - current_size
            additional = np.random.choice(paths, size=num_to_add, replace=True).tolist()
            paths.extend(additional)
        augmented_paths.extend(paths)

    np.random.shuffle(augmented_paths)
    return augmented_paths
