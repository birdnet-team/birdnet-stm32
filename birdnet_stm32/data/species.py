"""Species list utilities for dataset construction.

Provides functions to load, save, combine, and deduplicate species lists
used for building class-structured training and test datasets.
"""

from __future__ import annotations

import os


def load_species_list(path: str) -> list[str]:
    """Load a species list from a text file (one species per line).

    Args:
        path: Path to the species list file.

    Returns:
        List of species names (non-empty, stripped).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the resulting list is empty.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Species list not found: {path}")
    with open(path, encoding="utf-8") as f:
        species = [line.strip() for line in f if line.strip()]
    if not species:
        raise ValueError(f"Species list is empty: {path}")
    return species


def save_species_list(species: list[str], path: str) -> None:
    """Save a species list to a text file (one species per line).

    Args:
        species: List of species names.
        path: Output file path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sp in species:
            f.write(f"{sp}\n")


def open_species_list(path: str) -> list[str]:
    """Load and deduplicate a species list, sorted alphabetically.

    Args:
        path: Path to the species list file.

    Returns:
        Sorted, deduplicated list of species names.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the resulting list is empty.
    """
    species = load_species_list(path)
    seen: set[str] = set()
    unique = [x for x in species if not (x in seen or seen.add(x))]
    unique.sort()
    if not unique:
        raise ValueError(f"Species list is empty after deduplication: {path}")
    return unique


def combine_species_lists(
    file_list: list[str],
    output_file: str,
    max_species: int | None = None,
) -> list[str]:
    """Combine multiple species list files using round-robin selection.

    Reads species from each file, draws them round-robin, deduplicates while
    preserving order, optionally limits to *max_species*, sorts alphabetically,
    and writes the result.

    Args:
        file_list: Paths to species list files.
        output_file: Path to write the combined list.
        max_species: Maximum number of species to keep (None = no limit).

    Returns:
        The combined, sorted species list.

    Raises:
        FileNotFoundError: If any input file does not exist.
    """
    per_file: dict[str, list[str]] = {}
    for fname in file_list:
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Species list not found: {fname}")
        with open(fname, encoding="utf-8") as f:
            entries: list[str] = []
            for line in f:
                sp = line.strip()
                if sp and sp not in entries:
                    entries.append(sp)
            per_file[fname] = entries

    # Round-robin draw
    combined: list[str] = []
    while True:
        added = False
        for fname in file_list:
            if per_file[fname]:
                combined.append(per_file[fname].pop(0))
                added = True
        if not added:
            break

    # Deduplicate preserving order
    seen: set[str] = set()
    combined = [x for x in combined if not (x in seen or seen.add(x))]

    if max_species is not None:
        combined = combined[:max_species]

    combined.sort()
    save_species_list(combined, output_file)
    return combined
