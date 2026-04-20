#!/usr/bin/env bash
# Download the BirdNET-STM32 training/test dataset.
#
# Usage:
#   ./scripts/download_data.sh [DEST_DIR]
#
# Default destination: data/

set -euo pipefail

DEST="${1:-data}"

mkdir -p "$DEST/train" "$DEST/test"

echo "Downloading dataset to $DEST/ ..."

# TODO: Replace with actual data source URL.
# Example using a public archive:
#   curl -L "https://example.com/birdnet-stm32-data.tar.gz" | tar xz -C "$DEST"
#
# Dataset layout required:
#   data/train/<species_name>/*.wav
#   data/test/<species_name>/*.wav
#
# Special folder names (noise, silence, background, other) get all-zero labels.

echo "NOTE: This is a placeholder script."
echo "      Replace the URL above with your actual data archive location."
