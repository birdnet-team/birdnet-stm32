#!/usr/bin/env bash
# Download pre-trained BirdNET-STM32 checkpoints.
#
# Usage:
#   ./scripts/download_checkpoints.sh [DEST_DIR]
#
# Default destination: checkpoints/

set -euo pipefail

DEST="${1:-checkpoints}"
REPO="your-org/birdnet-stm32"
TAG="latest"

mkdir -p "$DEST"

echo "Downloading checkpoints to $DEST/ ..."

# TODO: Replace with actual release URL once published.
# Example using GitHub Releases:
#   gh release download "$TAG" --repo "$REPO" --pattern '*.keras' --dir "$DEST"
#   gh release download "$TAG" --repo "$REPO" --pattern '*.tflite' --dir "$DEST"
#   gh release download "$TAG" --repo "$REPO" --pattern '*_labels.txt' --dir "$DEST"
#   gh release download "$TAG" --repo "$REPO" --pattern '*_model_config.json' --dir "$DEST"

echo "NOTE: This is a placeholder script."
echo "      Update REPO and TAG with your GitHub release coordinates,"
echo "      then uncomment the gh commands above."
