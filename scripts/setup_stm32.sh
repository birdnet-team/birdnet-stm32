#!/usr/bin/env bash
# Set up the STM32 development toolchain for BirdNET-STM32 deployment.
#
# This script checks for (and optionally installs) the required tools:
#   - arm-none-eabi-gcc (ARM toolchain)
#   - STM32CubeIDE (manual install required)
#   - X-CUBE-AI / stedgeai (manual install required)
#
# Usage:
#   ./scripts/setup_stm32.sh

set -euo pipefail

RED='\033[1;31m'
GREEN='\033[32m'
YELLOW='\033[33m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC}   $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[MISS]${NC} $1"; }

echo "=== BirdNET-STM32 Toolchain Check ==="
echo

# --- ARM toolchain ---
if command -v arm-none-eabi-gcc &>/dev/null; then
    VER=$(arm-none-eabi-gcc --version | head -1)
    ok "arm-none-eabi-gcc: $VER"
else
    fail "arm-none-eabi-gcc not found"
    echo "  Install via package manager:"
    echo "    Ubuntu/Debian: sudo apt install gcc-arm-none-eabi"
    echo "    macOS:         brew install arm-none-eabi-gcc"
    echo "    Or download:   https://developer.arm.com/downloads/-/gnu-rm"
fi

echo

# --- arm-none-eabi-objcopy ---
if command -v arm-none-eabi-objcopy &>/dev/null; then
    ok "arm-none-eabi-objcopy: $(which arm-none-eabi-objcopy)"
else
    fail "arm-none-eabi-objcopy not found (installed with gcc-arm-none-eabi)"
fi

echo

# --- STM32CubeIDE ---
if [[ -n "${CUBEIDE_PATH:-}" ]] && [[ -d "$CUBEIDE_PATH" ]]; then
    ok "STM32CubeIDE: $CUBEIDE_PATH"
elif command -v stm32cubeide &>/dev/null; then
    ok "STM32CubeIDE: $(which stm32cubeide)"
else
    warn "STM32CubeIDE not detected (set CUBEIDE_PATH or install from st.com)"
    echo "  Download: https://www.st.com/en/development-tools/stm32cubeide.html"
fi

echo

# --- X-CUBE-AI / stedgeai ---
if [[ -n "${X_CUBE_AI_PATH:-}" ]]; then
    STEDGEAI="$X_CUBE_AI_PATH/Utilities/linux/stedgeai"
    if [[ -x "$STEDGEAI" ]]; then
        ok "stedgeai: $STEDGEAI"
    else
        fail "X_CUBE_AI_PATH set but stedgeai not found at $STEDGEAI"
    fi
elif command -v stedgeai &>/dev/null; then
    ok "stedgeai: $(which stedgeai)"
else
    fail "stedgeai (X-CUBE-AI) not found"
    echo "  Set X_CUBE_AI_PATH to the X-CUBE-AI installation directory,"
    echo "  or add stedgeai to your PATH."
    echo "  Download: https://www.st.com/en/embedded-software/x-cube-ai.html"
fi

echo

# --- Python environment ---
if python3 -c "import birdnet_stm32" 2>/dev/null; then
    ok "birdnet_stm32 Python package importable"
else
    warn "birdnet_stm32 not importable — run: pip install -e \".[dev]\""
fi

echo

# --- Config files ---
if [[ -f "config.toml" ]] || [[ -f "config.json" ]]; then
    ok "Deploy config found"
else
    warn "No config.toml or config.json found — copy from config.toml.example"
fi

echo "=== Done ==="
