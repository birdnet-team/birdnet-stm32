#!/usr/bin/env bash
# BirdNET-STM32 — one-liner setup script
#
# Usage:
#   bash setup.sh                      # interactive — prompts for paths
#   bash setup.sh /path/to/X-CUBE-AI   # non-interactive
#
# What it does:
#   1. Creates a Python 3.12 virtual environment (.venv)
#   2. Installs the package in editable mode with all extras
#   3. Generates config.json and config_n6l.json from your paths
#
# Prerequisites:
#   - Python 3.12+          (python3.12 -m venv)
#   - ARM GCC 13+           (arm-none-eabi-gcc — for board deployment only)
#   - X-CUBE-AI 10.2.0      (https://www.st.com/en/embedded-software/x-cube-ai.html)
#   - STM32CubeIDE           (optional — only needed if compiler_type=cubeide)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Colors ---------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[setup]${NC} $*"; }
ok()    { echo -e "${GREEN}[  ok ]${NC} $*"; }
err()   { echo -e "${RED}[error]${NC} $*" >&2; }

# ---- 1. Python venv -------------------------------------------------------
info "Creating Python virtual environment..."
if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
ok "Virtual environment activated: $(python --version)"

# ---- 2. Install package ---------------------------------------------------
info "Installing birdnet-stm32 (editable, all extras)..."
pip install --upgrade pip setuptools wheel -q
pip install -e ".[dev,deploy]" -q
ok "Package installed"

# ---- 3. Resolve X-CUBE-AI path -------------------------------------------
XCUBEAI="${1:-}"
if [[ -z "$XCUBEAI" ]]; then
    # Try common locations
    for candidate in \
        "$HOME/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.2.0" \
        "$HOME/Code/X-CUBE-AI.10.2.0" \
        "/opt/X-CUBE-AI.10.2.0"; do
        if [[ -d "$candidate" ]]; then
            XCUBEAI="$candidate"
            break
        fi
    done
fi

if [[ -z "$XCUBEAI" ]]; then
    echo ""
    info "X-CUBE-AI path not found automatically."
    read -rp "  Enter path to X-CUBE-AI 10.2.0 root directory: " XCUBEAI
fi

if [[ ! -d "$XCUBEAI" ]]; then
    err "X-CUBE-AI directory not found: $XCUBEAI"
    err "Download from https://www.st.com/en/embedded-software/x-cube-ai.html"
    exit 1
fi
ok "X-CUBE-AI found: $XCUBEAI"

# ---- 4. Resolve STM32CubeIDE path (optional) -----------------------------
CUBEIDE=""
for candidate in \
    "$HOME/stm32cubeide" \
    "$HOME/STM32CubeIDE" \
    "/opt/stm32cubeide" \
    "/opt/STM32CubeIDE"; do
    if [[ -d "$candidate" ]]; then
        CUBEIDE="$candidate"
        break
    fi
done

# ---- 5. Find objcopy -----------------------------------------------------
OBJCOPY="$(command -v arm-none-eabi-objcopy 2>/dev/null || true)"
if [[ -z "$OBJCOPY" ]]; then
    # Search in common STM32CubeIDE plugin locations
    OBJCOPY="$(find "${CUBEIDE:-/nonexistent}" -name arm-none-eabi-objcopy -type f 2>/dev/null | head -1 || true)"
fi
if [[ -z "$OBJCOPY" ]]; then
    OBJCOPY="/usr/bin/arm-none-eabi-objcopy"
    info "arm-none-eabi-objcopy not found; defaulting to $OBJCOPY"
fi

# ---- 6. Write config.json ------------------------------------------------
NPU_VAL="$XCUBEAI/Projects/STM32N6570-DK/Applications/NPU_Validation"

if [[ ! -f config.json ]]; then
    cat > config.json <<EOF
{
  "compiler_type": "gcc",
  "cubeide_path": "$CUBEIDE",
  "x_cube_ai_path": "$XCUBEAI",
  "model_path": "checkpoints/best_model_quantized.tflite",
  "output_dir": "validation/st_ai_output",
  "workspace_dir": "validation/st_ai_ws",
  "n6_loader_config": "config_n6l.json"
}
EOF
    ok "Created config.json"
else
    info "config.json already exists — skipping"
fi

# ---- 7. Write config_n6l.json --------------------------------------------
if [[ ! -f config_n6l.json ]]; then
    cat > config_n6l.json <<EOF
{
  "network.c": "$(pwd)/validation/st_ai_output/network.c",
  "project_path": "$NPU_VAL",
  "project_build_conf": "N6-DK",
  "skip_external_flash_programming": false,
  "skip_ram_data_programming": false,
  "objcopy_binary_path": "$OBJCOPY"
}
EOF
    ok "Created config_n6l.json"
else
    info "config_n6l.json already exists — skipping"
fi

# ---- Done -----------------------------------------------------------------
echo ""
ok "Setup complete!"
echo ""
info "Quick start:"
echo "  source .venv/bin/activate"
echo "  python train.py --data_path_train data/train --audio_frontend hybrid --mag_scale pwl"
echo "  python convert.py --checkpoint_path checkpoints/best_model.keras \\"
echo "    --model_config checkpoints/best_model_model_config.json --data_path_train data/train"
echo "  python test.py --model_path checkpoints/best_model_quantized.tflite \\"
echo "    --model_config checkpoints/best_model_model_config.json --data_path_test data/test"
echo "  python -m birdnet_stm32 board-test"
echo ""
