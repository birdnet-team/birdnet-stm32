# Building & Flashing

## Prerequisites

| Tool | Version | Required For |
|---|---|---|
| [ST Edge AI Core](https://www.st.com/en/development-tools/stedgeai-core.html) | 4.1.0 (4.0.1, X-CUBE-AI 10.2 also work) | `stedgeai` CLI, `n6_loader.py`, LL_ATON runtime, NPU_Validation project |
| [ARM GNU Toolchain](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads) | 14.3+ | `arm-none-eabi-gcc` cross-compiler |
| [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html) | 1.19+ | Provides `STM32_Programmer_CLI` and GDB server |
| Python | 3.12+ | Host-side orchestration (`board_test.py`) |
| pyserial | 3.5+ | UART capture during board-test |

The board must be connected via USB (ST-LINK V3). The ST-LINK provides both the
SWD debug interface and the VCP UART.

## The Easy Way: `board-test`

The `board-test` CLI command handles everything — model compilation, firmware
patching, building, flashing, UART capture, and cleanup:

```bash
# Make sure config.json points to your ST Edge AI Core and CubeIDE installs
python -m birdnet_stm32 board-test --config config.json --timeout 300
```

This is the recommended workflow. The sections below explain what happens under
the hood so you can adapt it.

## How the Build Works

The firmware does **not** have a standalone build system. It is designed to be
**overlaid** onto ST's NPU_Validation project, which provides the complete HAL,
BSP, LL_ATON runtime, linker scripts, startup code, and Makefile.

### Why Overlay Instead of Standalone?

The NPU_Validation project contains ~50 source files, pre-built LL_ATON
libraries, and complex linker scripts specific to the N6's multi-bank SRAM
layout. Reproducing this as a standalone project would be fragile and hard to
maintain across ST toolchain versions. The overlay approach:

- Uses ST's tested build infrastructure as-is.
- Stays compatible across toolchain updates (X-CUBE-AI 10.2, ST Edge AI Core 4.0 and 4.1).
- Adds only the files we need (6 `.c` + 12 `.h` + FatFs + HAL_SD).
- Cleans up after itself (backup/restore of every patched file).

### Build Pipeline (Step by Step)

The Python orchestrator (`birdnet_stm32/deploy/board_test.py`) automates these
steps:

#### 1. Generate NPU artifacts

```bash
stedgeai generate \
  --model checkpoints/best_model_quantized.tflite \
  --target stm32n6 \
  --st-neural-art \
  --enable-epoch-controller \
  --compression none \
  --output st_ai_output \
  --workspace st_ai_ws
```

`--enable-epoch-controller` is added from ST Edge AI Core 4.0 (older cores
lack it): the NPU's epoch controller then runs the schedule without the
Cortex-M55 stepping each epoch.

This compiles the TFLite model into:

- `network.c` / `network.h` — model graph description and weight pointers
- `network_ecl.h` — epoch-compiled layer metadata
- Weight binary blobs (flashed to external NOR)

#### 2. Patch NPU_Validation

The orchestrator copies our firmware sources into the NPU_Validation project
tree:

| Source | Destination in NPU_Validation |
|---|---|
| `firmware/Src/*.c` | `Core/Src/` |
| `firmware/Inc/*.h` | `Core/Inc/` |
| `firmware/Drivers/HAL_SD/*.c` | `Drivers/STM32N6xx_HAL_Driver/Src/` |
| `firmware/Drivers/HAL_SD/*.h` | `Drivers/STM32N6xx_HAL_Driver/Inc/` (only if the HAL lacks them) |
| `firmware/Drivers/stm32n6570_discovery_sd.*` | `Drivers/BSP/STM32N6570-DK/` |
| `firmware/Drivers/FatFs/` | `FatFs/` (new directory) |

`Drivers/` is the project's own up to X-CUBE-AI 10.2; from ST Edge AI Core 4.0
the projects share `Projects/STM32N6570-DK/Applications/Drivers`. The 4.x HAL
ships its own SD headers (with renamed delay-block registers), so they are kept.

It also:

- **Auto-generates `app_config.h`** from `model_config.json` via
  `gen_app_config.py` — sample rate, chunk duration (supports fractional
  values like 2.9 s), chunk samples, FFT size, hop length, spectrogram width,
  number of classes, frontend mode, and all NPU_Validation board support
  defines (`USE_OVERDRIVE`, `USE_UART_BAUDRATE`, etc.).
- **Auto-generates `app_labels.h`** from the labels file (class name string
  array).
- **Patches the Makefile** to compile the new `.c` files and add include paths,
  and passes the generated network's name (`-DAPP_NN_NAME`, see below).
- **Enables `HAL_SD_MODULE_ENABLED`** in `stm32n6xx_hal_conf.h`.

!!! note "Network name"
    `stedgeai` names the generated C symbols after the network:
    `LL_ATON_Input_Buffers_Info_Default()` and `NN_Instance_Default` up to
    X-CUBE-AI 10.2, `..._network` from ST Edge AI Core 4.0. The firmware calls
    them through `NN_FN()` / `NN_INSTANCE` in `main.c`, built from
    `APP_NN_NAME`; `board-test` and the standalone Makefile read the name from
    the generated `network.c` and pass `-DAPP_NN_NAME`. A link error such as
    *undefined reference to `LL_ATON_EC_Network_Init_Default`* means the name
    was not passed.

!!! warning "Every patched file is backed up"
    Before modifying any NPU_Validation file, the orchestrator creates a `.bak`
    copy. After the run completes (or fails), all backups are restored. You
    should never see leftover modifications in the NPU_Validation directory.

#### 3. Build with ARM GCC

```bash
make -C "$NPU_VALIDATION" -j$(nproc)
```

The Makefile compiles all C sources, links the LL_ATON runtime library, and
produces a `.elf` binary.

#### 4. Flash via `n6_loader.py`

The `n6_loader.py` script (part of ST Edge AI Core, in `Utilities/scripts/N6_scripts`
from 4.1 and `scripts/N6_scripts` before) uses GDB to:

1. Load the `.elf` binary into internal SRAM.
2. Set a hardware breakpoint at `aiValidationInit()`.
3. Start execution — the firmware initializes clocks and memory-maps external
   NOR flash.
4. When the breakpoint hits, GDB writes the NPU weight blobs into external NOR
   via the `STM32_Programmer_CLI`.
5. GDB removes the breakpoint and continues — the firmware proceeds with SD
   card processing.

!!! danger "Do not remove `aiValidationInit()`"
    This function exists solely as a GDB breakpoint target. Its body is just
    `__asm volatile("nop")`. If you rename or remove it, the flash process
    fails silently and the board hangs.

#### 5. UART Capture

The host opens the serial port (typically `/dev/ttyACM0`) at 921,600 baud and
captures all firmware output until the `=== DONE ===` marker or timeout.
Results are parsed and displayed. See [UART Protocol](protocol.md).

#### 6. Cleanup

All `.bak` files are restored, returning the NPU_Validation project to its
original state.

## Init Sequence

The firmware's `main()` function initializes the board in a specific order that
**must not be changed** (it mirrors ST's NPU_Validation reference):

| Step | Function | Purpose |
|---|---|---|
| 1 | `set_vector_table_addr()` | Point VTOR to correct address (n6_loader loads code at non-default offset) |
| 2 | `HAL_Init()` | HAL tick, NVIC priority grouping |
| 3 | `SystemClock_Config_ResetClocks()` | Reset all clock domains to known state |
| 4 | `system_init_post()` | Post-reset cleanup (clear pending interrupts) |
| 5 | `SCB_EnableICache()` / `SCB_EnableDCache()` | Enable CPU caches |
| 6 | Clock configuration | Default: `SystemClock_Config_HSI_400()`, CPU/NIC/NOC/NPU at 400 MHz on nominal core voltage; optional overdrive first raises VDD core, then runs the CPU at 800 MHz and the NPU at 1 GHz |
| 7 | `fuse_vddio()` | Configure IO voltage rails for XSPI interfaces |
| 8 | `UART_Config()` | USART1 at 921,600 baud (ST-LINK VCP) |
| 9 | `BSP_XSPI_RAM_Init()` + `BSP_XSPI_NOR_Init()` | Memory-map external HyperRAM and NOR flash |
| 10 | `NPU_Config()` + `RISAF_Config()` | NPU clocks, register interface, security attributes |
| 11 | `aiValidationInit()` | GDB breakpoint stub (NOR flash happens here) |

!!! warning "XSPI before NPU"
    External memory init (step 9) **must** happen before NPU init (step 10)
    because NPU weights live in NOR flash. If NOR isn't memory-mapped when the
    NPU reads weights, you get a bus fault. See
    [Troubleshooting](troubleshooting.md#xspi-initialization-order).

## Manual Build (Advanced)

If you want to build without the Python orchestrator:

```bash
# 1. Set paths
export XCUBEAI="/path/to/STEdgeAI/4.1"
export NPU_VAL="$XCUBEAI/Projects/STM32N6570-DK/Applications/NPU_Validation"

# 2. Generate network files
$XCUBEAI/Utilities/linux/stedgeai generate \
  --model checkpoints/best_model_quantized.tflite \
  --target stm32n6 --st-neural-art --enable-epoch-controller \
  --output "$NPU_VAL/Model"

# 3. Copy firmware sources
cp firmware/Src/*.c "$NPU_VAL/Core/Src/"
cp firmware/Inc/*.h "$NPU_VAL/Core/Inc/"
# ... (copy drivers, FatFs, patch Makefile — see board_test.py for details)

# 4. Build
make -C "$NPU_VAL" -j$(nproc)

# 5. Build and flash (n6_loader.py builds the project named in config_n6l.json)
python "$XCUBEAI/Utilities/scripts/N6_scripts/n6_loader.py" \
  --n6-loader-config config_n6l.json
```

This is error-prone — the automated `board-test` command handles dozens of
edge cases (Makefile patching, HAL config flags, backup/restore). Use it unless
you have a specific reason not to.

## Config Files

### `config.json` (Machine-Local)

Maps tool paths on your machine. **Do not commit this file** — it's in
`.gitignore`.

```json
{
  "compiler_type": "gcc",
  "cubeide_path": "/path/to/stm32cubeide",
  "x_cube_ai_path": "/path/to/STEdgeAI/4.1",
  "model_path": "checkpoints/best_model_quantized.tflite",
  "output_dir": "validation/st_ai_output",
  "workspace_dir": "validation/st_ai_ws",
  "n6_loader_config": "config_n6l.json"
}
```

`x_cube_ai_path` is the version folder of the ST Edge AI Core install (the one
holding `Utilities/` and `Projects/`), or the root of an X-CUBE-AI 10.2 tree.

### `config_n6l.json` (Machine-Local)

Path mappings for `n6_loader.py`. See `config_n6l.example.json` for the
template.

### `setup.sh`

One-liner that creates `.venv`, installs Python deps, auto-detects tool paths,
and generates both config files:

```bash
bash setup.sh
```
