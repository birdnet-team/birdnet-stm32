"""Host-side orchestrator for on-board SD card inference tests.

Workflow:
1. Generate NPU artifacts from the quantized TFLite model (stedgeai generate).
2. Patch the firmware to include the generated network and labels.
3. Build the firmware (CMake / GCC for Cortex-M55).
4. Flash the firmware to the STM32N6570-DK board.
5. Monitor serial output for results and completion.
6. (Optional) Parse results and compare against expected labels.
"""

import os
import re
import sys
import time
from dataclasses import dataclass, field

import serial

from birdnet_stm32.deploy.config import DeployConfig


@dataclass
class BoardTestConfig:
    """Configuration for on-board SD card inference tests.

    Attributes:
        deploy_cfg: Base deployment configuration.
        firmware_dir: Path to the firmware source directory.
        labels_path: Path to the _labels.txt file.
        serial_port: Serial port for UART output.
        serial_baud: Baud rate for UART.
        timeout: Maximum seconds to wait for board completion.
    """

    deploy_cfg: DeployConfig = field(default_factory=DeployConfig)
    firmware_dir: str = "firmware"
    labels_path: str = ""
    serial_port: str = "/dev/ttyACM0"
    serial_baud: int = 115200
    timeout: int = 300


def generate_labels_header(labels_path: str, output_path: str, num_classes: int):
    """Generate a C header with class labels from a labels.txt file.

    Args:
        labels_path: Path to _labels.txt (one label per line).
        output_path: Output .h file path.
        num_classes: Expected number of classes.
    """
    with open(labels_path) as f:
        labels = [line.strip() for line in f if line.strip()]

    if len(labels) != num_classes:
        print(f"[WARN] labels.txt has {len(labels)} entries, expected {num_classes}")

    with open(output_path, "w") as f:
        f.write("/* Auto-generated — do not edit. */\n")
        f.write("#ifndef APP_LABELS_H\n#define APP_LABELS_H\n\n")
        f.write(f"#define APP_NUM_CLASSES_ACTUAL {len(labels)}\n\n")
        f.write("static const char * const APP_LABELS[] = {\n")
        for label in labels:
            escaped = label.replace('"', '\\"')
            f.write(f'    "{escaped}",\n')
        f.write("};\n\n#endif /* APP_LABELS_H */\n")
    print(f"[OK] Generated labels header: {output_path} ({len(labels)} classes)")


def monitor_serial(port: str, baud: int, timeout: int) -> str:
    """Monitor serial output from the board until completion or timeout.

    Looks for the '=== DONE ===' marker in the output.

    Args:
        port: Serial port path.
        baud: Baud rate.
        timeout: Maximum seconds to wait.

    Returns:
        Complete serial output as a string.
    """
    output_lines: list[str] = []
    print(f"[serial] Monitoring {port} @ {baud} baud (timeout: {timeout}s)")

    try:
        with serial.Serial(port, baud, timeout=1) as ser:
            start = time.monotonic()
            while (time.monotonic() - start) < timeout:
                line = ser.readline().decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue
                print(f"  > {line}")
                output_lines.append(line)
                if "=== DONE ===" in line:
                    # Read a few more lines for the summary
                    for _ in range(5):
                        extra = ser.readline().decode("utf-8", errors="replace").rstrip()
                        if extra:
                            print(f"  > {extra}")
                            output_lines.append(extra)
                    break
            else:
                print(f"[WARN] Timeout after {timeout}s — board may still be processing")
    except serial.SerialException as e:
        print(f"[ERROR] Serial: {e}")

    return "\n".join(output_lines)


def parse_serial_results(output: str) -> list[dict]:
    """Parse detection results from serial output.

    Expects lines like:
        [1/5] filename.wav:
            [1] Species Name: 95.3%

    Args:
        output: Raw serial output string.

    Returns:
        List of dicts with 'filename' and 'detections' keys.
    """
    results: list[dict] = []
    current_file = None

    for line in output.split("\n"):
        # Match file header: [N/M] filename.wav
        file_match = re.match(r"\s*\[\d+/\d+\]\s+(.+\.wav)", line, re.IGNORECASE)
        if file_match:
            current_file = {"filename": file_match.group(1).strip(), "detections": []}
            results.append(current_file)
            continue

        # Match detection: [K] Label: XX.X%
        det_match = re.match(r"\s*\[\d+\]\s+(.+):\s+(\d+\.\d+)%", line)
        if det_match and current_file is not None:
            current_file["detections"].append(
                {"label": det_match.group(1).strip(), "score_pct": float(det_match.group(2))}
            )

    return results


def run_board_test(cfg: BoardTestConfig) -> str:
    """Execute the full on-board inference test.

    Steps:
    1. Run stedgeai generate (compile model for NPU).
    2. Copy labels.txt to SD card root (if accessible) or firmware build dir.
    3. Build and flash firmware.
    4. Monitor serial output until completion.

    Args:
        cfg: Board test configuration.

    Returns:
        Raw serial output from the board.
    """
    deploy = cfg.deploy_cfg

    # Step 1: Validate prerequisites
    if not os.path.isfile(deploy.model_path):
        print(f"[ERROR] Model not found: {deploy.model_path}")
        sys.exit(1)
    if cfg.labels_path and not os.path.isfile(cfg.labels_path):
        print(f"[ERROR] Labels file not found: {cfg.labels_path}")
        sys.exit(1)

    print("\n=== BirdNET-STM32 Board Test ===\n")
    print(f"Model:   {deploy.model_path}")
    print(f"Labels:  {cfg.labels_path or '(default)'}")
    print(f"Serial:  {cfg.serial_port} @ {cfg.serial_baud}")
    print(f"Timeout: {cfg.timeout}s\n")

    # Step 2: Generate labels header (for firmware that embeds labels)
    if cfg.labels_path:
        labels_h = os.path.join(cfg.firmware_dir, "Inc", "app_labels.h")
        with open(cfg.labels_path) as f:
            num_labels = sum(1 for line in f if line.strip())
        generate_labels_header(cfg.labels_path, labels_h, num_labels)

    # Step 3: Generate NPU artifacts
    print("\n--- Step 1: Generate NPU artifacts ---")
    from birdnet_stm32.deploy.stedgeai import generate

    generate(deploy)

    # Step 4: Build and flash firmware
    # This step depends on the user's toolchain setup.
    # The n6_loader.py approach is used for the NPU_Validation project;
    # for the custom firmware, the user needs to integrate our source files
    # into their build system.
    print("\n--- Step 2: Build and flash firmware ---")
    print("[INFO] Automatic build/flash for custom firmware is not yet implemented.")
    print("[INFO] Please build and flash the firmware manually, then press Enter")
    print("[INFO] to start serial monitoring.")
    print("[INFO]")
    print("[INFO] Quick guide:")
    print("[INFO]   1. Copy firmware/Src/*.c and firmware/Inc/*.h into your project")
    print(f"[INFO]   2. Copy {deploy.output_dir}/network.c into X-CUBE-AI/App/")
    print("[INFO]   3. Build and flash via STM32CubeIDE or make")
    print("[INFO]   4. Ensure the SD card with audio/ folder is inserted")
    input("\nPress Enter to start monitoring serial output...")

    # Step 5: Monitor serial output
    print("\n--- Step 3: Monitor board output ---")
    output = monitor_serial(cfg.serial_port, cfg.serial_baud, cfg.timeout)

    # Step 6: Parse and report
    results = parse_serial_results(output)
    if results:
        print(f"\n--- Summary: {len(results)} files processed ---")
        for r in results:
            dets = ", ".join(f"{d['label']} ({d['score_pct']}%)" for d in r["detections"])
            print(f"  {r['filename']}: {dets or 'no detections'}")
    else:
        print("\n[WARN] No results parsed from serial output")

    return output
