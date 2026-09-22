"""Board test: standalone firmware on STM32N6570-DK.

Full on-device pipeline — nothing is precomputed on the host:
1. Deploy model: stedgeai generate → patch NPU_Validation project → n6_loader
   build + flash.
2. Firmware on board: read WAV from SD card → frontend-specific preprocessing
   (raw normalization, STFT, or STFT + mel) → NPU inference → UART results.
3. This script captures UART output and parses per-file predictions.
4. Optionally, the same .tflite runs on the host over local copies of the SD
   card's WAV files, through the host's own evaluation preprocessing, and each
   file's board result is checked against it. That comparison is the point of
   the test: a model that ships has to compute on the device what it computes
   on the host.

Requires:
- USB-connected STM32N6570-DK with an SD card containing audio/ WAV files.
- pyserial (pip install pyserial).
"""

import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import serial

from birdnet_stm32.deploy.config import DeployConfig
from birdnet_stm32.deploy.stedgeai import generate

log = logging.getLogger(__name__)

# NPU_Validation's misc_toolbox uses this baud rate (from its app_config.h).
UART_BAUDRATE = 921600

# Marker the firmware prints when all files have been processed.
DONE_MARKER = "=== DONE ==="

# Sentinel appended to the NPU_Validation Makefile so we can detect + remove
# our patch on cleanup.
MAKEFILE_SENTINEL = "# --- BirdNET-STM32 board test additions ---"

# hal_conf.h line to uncomment for SD card support.
HAL_SD_COMMENTED = "//#define HAL_SD_MODULE_ENABLED"
HAL_SD_UNCOMMENTED = "#define HAL_SD_MODULE_ENABLED"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class BoardTestConfig:
    """Configuration for standalone on-board inference tests.

    Attributes:
        deploy_cfg: Base deployment configuration (paths to tools, model, etc.).
        model_config_path: Path to the _model_config.json file.
        labels_path: Path to the _labels.txt file.
        serial_port: Serial port for UART capture (e.g. /dev/ttyACM0).
        top_k: Number of top predictions to show per file.
        score_threshold: Minimum score to display.
        timeout: Maximum seconds to wait for firmware to finish.
        host_audio_dir: Local copy of the SD card's audio/ folder. When set, the
            host scores the same files and the board is checked against it.
        parity_tolerance: Margin within which a different top-1 is a tie, and
            within which a host score counts as borderline to the detection
            threshold. Larger score differences are flagged, not failed.
        detection_threshold: Score at which the device reports a detection.
    """

    deploy_cfg: DeployConfig = field(default_factory=DeployConfig)
    model_config_path: str = ""
    labels_path: str = ""
    serial_port: str = "/dev/ttyACM0"
    top_k: int = 5
    score_threshold: float = 0.01
    timeout: int = 300
    host_audio_dir: str = ""
    parity_tolerance: float = 0.05
    detection_threshold: float = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_model_config(config_path: str) -> dict:
    """Load model configuration from JSON.

    Args:
        config_path: Path to _model_config.json.

    Returns:
        Dict with model configuration.
    """
    with open(config_path) as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Model configuration must be a JSON object: {config_path}")
    return config


def load_labels(labels_path: str) -> list[str]:
    """Load class labels from a labels.txt file.

    Args:
        labels_path: Path to _labels.txt (one label per line).

    Returns:
        List of label strings.
    """
    with open(labels_path) as f:
        return [line.strip() for line in f if line.strip()]


def _firmware_dir() -> Path:
    """Return the path to the firmware/ directory shipped with this package."""
    return Path(__file__).resolve().parent.parent.parent / "firmware"


def _load_gen_app_config():
    """Import firmware/gen_app_config.py as a module."""
    script = _firmware_dir() / "gen_app_config.py"
    spec = importlib.util.spec_from_file_location("gen_app_config", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _generate_app_labels_h(labels: list[str]) -> str:
    """Generate the contents of ``app_labels.h`` from a list of class labels.

    Args:
        labels: Ordered class label strings (one per output neuron).

    Returns:
        C header source as a string.
    """
    return str(_load_gen_app_config().generate_app_labels_h(labels))


# ---------------------------------------------------------------------------
# Project patching — copies firmware sources into NPU_Validation tree
# ---------------------------------------------------------------------------


def _resolve_project_path(cfg: DeployConfig) -> Path:
    """Read the NPU_Validation project path from the n6_loader config."""
    with open(cfg.n6_loader_config) as f:
        n6cfg = json.load(f)
    return Path(n6cfg["project_path"])


def _backup(path: Path) -> Path:
    """Create a .bak copy of *path* if it exists.  Returns the backup path."""
    bak = path.with_suffix(path.suffix + ".bak")
    if path.exists():
        shutil.copy2(path, bak)
    return bak


def _restore(bak: Path) -> None:
    """Restore original from a .bak file and remove the backup."""
    orig = Path(str(bak).removesuffix(".bak"))
    if bak.exists():
        shutil.move(str(bak), str(orig))
    elif orig.exists():
        # New file we added — remove it.
        orig.unlink()


def patch_project(
    project: Path,
    model_cfg: dict,
    num_classes: int,
    labels: list[str],
) -> list[Path]:
    """Copy firmware sources into the NPU_Validation project and patch build files.

    Args:
        project: Path to the NPU_Validation project root.
        model_cfg: Model configuration dict from _model_config.json.
        num_classes: Number of output classes.
        labels: Ordered list of class label strings.  If empty, generates
                placeholder ``class_0`` … ``class_N`` labels.

    Returns a list of .bak paths that must be passed to ``restore_project``.
    """
    fw = _firmware_dir()
    core_src = project / "Core" / "Src"
    core_inc = project / "Core" / "Inc"
    hal_src = project / "Drivers" / "STM32N6xx_HAL_Driver" / "Src"
    bsp_dk = project / "Drivers" / "BSP" / "STM32N6570-DK"
    fatfs_dst = project / "FatFs"
    makefile = project / "armgcc" / "Makefile"
    hal_conf = core_inc / "stm32n6xx_hal_conf.h"
    app_conf = core_inc / "app_config.h"

    backups: list[Path] = []

    # --- 1. Copy firmware C sources into Core/Src --------------------------
    for name in ("main.c", "wav_reader.c", "audio_stft.c", "audio_mel.c", "sd_handler.c", "fft.c"):
        dst = core_src / name
        backups.append(_backup(dst))
        shutil.copy2(fw / "Src" / name, dst)

    # --- 2. Copy firmware headers into Core/Inc ----------------------------
    #  Skip app_config.h — we patch the NPU_Validation original in step 6.
    for name in ("wav_reader.h", "audio_stft.h", "audio_mel.h", "sd_handler.h", "fft.h"):
        dst = core_inc / name
        backups.append(_backup(dst))
        shutil.copy2(fw / "Inc" / name, dst)

    # FatFs config headers also go to Core/Inc (already in include path).
    for name in ("ffconf.h", "sd_diskio_config.h"):
        dst = core_inc / name
        backups.append(_backup(dst))
        shutil.copy2(fw / "Config" / name, dst)

    # --- 2b. Generate app_labels.h from the model's label list -------------
    if not labels:
        labels = [f"class_{i}" for i in range(num_classes)]
    app_labels_dst = core_inc / "app_labels.h"
    backups.append(_backup(app_labels_dst))
    app_labels_dst.write_text(_generate_app_labels_h(labels))
    log.info("Generated app_labels.h with %d labels", len(labels))

    # --- 3. Copy HAL SD driver sources and headers --------------------------
    hal_inc = project / "Drivers" / "STM32N6xx_HAL_Driver" / "Inc"
    for name in ("stm32n6xx_hal_sd.c", "stm32n6xx_ll_sdmmc.c"):
        dst = hal_src / name
        backups.append(_backup(dst))
        shutil.copy2(fw / "Drivers" / "HAL_SD" / name, dst)
    for name in ("stm32n6xx_hal_sd.h", "stm32n6xx_hal_sd_ex.h", "stm32n6xx_ll_sdmmc.h", "stm32n6xx_ll_dlyb.h"):
        dst = hal_inc / name
        backups.append(_backup(dst))
        shutil.copy2(fw / "Drivers" / "HAL_SD" / name, dst)

    # --- 4. Copy BSP SD driver ---------------------------------------------
    for name in ("stm32n6570_discovery_sd.c", "stm32n6570_discovery_sd.h"):
        dst = bsp_dk / name
        backups.append(_backup(dst))
        shutil.copy2(fw / "Drivers" / name, dst)

    # --- 5. Copy FatFs middleware -------------------------------------------
    fatfs_dst.mkdir(exist_ok=True)
    backups.append(fatfs_dst)  # entire dir — removed on cleanup
    for p in (fw / "Drivers" / "FatFs").iterdir():
        if p.is_file():
            shutil.copy2(p, fatfs_dst / p.name)

    # --- 6. Generate app_config.h from model_config.json ------------------
    #  Replaces the NPU_Validation app_config.h with a generated version
    #  that includes both the board support defines and our audio/inference
    #  parameters from model_config.json.
    backups.append(_backup(app_conf))
    _patch_app_config(app_conf, model_cfg, num_classes)

    # --- 7. Patch hal_conf.h — enable HAL_SD module ------------------------
    backups.append(_backup(hal_conf))
    text = hal_conf.read_text()
    if HAL_SD_COMMENTED in text:
        hal_conf.write_text(text.replace(HAL_SD_COMMENTED, HAL_SD_UNCOMMENTED))
        log.info("Enabled HAL_SD_MODULE in hal_conf.h")

    # --- 8. Patch Makefile — add our sources and include paths -------------
    backups.append(_backup(makefile))
    _patch_makefile(makefile)

    log.info("Project patched (%d backups created)", len(backups))
    return backups


def _patch_app_config(path: Path, model_cfg: dict, num_classes: int) -> None:
    """Overwrite app_config.h with values generated from model_config.json."""
    gen = _load_gen_app_config()
    path.write_text(gen.generate_app_config_h(model_cfg, num_classes))
    log.info("Generated app_config.h with model parameters")


MAKEFILE_END_SENTINEL = "# --- end BirdNET-STM32 additions ---"


def _patch_makefile(path: Path) -> None:
    """Insert BirdNET sources before the OBJECTS definition in the Makefile."""
    text = path.read_text()
    # Strip any leftover patch block (e.g. from a corrupted .bak restore).
    if MAKEFILE_SENTINEL in text:
        start = text.find(MAKEFILE_SENTINEL)
        end = text.find(MAKEFILE_END_SENTINEL)
        if start != -1 and end != -1:
            # Remove from the start of the sentinel line to end of end-marker line.
            line_start = text.rfind("\n", 0, start)
            line_start = 0 if line_start == -1 else line_start
            line_end = text.find("\n", end)
            line_end = len(text) if line_end == -1 else line_end + 1
            text = text[:line_start] + text[line_end:]
            log.info("Stripped leftover Makefile patch block")
        else:
            log.info("Makefile already patched — skipping")
            return

    addition = f"""\
{MAKEFILE_SENTINEL}
FATFS_PATH = $(PROJECT_PATH)/FatFs
DRIVER_SOURCES += $(N6_DRIVER_PATH)/Src/stm32n6xx_hal_sd.c
DRIVER_SOURCES += $(N6_DRIVER_PATH)/Src/stm32n6xx_ll_sdmmc.c
DRIVER_SOURCES += $(DK_DRIVER_PATH)/stm32n6570_discovery_sd.c
C_SOURCES += $(CORE_PATH)/Src/wav_reader.c
C_SOURCES += $(CORE_PATH)/Src/audio_stft.c
C_SOURCES += $(CORE_PATH)/Src/audio_mel.c
C_SOURCES += $(CORE_PATH)/Src/sd_handler.c
C_SOURCES += $(CORE_PATH)/Src/fft.c
C_SOURCES += $(FATFS_PATH)/ff.c
C_SOURCES += $(FATFS_PATH)/diskio.c
C_SOURCES += $(FATFS_PATH)/ff_gen_drv.c
C_SOURCES += $(FATFS_PATH)/sd_diskio.c
C_INCLUDES += -I$(FATFS_PATH)
{MAKEFILE_END_SENTINEL}
"""
    # Insert before OBJECTS definition so pattern substitution picks up our sources.
    # The OBJECTS line uses $(C_SOURCES:...) pattern substitution.
    marker = "OBJECTS = $(C_SOURCES:"
    idx = text.find(marker)
    if idx == -1:
        # Fallback: append to end.
        text += "\n" + addition
    else:
        # Insert just before the OBJECTS line.
        line_start = text.rfind("\n", 0, idx)
        if line_start == -1:
            line_start = 0
        else:
            line_start += 1
        text = text[:line_start] + addition + "\n" + text[line_start:]
    path.write_text(text)
    log.info("Patched Makefile with BirdNET sources")


def restore_project(backups: list[Path]) -> None:
    """Undo all project patches by restoring .bak files and removing new files.

    For each entry in *backups*:
    - If it is a directory (FatFs), remove it entirely.
    - If it is a ``.bak`` path and the .bak exists, move it back over the original.
    - If it is a ``.bak`` path but no .bak was created (file was new), delete
      the original that we copied in.
    """
    for entry in reversed(backups):
        if entry.is_dir() and entry.suffix != ".bak":
            shutil.rmtree(entry, ignore_errors=True)
        else:
            _restore(entry)
    log.info("Project restored")


# ---------------------------------------------------------------------------
# Serial capture
# ---------------------------------------------------------------------------


def _serial_capture(
    port: str,
    baudrate: int,
    lines: list[str],
    done: threading.Event,
    timeout: float,
) -> None:
    """Background thread: read UART lines until DONE marker or timeout.

    Handles board resets (port disconnection) by retrying the connection.
    """
    deadline = time.monotonic() + timeout
    ser = None

    while not done.is_set() and time.monotonic() < deadline:
        # (Re-)open the serial port if needed.
        if ser is None:
            try:
                ser = serial.Serial(port, baudrate, timeout=1)
            except serial.SerialException:
                time.sleep(0.5)
                continue

        try:
            raw = ser.readline()
        except serial.SerialException:
            # Port disconnected (board reset during flash).
            ser.close()
            ser = None
            continue

        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        lines.append(line)
        if DONE_MARKER in line:
            done.set()
            # Read remaining summary lines (Processed + Benchmark).
            for _ in range(5):
                try:
                    extra = ser.readline()
                    if not extra:
                        break
                    lines.append(extra.decode("utf-8", errors="replace").rstrip("\r\n"))
                except serial.SerialException:
                    break
            break

    if ser is not None:
        ser.close()


# ---------------------------------------------------------------------------
# Serial output parser
# ---------------------------------------------------------------------------

# Example firmware output:
#   [1/10] song_sparrow_01.wav
#     [1] Melospiza melodia_Song Sparrow: 87.3%
#     [2] Passerella iliaca_Fox Sparrow: 4.1%
_RE_FILE = re.compile(r"^\[(\d+)/(\d+)\]\s+(.+)$")
_RE_DET = re.compile(r"^\s+\[(\d+)\]\s+(.+?):\s+([\d.]+)%$")
_RE_SUMMARY = re.compile(r"^Processed:\s+(\d+)\s*/\s*(\d+)\s+files\s+\((\d+)\s+errors\)")
_RE_BENCH = re.compile(r"^\s+\[BENCH\]\s+read=(\d+)ms\s+stft=(\d+)ms\s+npu=(\d+)ms\s+total=(\d+)ms$")
_RE_BENCH_SUMMARY = re.compile(r"^Benchmark:.*avg read=(\d+)ms\s+stft=(\d+)ms\s+npu=(\d+)ms\s+total=(\d+)ms\)$")


def parse_serial_output(
    lines: list[str],
    labels: list[str],
    top_k: int = 5,
    threshold: float = 0.01,
) -> dict:
    """Parse firmware UART output into structured results.

    Args:
        lines: Raw serial lines captured from the board.
        labels: Class label list (for reference; firmware already prints names).
        top_k: Max detections per file to keep.
        threshold: Minimum score fraction (0–1) to include.

    Returns:
        Dict with 'results' (list of per-file dicts), 'processed', 'errors',
        'raw_lines'.
    """
    results: list[dict] = []
    current_file: dict | None = None
    processed = 0
    total = 0
    errors = 0
    benchmark_avg: dict | None = None

    for line in lines:
        m = _RE_FILE.match(line)
        if m:
            if current_file is not None:
                results.append(current_file)
            current_file = {
                "file": m.group(3),
                "detections": [],
                "bench": None,
            }
            continue

        m = _RE_DET.match(line)
        if m and current_file is not None:
            score = float(m.group(3)) / 100.0
            if score >= threshold and len(current_file["detections"]) < top_k:
                current_file["detections"].append(
                    {
                        "label": m.group(2),
                        "score": score,
                    }
                )
            continue

        m = _RE_BENCH.match(line)
        if m and current_file is not None:
            current_file["bench"] = {
                "read_ms": int(m.group(1)),
                "stft_ms": int(m.group(2)),
                "npu_ms": int(m.group(3)),
                "total_ms": int(m.group(4)),
            }
            continue

        m = _RE_BENCH_SUMMARY.match(line)
        if m:
            benchmark_avg = {
                "avg_read_ms": int(m.group(1)),
                "avg_stft_ms": int(m.group(2)),
                "avg_npu_ms": int(m.group(3)),
                "avg_total_ms": int(m.group(4)),
            }
            continue

        m = _RE_SUMMARY.match(line)
        if m:
            processed = int(m.group(1))
            total = int(m.group(2))
            errors = int(m.group(3))

    if current_file is not None:
        results.append(current_file)

    return {
        "results": results,
        "processed": processed,
        "total": total,
        "errors": errors,
        "benchmark": benchmark_avg,
        "raw_lines": lines,
    }


# ---------------------------------------------------------------------------
# Host x board parity
# ---------------------------------------------------------------------------

# Firmware defaults (firmware/gen_app_config.py): it prints at most this many
# labels per file, and none scoring below the threshold.
FIRMWARE_TOP_K = 5
FIRMWARE_SCORE_THRESHOLD = 0.01


def firmware_top_k(scores: np.ndarray, k: int, threshold: float) -> list[int]:
    """Return the class indices the firmware would print, in its order.

    A replica of ``print_top_k`` in firmware/Src/main.c: a partial selection
    sort that swaps only on a strictly greater score, stopping at the first
    score below ``threshold``. INT8 outputs tie often -- several classes can
    saturate at the same code -- so the tie order has to be the firmware's,
    not numpy's.
    """
    scores = np.asarray(scores, dtype=np.float32)
    indices = list(range(len(scores)))
    for i in range(min(k, len(scores))):
        for j in range(i + 1, len(scores)):
            if scores[indices[j]] > scores[indices[i]]:
                indices[i], indices[j] = indices[j], indices[i]
    picked = []
    for i in range(min(k, len(scores))):
        if scores[indices[i]] < threshold:
            break
        picked.append(indices[i])
    return picked


def host_reference_scores(
    model_path: str, model_cfg: dict, audio_dir: str | Path, board_files: list[str]
) -> dict[str, np.ndarray]:
    """Score the board's files on the host with the same .tflite.

    Each file goes through the host's evaluation preprocessing
    (``make_chunks_for_file``), and its first chunk is scored -- the chunk the
    firmware reads. Using the host pipeline rather than a re-implementation of
    the firmware is deliberate: a firmware frontend that diverges from what the
    model was evaluated with has to show up as a parity failure.

    Args:
        model_path: The .tflite deployed to the board.
        model_cfg: Its model config.
        audio_dir: Local copy of the SD card's audio/ folder.
        board_files: File names as the board printed them (FAT may upper-case).

    Returns:
        Board file name -> [num_classes] host scores.
    """
    from birdnet_stm32.evaluation.metrics import make_chunks_for_file
    from birdnet_stm32.models.runners import TFLiteRunner, runner_for_config

    local = {path.name.upper(): path for path in Path(audio_dir).iterdir() if path.suffix.lower() == ".wav"}
    missing = [name for name in board_files if name.upper() not in local]
    if missing:
        raise FileNotFoundError(f"No local copy of board file(s) {missing} in {audio_dir}")
    runner = runner_for_config(TFLiteRunner(model_path), model_cfg)
    scores = {}
    for name in board_files:
        chunks = make_chunks_for_file(
            str(local[name.upper()]),
            model_cfg,
            model_cfg["audio_frontend"],
            model_cfg.get("mag_scale", "none"),
            int(model_cfg["fft_length"]),
            0.0,
        )
        if not chunks:
            raise RuntimeError(f"Host could not read {name}")
        scores[name] = np.asarray(runner.predict(np.asarray(chunks[0], np.float32)[None]))[0]
    return scores


def load_truth(audio_dir: str | Path) -> dict[str, str]:
    """Read ``file -> true_species`` from a manifest.csv beside or above audio_dir."""
    import csv

    for candidate in (Path(audio_dir) / "manifest.csv", Path(audio_dir).parent / "manifest.csv"):
        if candidate.is_file():
            with candidate.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            if rows and {"file", "true_species"} <= rows[0].keys():
                return {row["file"].upper(): row["true_species"] for row in rows}
    return {}


def compare_board_host(
    board_results: list[dict],
    host_scores: dict[str, np.ndarray],
    labels: list[str],
    *,
    top_k: int,
    threshold: float,
    tolerance: float,
    detection_threshold: float = 0.5,
    truth: dict[str, str] | None = None,
) -> dict:
    """Check that the board reports the same detections as the host.

    A file agrees when both hold:

    - the board's top-1 is the host's top-1, or a label the host scores within
      ``tolerance`` of its own top-1 (a tie, not a disagreement);
    - board and host make the same call at ``detection_threshold`` -- unless
      the host's score is within ``tolerance`` of the threshold, where INT8
      rounding on the NPU can legitimately tip it (flagged ``borderline``).

    Score differences above ``tolerance`` that change neither are flagged
    ``drift`` but do not fail the file. A broken model fails on top-1.

    Returns:
        Dict with per-file ``rows``, ``passed``, and summary counts.
    """
    index = {label: position for position, label in enumerate(labels)}
    truth = truth or {}
    rows = []
    for result in board_results:
        name = result["file"]
        scores = np.asarray(host_scores[name], dtype=np.float32)
        host_top = firmware_top_k(scores, top_k, threshold)
        board = [(d["label"], d["score"]) for d in result["detections"]]
        unknown = [label for label, _ in board if label not in index]
        if unknown:
            raise ValueError(f"{name}: board printed labels the model does not have: {unknown}")
        board_top1 = board[0][0] if board else ""
        host_top1 = labels[host_top[0]] if host_top else ""
        max_diff = max((abs(score - float(scores[index[label]])) for label, score in board), default=0.0)
        if board_top1 == host_top1:
            agreement = "match"
        elif board_top1 and host_top1 and scores[index[board_top1]] >= scores[host_top[0]] - tolerance:
            agreement = "tie"
        else:
            agreement = "mismatch"
        board_score = board[0][1] if board else 0.0
        host_score = float(scores[host_top[0]]) if host_top else 0.0
        same_call = (board_score >= detection_threshold) == (host_score >= detection_threshold)
        borderline = not same_call and abs(host_score - detection_threshold) <= tolerance
        flags = [flag for flag, on in (("borderline", borderline), ("drift", max_diff > tolerance)) if on]
        true_label = truth.get(name.upper(), "")
        rows.append(
            {
                "file": name,
                "board_top1": board_top1,
                "board_score": board_score,
                "host_top1": host_top1,
                "host_score": host_score,
                "board_labels": [label for label, _ in board],
                "host_labels": [labels[i] for i in host_top],
                "max_score_diff": float(max_diff),
                "agreement": agreement,
                "flags": flags,
                "ok": agreement != "mismatch" and (same_call or borderline),
                "true_label": true_label,
            }
        )
    with_truth = [row for row in rows if row["true_label"]]
    return {
        "rows": rows,
        "passed": bool(rows) and all(row["ok"] for row in rows),
        "files": len(rows),
        "agreeing": sum(row["ok"] for row in rows),
        "top1_matches": sum(row["agreement"] == "match" for row in rows),
        "ties": sum(row["agreement"] == "tie" for row in rows),
        "max_score_diff": max((row["max_score_diff"] for row in rows), default=0.0),
        "tolerance": tolerance,
        "detection_threshold": detection_threshold,
        "flagged": sum(bool(row["flags"]) for row in rows),
        "board_correct": sum(row["board_top1"] == row["true_label"] for row in with_truth) if with_truth else None,
        "host_correct": sum(row["host_top1"] == row["true_label"] for row in with_truth) if with_truth else None,
        "labelled_files": len(with_truth),
    }


def print_parity(parity: dict) -> None:
    """Print the per-file host x board comparison and its verdict."""
    print("\n--- Host x board parity ---")
    print(f"  {'file':<16} {'board top-1':<28} {'host top-1':<28} {'max|d|':>7}  {'':<18} truth")
    for row in parity["rows"]:
        board = f"{row['board_top1']} {row['board_score']:.3f}"
        host = f"{row['host_top1']} {row['host_score']:.3f}"
        flag = "+".join([row["agreement"], *row["flags"]])
        flag = flag if row["ok"] else f"FAIL:{flag}"
        print(
            f"  {row['file']:<16} {board:<28} {host:<28} {row['max_score_diff']:7.4f}  {flag:<18} {row['true_label']}"
        )
    verdict = "PASS" if parity["passed"] else "FAIL"
    print(
        f"\n  PARITY {verdict}: {parity['agreeing']}/{parity['files']} files agree "
        f"({parity['top1_matches']} same top-1, {parity['ties']} ties), "
        f"detection threshold {parity['detection_threshold']}, {parity['flagged']} flagged, "
        f"max |board - host| {parity['max_score_diff']:.4f} (tolerance {parity['tolerance']})"
    )
    if parity["labelled_files"]:
        print(
            f"  Correct top-1: board {parity['board_correct']}/{parity['labelled_files']}, "
            f"host {parity['host_correct']}/{parity['labelled_files']}"
        )


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def run_board_test(cfg: BoardTestConfig) -> dict:
    """Execute the full on-board inference test.

    Steps:
    1. stedgeai generate → produce NPU binary from the TFLite model.
    2. Patch NPU_Validation project with our firmware sources.
    3. n6_loader: copy network.c, build firmware, flash, run.
    4. Capture UART output until '=== DONE ===' marker.
    5. Restore the NPU_Validation project to its original state.
    6. Parse and report results.

    Args:
        cfg: Board test configuration.

    Returns:
        Dict with 'results', 'processed', 'errors', 'raw_lines', 'labels'.
    """
    deploy = cfg.deploy_cfg

    # --- Validate prerequisites ---
    if not os.path.isfile(deploy.model_path):
        print(f"[ERROR] Model not found: {deploy.model_path}")
        sys.exit(1)
    if not os.path.isfile(deploy.stedgeai_path):
        print(f"[ERROR] stedgeai not found: {deploy.stedgeai_path}")
        sys.exit(1)
    if not os.path.isfile(cfg.model_config_path):
        print(f"[ERROR] Model config not found: {cfg.model_config_path}")
        sys.exit(1)
    if not os.path.isfile(deploy.n6_loader_script):
        print(f"[ERROR] n6_loader.py not found: {deploy.n6_loader_script}")
        sys.exit(1)
    if not os.path.isfile(deploy.n6_loader_config):
        print(f"[ERROR] n6_loader config not found: {deploy.n6_loader_config}")
        sys.exit(1)

    model_cfg = load_model_config(cfg.model_config_path)
    labels = load_labels(cfg.labels_path) if cfg.labels_path else []
    num_classes = len(labels) if labels else model_cfg.get("num_classes", 1000)

    print("\n=== BirdNET-STM32 Board Test (standalone firmware) ===\n")
    print(f"  Model:       {deploy.model_path}")
    print(f"  Config:      {cfg.model_config_path}")
    print(f"  Labels:      {cfg.labels_path or '(none)'} ({num_classes} classes)")
    print(f"  Serial port: {cfg.serial_port}")
    print(f"  Timeout:     {cfg.timeout}s")
    print()

    project = _resolve_project_path(deploy)

    # Step 1: stedgeai generate
    print("--- Step 1: Generate NPU binary (stedgeai generate) ---")
    generate(deploy)

    # Step 2: Patch the NPU_Validation project
    print("\n--- Step 2: Patch NPU_Validation project ---")
    backups = patch_project(project, model_cfg, num_classes, labels)

    try:
        # Step 3: Start serial capture (before n6_loader starts the firmware)
        print("\n--- Step 3: Build, flash, and run ---")
        serial_lines: list[str] = []
        done_event = threading.Event()
        serial_thread = threading.Thread(
            target=_serial_capture,
            args=(cfg.serial_port, UART_BAUDRATE, serial_lines, done_event, cfg.timeout),
            daemon=True,
        )
        serial_thread.start()

        # Step 4: n6_loader — copies network.c, builds, flashes, runs firmware
        n6_cmd = [
            sys.executable,
            deploy.n6_loader_script,
            "--n6-loader-config",
            deploy.n6_loader_config,
            "--clean",
        ]
        print(f"  $ {' '.join(n6_cmd)}")
        result = subprocess.run(n6_cmd, check=False)
        if result.returncode != 0:
            print(f"[ERROR] n6_loader failed (exit code {result.returncode})")
            sys.exit(result.returncode)

        # Wait for firmware to finish (or timeout)
        print("\n--- Step 4: Waiting for firmware output ---")
        if not done_event.is_set():
            done_event.wait(timeout=cfg.timeout)
        serial_thread.join(timeout=5)

        if not done_event.is_set():
            print(f"[WARN] Timeout after {cfg.timeout}s — firmware may not have finished")

    finally:
        # Step 5: Restore original project files
        print("\n--- Step 5: Restore NPU_Validation project ---")
        restore_project(backups)

    # Step 6: Parse and display results
    print("\n--- Raw UART output ---")
    for sl in serial_lines:
        print(f"  | {sl}")

    print("\n--- Results ---")
    parsed = parse_serial_output(serial_lines, labels, cfg.top_k, cfg.score_threshold)

    for r in parsed["results"]:
        det_str = ", ".join(f"{d['label']} ({d['score']:.1%})" for d in r["detections"])
        print(f"\n  {r['file']}")
        if det_str:
            print(f"    {det_str}")
        else:
            print("    (no detections above threshold)")
        if r.get("bench"):
            b = r["bench"]
            print(f"    [{b['read_ms']}ms read, {b['stft_ms']}ms STFT, {b['npu_ms']}ms NPU, {b['total_ms']}ms total]")

    print(f"\n=== DONE: {parsed['processed']}/{parsed['total']} files ({parsed['errors']} errors) ===")

    bench = parsed.get("benchmark")
    if bench:
        chunk_sec = model_cfg.get("chunk_duration", 3)
        avg_total = bench["avg_total_ms"]
        rtf = avg_total / (chunk_sec * 1000) if chunk_sec else 0
        speedup = (chunk_sec * 1000) / avg_total if avg_total else 0
        print(f"\n--- Benchmark (per file, averaged over {parsed['processed']} files) ---")
        print(f"  SD read:        {bench['avg_read_ms']} ms")
        print(f"  STFT (M55):     {bench['avg_stft_ms']} ms")
        print(f"  NPU inference:  {bench['avg_npu_ms']} ms")
        print(f"  Total:          {avg_total} ms")
        print(f"  Real-time factor: {rtf:.4f}x  ({speedup:.0f}x faster than real-time)")

    parsed["labels"] = labels
    parsed["parity"] = None
    if cfg.host_audio_dir:
        if not labels:
            raise ValueError("Host x board parity needs the model's labels file")
        board_files = [r["file"] for r in parsed["results"]]
        local = [p for p in Path(cfg.host_audio_dir).iterdir() if p.suffix.lower() == ".wav"]
        unprocessed = sorted({p.name.upper() for p in local} - {name.upper() for name in board_files})
        host_scores = host_reference_scores(deploy.model_path, model_cfg, cfg.host_audio_dir, board_files)
        parity = compare_board_host(
            parsed["results"],
            host_scores,
            labels,
            top_k=min(cfg.top_k, FIRMWARE_TOP_K),
            threshold=max(cfg.score_threshold, FIRMWARE_SCORE_THRESHOLD),
            tolerance=cfg.parity_tolerance,
            detection_threshold=cfg.detection_threshold,
            truth=load_truth(cfg.host_audio_dir),
        )
        if unprocessed:
            parity["passed"] = False
            print(f"\n[WARN] Not processed on the board: {unprocessed}")
        parity["unprocessed"] = unprocessed
        print_parity(parity)
        parsed["parity"] = parity
    return parsed
