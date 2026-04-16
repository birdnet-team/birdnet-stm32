"""On-board inference test for the STM32N6570-DK.

Full end-to-end pipeline:
1. Pre-compute STFT spectrograms from real audio files (host).
2. Deploy the model: stedgeai generate → n6_loader compile+flash (automated).
3. Run stedgeai validate with real spectrograms → NPU inference on the board.
4. Read back on-device predictions, map to class labels, and report.

The NPU inference runs on real hardware with real audio data. The STFT is
deterministic fixed math (identical on host and target), so it is computed
on the host to avoid needing the STM32CubeN6 FatFs/SDMMC/CMSIS-DSP sources
that are not part of X-CUBE-AI.
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from birdnet_stm32.deploy.config import DeployConfig


@dataclass
class BoardTestConfig:
    """Configuration for on-board inference tests.

    Attributes:
        deploy_cfg: Base deployment configuration.
        model_config_path: Path to the _model_config.json file.
        labels_path: Path to the _labels.txt file.
        audio_dir: Directory of WAV files to test.
        top_k: Number of top predictions to show per file.
        score_threshold: Minimum score to display.
    """

    deploy_cfg: DeployConfig = field(default_factory=DeployConfig)
    model_config_path: str = ""
    labels_path: str = ""
    audio_dir: str = "data/test"
    top_k: int = 5
    score_threshold: float = 0.01


def load_model_config(config_path: str) -> dict:
    """Load model configuration from JSON.

    Args:
        config_path: Path to _model_config.json.

    Returns:
        Dict with model configuration.
    """
    with open(config_path) as f:
        return json.load(f)


def load_labels(labels_path: str) -> list[str]:
    """Load class labels from a labels.txt file.

    Args:
        labels_path: Path to _labels.txt (one label per line).

    Returns:
        List of label strings.
    """
    with open(labels_path) as f:
        return [line.strip() for line in f if line.strip()]


def prepare_spectrograms(
    audio_dir: str,
    model_cfg: dict,
    max_files: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """Compute STFT spectrograms from WAV files for on-board inference.

    Matches the hybrid frontend: linear magnitude STFT, normalized to [0, 1],
    shape [N, fft_bins, spec_width, 1].

    Args:
        audio_dir: Directory tree containing WAV files.
        model_cfg: Model config dict (sample_rate, fft_length, etc.).
        max_files: Maximum number of files to process (0 = all).

    Returns:
        Tuple of (spectrograms array, list of file paths).
    """
    from utils.audio import get_spectrogram_from_audio, load_audio_file

    sr = model_cfg["sample_rate"]
    n_fft = model_cfg["fft_length"]
    spec_width = model_cfg["spec_width"]
    chunk_duration = model_cfg["chunk_duration"]
    fft_bins = n_fft // 2 + 1

    # Collect WAV files
    wav_files = sorted(
        str(p) for p in Path(audio_dir).rglob("*.wav")
    )
    if not wav_files:
        print(f"[ERROR] No .wav files found in {audio_dir}")
        sys.exit(1)
    if max_files > 0:
        wav_files = wav_files[:max_files]

    print(f"[OK] Found {len(wav_files)} audio files in {audio_dir}")

    spectrograms = []
    file_paths = []

    for i, wav_path in enumerate(wav_files):
        chunks = load_audio_file(
            wav_path,
            sample_rate=sr,
            chunk_duration=chunk_duration,
            random_offset=False,
        )
        if len(chunks) == 0:
            print(f"  [{i + 1}] SKIP {wav_path} (could not load)")
            continue

        # For each file, compute spectrogram of the first chunk
        # (matching how the model sees a single 3s window)
        chunk = chunks[0]
        spec = get_spectrogram_from_audio(
            chunk,
            sample_rate=sr,
            n_fft=n_fft,
            mel_bins=-1,  # linear STFT for hybrid frontend
            spec_width=spec_width,
        )
        if spec is None or spec.size == 0:
            print(f"  [{i + 1}] SKIP {wav_path} (empty spectrogram)")
            continue

        assert spec.shape == (fft_bins, spec_width), (
            f"Expected ({fft_bins}, {spec_width}), got {spec.shape}"
        )
        spectrograms.append(spec.astype(np.float32))
        file_paths.append(wav_path)
        rel = os.path.relpath(wav_path, audio_dir)
        print(f"  [{i + 1}/{len(wav_files)}] {rel}")

    if not spectrograms:
        print("[ERROR] No valid spectrograms produced")
        sys.exit(1)

    # Stack into [N, fft_bins, spec_width, 1]
    arr = np.stack(spectrograms)[:, :, :, np.newaxis]
    print(f"[OK] Prepared {arr.shape[0]} spectrograms, shape {arr.shape}")
    return arr, file_paths


def run_board_test(cfg: BoardTestConfig) -> dict:
    """Execute the full on-board inference test.

    Steps:
    1. Pre-compute STFT spectrograms from audio files.
    2. Deploy model to board (generate + compile + flash).
    3. Run on-target validation with real spectrograms.
    4. Read back NPU predictions and report.

    Args:
        cfg: Board test configuration.

    Returns:
        Dict with 'files', 'predictions', and 'labels' keys.
    """
    deploy = cfg.deploy_cfg

    # Validate prerequisites
    if not os.path.isfile(deploy.model_path):
        print(f"[ERROR] Model not found: {deploy.model_path}")
        sys.exit(1)
    if not os.path.isfile(deploy.stedgeai_path):
        print(f"[ERROR] stedgeai not found: {deploy.stedgeai_path}")
        sys.exit(1)
    if not os.path.isfile(cfg.model_config_path):
        print(f"[ERROR] Model config not found: {cfg.model_config_path}")
        sys.exit(1)

    model_cfg = load_model_config(cfg.model_config_path)
    labels = load_labels(cfg.labels_path) if cfg.labels_path else []

    print("\n=== BirdNET-STM32 Board Test ===\n")
    print(f"Model:      {deploy.model_path}")
    print(f"Config:     {cfg.model_config_path}")
    print(f"Labels:     {cfg.labels_path or '(none)'} ({len(labels)} classes)")
    print(f"Audio dir:  {cfg.audio_dir}")
    print(f"Top-K:      {cfg.top_k}")
    print()

    # Step 1: Prepare spectrograms from real audio
    print("--- Step 1: Prepare spectrograms from audio files ---")
    specs, file_paths = prepare_spectrograms(cfg.audio_dir, model_cfg)

    # Save as .npy for stedgeai --valinput
    valinput_path = os.path.join(deploy.output_dir, "board_test_inputs.npy")
    os.makedirs(deploy.output_dir, exist_ok=True)
    np.save(valinput_path, specs)
    print(f"[OK] Saved validation inputs to {valinput_path}")

    # Step 2: Deploy model to board (generate + compile + flash)
    print("\n--- Step 2: Deploy model to board ---")
    from birdnet_stm32.deploy.stedgeai import generate, load_to_target

    generate(deploy)
    load_to_target(deploy)

    # Step 3: Run on-target validation with real spectrograms
    print("\n--- Step 3: Run on-target inference ---")
    validate_cmd = [
        deploy.stedgeai_path,
        "validate",
        "--model", deploy.model_path,
        "--target", "stm32n6",
        "--mode", "target",
        "--desc", "serial:921600",
        "--valinput", valinput_path,
        "--output", deploy.output_dir,
        "--workspace", deploy.workspace_dir,
        "--save-csv",
        "--no-exec-model",
    ]
    print(f"  $ {' '.join(validate_cmd)}")
    result = subprocess.run(validate_cmd, check=False)
    if result.returncode != 0:
        print(f"[ERROR] On-target validation failed (exit code {result.returncode})")
        sys.exit(result.returncode)

    # Step 4: Read back on-device predictions
    print("\n--- Step 4: Results ---")
    c_outputs_path = os.path.join(deploy.output_dir, "network_val_c_outputs_1.npy")
    if not os.path.isfile(c_outputs_path):
        print(f"[ERROR] On-device outputs not found: {c_outputs_path}")
        sys.exit(1)

    predictions = np.load(c_outputs_path)
    # stedgeai may add extra batch dims — squeeze to [N, num_classes]
    while predictions.ndim > 2:
        predictions = predictions.squeeze(axis=1)
    print(f"[OK] Read {predictions.shape[0]} predictions from board, shape {predictions.shape}")

    num_classes = predictions.shape[1]
    if len(labels) != num_classes:
        labels = [f"class_{i}" for i in range(num_classes)]

    # Print per-file results
    results = []
    for idx, fpath in enumerate(file_paths):
        if idx >= predictions.shape[0]:
            break
        scores = predictions[idx]
        top_indices = np.argsort(scores)[::-1][: cfg.top_k]
        rel_path = os.path.relpath(fpath, cfg.audio_dir)

        detections = []
        for _rank, ci in enumerate(top_indices):
            score = float(scores[ci])
            if score < cfg.score_threshold:
                continue
            detections.append({"label": labels[ci], "score": score})

        results.append({"file": rel_path, "detections": detections})

        det_str = ", ".join(
            f"{d['label']} ({d['score']:.1%})" for d in detections
        )
        print(f"\n  [{idx + 1}/{len(file_paths)}] {rel_path}")
        if det_str:
            print(f"    {det_str}")
        else:
            print("    (no detections above threshold)")

    print(f"\n=== DONE: {len(results)} files tested on board NPU ===")
    return {"files": file_paths, "predictions": predictions, "labels": labels, "results": results}
