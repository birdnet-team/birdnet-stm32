"""
Evaluation script for birdnet-stm32 models on a class-structured audio dataset.

This module:
- Loads either a Keras (.keras) or TFLite (.tflite) model and wraps it with a simple runner.
- Splits each audio file into fixed-length chunks matching training, runs inference per chunk,
  pools chunk-level scores to file-level, and computes metrics.
- Optionally renders a few frontend spectrograms for sanity checking (Keras models only).
- Prints summary metrics and simple ASCII visualizations (histogram, PR curve). Can save CSV.

Input conventions by frontend (see training config JSON):
- precomputed/librosa: inputs are mel spectrograms [B, num_mels, spec_width, 1]
- hybrid: inputs are linear STFT magnitudes [B, fft_bins, spec_width, 1]
- raw/tf: inputs are waveforms [B, T, 1], T = sample_rate * chunk_duration, peak-normalized to [-1,1]

Metrics:
- ROC-AUC (micro), F1/precision/recall at 0.5 threshold, AP per class, class-mean AP (cmAP), and micro mAP.

Notes:
- Random seeds are set (NumPy, TF) but full determinism is not enforced.
- TFLite interpreter runs without delegates here to reduce numeric variation across platforms.
"""

import os
import argparse
import json
import math
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import warnings

from train import SUPPORTED_AUDIO_EXTS, load_file_paths_from_directory, AudioFrontendLayer, dataset_sanity_check
from utils.audio import load_audio_file, get_spectrogram_from_audio

np.random.seed(42)
tf.random.set_seed(42)

# Mute TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_model_runner(model_path):
    """
    Load a .keras or .tflite model and return a runner with a predict(batch) method.

    Args:
        model_path (str): Path to a saved model (.keras or .tflite).

    Returns:
        KerasRunner | TFLiteRunner: Object exposing predict(x_batch) -> np.ndarray with shape [B, C].
    """
    if model_path.lower().endswith(".tflite"):
        return TFLiteRunner(model_path)
    # Keras model
    model = tf.keras.models.load_model(
        model_path, compile=False, custom_objects={"AudioFrontendLayer": AudioFrontendLayer}
    )
    return KerasRunner(model)


class KerasRunner:
    """
    Thin wrapper for a Keras model to standardize batch prediction.
    """

    def __init__(self, model: tf.keras.Model):
        """
        Args:
            model (tf.keras.Model): Loaded Keras model (compiled=False is fine).
        """
        self.model = model
        # Determine input names for dict feeding
        try:
            self.input_names = [t.name.split(":")[0] for t in self.model.inputs]
        except Exception:
            self.input_names = []

    def predict(self, x_batch: np.ndarray) -> np.ndarray:
        """
        Run a forward pass on a batch.

        Args:
            x_batch (np.ndarray): Input batch in the model's expected shape and dtype.

        Returns:
            np.ndarray: Model outputs [B, C] as float32.
        """
        x_batch = x_batch.astype(np.float32, copy=False)
        # Prefer dict feeding to match named inputs (avoids warnings)
        if getattr(self, "input_names", None) and len(self.input_names) == 1:
            feed = {self.input_names[0]: x_batch}
            try:
                return self.model(feed, training=False).numpy()
            except Exception:
                pass
        # Fallback: positional feeding
        return self.model(x_batch, training=False).numpy()


class TFLiteRunner:
    """
    TFLite model runner using the builtin interpreter (no delegates).
    """

    def __init__(self, model_path: str):
        """
        Args:
            model_path (str): Path to a .tflite model file.
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path, experimental_delegates=[])
        self.input_index = None
        self.output_index = None
        self._allocate()

    def _allocate(self):
        """
        Allocate tensors and cache input/output tensor indices.
        """
        self.interpreter.allocate_tensors()
        in_det = self.interpreter.get_input_details()[0]
        out_det = self.interpreter.get_output_details()[0]
        self.input_index = in_det["index"]
        self.output_index = out_det["index"]

    def _ensure_shape(self, shape):
        """
        Resize the interpreter input tensor to match the batch shape, if needed.

        Args:
            shape (tuple[int, ...]): Desired input tensor shape.
        """
        in_det = self.interpreter.get_input_details()[0]
        cur = in_det["shape"]
        if list(cur) != list(shape):
            self.interpreter.resize_tensor_input(self.input_index, shape)
            self._allocate()

    def predict(self, x_batch: np.ndarray) -> np.ndarray:
        """
        Run a forward pass on a batch.

        Args:
            x_batch (np.ndarray): Input batch in the model's expected shape and dtype.

        Returns:
            np.ndarray: Model outputs [B, C] as float32.
        """
        # TFLite supports batching; ensure matching shape then run once
        x_batch = x_batch.astype(np.float32, copy=False)
        self._ensure_shape(x_batch.shape)
        self.interpreter.set_tensor(self.input_index, x_batch)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_index)
        return out


def lme_pooling(scores: np.ndarray, beta: float = 10.0) -> np.ndarray:
    """
    Log-Mean-Exponential pooling over chunks:
        pooled = log(mean(exp(beta * s_i))) / beta

    Args:
        scores (np.ndarray): [N_chunks, C] chunk scores in [0, 1].
        beta (float): Temperature; beta→0 approximates mean, beta→∞ approximates max.

    Returns:
        np.ndarray: [C] pooled scores.

    Notes:
        Uses a max-trick for numerical stability on large beta.
    """
    if scores.size == 0:
        return scores
    m = np.max(beta * scores, axis=0, keepdims=True)
    lme = m + np.log(np.mean(np.exp(beta * scores - m), axis=0, keepdims=True) + 1e-12)
    return (lme / beta).ravel()


def pool_scores(chunk_scores: np.ndarray, method: str = "average", beta: float = 10.0) -> np.ndarray:
    """
    Pool chunk-level scores [N, C] to file-level [C].

    Args:
        chunk_scores (np.ndarray): Array of shape [N_chunks, C].
        method (str): 'avg'|'mean'|'average' | 'max' | 'lme' (log-mean-exp).
        beta (float): Temperature for 'lme' pooling.

    Returns:
        np.ndarray: [C] pooled scores.
    """
    method = method.lower()
    if chunk_scores.ndim != 2:
        raise ValueError("chunk_scores must be [N_chunks, C]")
    if chunk_scores.shape[0] == 0:
        return np.zeros((chunk_scores.shape[1],), dtype=np.float32)
    if method in ("avg", "mean", "average"):
        return np.mean(chunk_scores, axis=0)
    if method == "max":
        return np.max(chunk_scores, axis=0)
    if method in ("lme", "log_mean_exp", "log_mean_exponential"):
        return lme_pooling(chunk_scores, beta=beta)
    raise ValueError(f"Unsupported pooling method: {method}")


def make_chunks_for_file(path: str, cfg: dict, frontend: str, mag_scale: str, n_fft: int, chunk_overlap: float):
    """
    Build a list of model-ready inputs from one audio file by chunking.

    Shapes per frontend:
      - precomputed/librosa: [mel_bins, spec_width, 1] (mel spectrogram magnitude)
      - hybrid:              [fft_bins, spec_width, 1] (linear STFT magnitude)
      - raw/tf:              [T, 1] waveform, peak-normalized to [-1, 1]

    Args:
        path (str): Audio file path.
        cfg (dict): Training config dict with keys: sample_rate, chunk_duration, num_mels, spec_width.
        frontend (str): 'precomputed'|'librosa'|'hybrid'|'raw'|'tf'.
        mag_scale (str): Magnitude scaling applied in precomputed paths ('none'|'pcen'|'pwl'|'db').
        n_fft (int): FFT length for spectrogram computation.
        chunk_overlap (float): Overlap fraction (seconds) between consecutive chunks.

    Returns:
        list[np.ndarray]: List of per-chunk inputs ready for model.predict.
    """
    sr = int(cfg["sample_rate"])
    cd = int(cfg["chunk_duration"])
    num_mels = int(cfg["num_mels"])
    spec_width = int(cfg["spec_width"])
    T = int(sr * cd)

    chunks = load_audio_file(path, sample_rate=sr, max_duration=60, chunk_duration=cd, random_offset=False, chunk_overlap=chunk_overlap)
    
    out = []
    if frontend in ("precomputed", "librosa"):
        for ch in chunks:
            S = get_spectrogram_from_audio(
                ch, sample_rate=sr, n_fft=n_fft, mel_bins=num_mels, spec_width=spec_width, mag_scale=mag_scale
            )
            out.append(S[:, :, None].astype(np.float32))
    elif frontend == "hybrid":
        fft_bins = n_fft // 2 + 1
        for ch in chunks:
            S = get_spectrogram_from_audio(ch, sample_rate=sr, n_fft=n_fft, mel_bins=-1, spec_width=spec_width)  # linear mag
            if S.shape[0] != fft_bins:
                # guard, but get_spectrogram_from_audio should ensure this
                S = S[:fft_bins, :spec_width]
            out.append(S[:, :, None].astype(np.float32))
    elif frontend in ("tf", "raw"):
        chunk_len = int(cfg["chunk_duration"] * cfg["sample_rate"])
        for ch in chunks:
            x = ch[:chunk_len]
            if x.shape[0] < chunk_len:
                x = np.pad(x, (0, chunk_len - x.shape[0]))
            x = x / (np.max(np.abs(x)) + 1e-6)
            out.append(x[:, None].astype(np.float32))
    else:
        raise ValueError(f"Invalid audio_frontend: {frontend}")
    return out


def evaluate(model_runner, files, classes, cfg, pooling="average", batch_size=64, overlap=0.0, mep_beta=10.0):
    """
    Run inference per chunk, pool to file-level, and compute evaluation metrics.

    Args:
        model_runner (KerasRunner|TFLiteRunner): Runner exposing predict(x_batch).
        files (list[str]): List of file paths to evaluate.
        classes (list[str]): Ordered class names (used to derive labels from directory names).
        cfg (dict): Training config dict (sample_rate, chunk_duration, num_mels, spec_width, fft_length, audio_frontend, mag_scale).
        pooling (str): 'avg'|'max'|'lme' pooling for chunk-to-file aggregation.
        batch_size (int): Batch size for per-chunk inference.
        overlap (float): Overlap in seconds for chunking (0..chunk_duration-0.1).
        mep_beta (float): Temperature for LME pooling.

    Returns:
        tuple:
            - metrics (dict): roc-auc, f1, precision, recall, mAP, cmAP, ap_per_class.
            - per_file (list[dict]): One dict per file with file path, label, and pooled scores list.
            - y_true (np.ndarray): One-hot ground-truth labels [N_files, C].
            - y_scores (np.ndarray): Pooled scores [N_files, C].
    """
    frontend = cfg["audio_frontend"]
    mag_scale = cfg.get("mag_scale", "none")
    n_fft = int(cfg["fft_length"])
    num_classes = len(classes)

    y_true = []
    y_scores = []
    per_file = []

    # Single progress bar over files
    for path in tqdm(files, total=len(files), desc="Evaluating", unit="file"):
        # Determine class (single label) from parent folder; skip unknown
        label_name = os.path.basename(os.path.dirname(path))
        if label_name not in classes:
            continue
        target = np.zeros((num_classes,), dtype=np.float32)
        target[classes.index(label_name)] = 1.0

        # Build chunk inputs
        chunks = make_chunks_for_file(path, cfg, frontend, mag_scale, n_fft, overlap)
        if len(chunks) == 0:
            continue

        # Batch predict
        preds = []
        for i in range(0, len(chunks), batch_size):
            batch = np.stack(chunks[i : i + batch_size], axis=0)
            p = model_runner.predict(batch)
            preds.append(p)
        chunk_scores = np.concatenate(preds, axis=0)  # [N,C]

        pooled = pool_scores(chunk_scores, method=pooling, beta=mep_beta)

        y_true.append(target)
        y_scores.append(pooled)
        per_file.append(
            {
                "file": path,
                "label": label_name,
                "scores": pooled.tolist(),
            }
        )
        
        # DEBUG: Show label and top predicted class
        #top_pred_idx = np.argmax(pooled)
        #top_pred_class = classes[top_pred_idx]
        #top_pred_score = pooled[top_pred_idx]
        #print(f"File: {os.path.basename(path)}, True Label: {label_name}, Predicted: {top_pred_class} ({top_pred_score:.4f})")

    if len(y_true) == 0:
        raise RuntimeError("No valid test samples found for the provided class set.")

    y_true = np.asarray(y_true, dtype=np.float32)
    y_scores = np.asarray(y_scores, dtype=np.float32)

    metrics = {}
    
    # ROC-AUC
    try:
        metrics["roc-auc"] = float(roc_auc_score(y_true, y_scores, average="micro"))
    except Exception:
        metrics["roc-auc"] = float("nan")
        
    # F1 at 0.5 threshold
    y_pred = (y_scores >= 0.5).astype(np.float32)
    tp = np.sum(y_true * y_pred)
    fp = np.sum((1 - y_true) * y_pred)
    fn = np.sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    if precision + recall > 0:
        metrics["f1"] = 2 * (precision * recall) / (precision + recall)
    else:
        metrics["f1"] = 0.0 
    metrics["precision"] = precision
    metrics["recall"] = recall

    # Average Precision (AP)
    # Per-class AP
    ap_per_class = []
    for ci in range(y_true.shape[1]):
        try:
            ap = average_precision_score(y_true[:, ci], y_scores[:, ci])
        except Exception:
            ap = np.nan
        ap_per_class.append(ap)
    ap_valid = [a for a in ap_per_class if not (a is None or (isinstance(a, float) and math.isnan(a)))]
    metrics["ap_per_class"] = ap_per_class
    metrics["cmAP"] = float(np.mean(ap_valid)) if len(ap_valid) else float("nan")

    # Micro AP (treat all labels jointly)
    try:
        metrics["mAP"] = float(average_precision_score(y_true, y_scores, average="micro"))
    except Exception:
        metrics["maP"] = float("nan")

    return metrics, per_file, y_true, y_scores


def print_ascii_histogram(scores, bins=10, width=40):
    """
    Print an ASCII histogram of scores in [0,1].

    Args:
        scores (np.ndarray): Array of scores in [0, 1].
        bins (int): Number of histogram bins.
        width (int): Bar width in characters.
    """
    hist, bin_edges = np.histogram(scores, bins=bins, range=(0, 1))
    max_count = np.max(hist)
    for i in range(bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        bar = "#" * int(width * hist[i] / max_count) if max_count > 0 else ""
        print(f"{left:4.2f} - {right:4.2f} | {bar} ({hist[i]})")


def print_ascii_pr_curve(y_true, y_scores, bins=10, width=40):
    """
    Print an ASCII PR curve with fixed precision bins (1.0, 0.9, ..., 0.0).
    
    Args:
        y_true (np.ndarray): One-hot ground-truth labels [N_files, C].
        y_scores (np.ndarray): Pooled scores [N_files, C].
        bins (int): Number of precision bins.
        width (int): Bar width in characters.
    """
    # Flatten for micro-averaged PR
    y_true = y_true.ravel()
    y_scores = y_scores.ravel()
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    # Remove last point (sklearn adds an extra 0 recall, 1 precision point)
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    # Fixed bins: 1.0, 0.9, ..., 0.0 (top to bottom)
    bin_edges = np.linspace(1.0, 0.0, bins + 1)
    print("\nASCII Precision-Recall Curve (precision ↓, recall →):")
    for i in range(bins):
        p_lo = bin_edges[i + 1]
        p_hi = bin_edges[i]
        # Find recall for all points with precision in this bin
        mask = (precisions >= p_lo) & (precisions <= p_hi)
        if np.any(mask):
            max_recall = np.max(recalls[mask])
        else:
            max_recall = 0.0
        bar = "#" * int(width * max_recall)
        print(f"{p_hi:4.1f} | {bar} ({max_recall:4.2f})")
        

def save_predictions_csv(per_file, classes, out_path):
    """
    Save per-file predictions to CSV: file,label,top1_label,top1_score,<class columns...>
    Args:
        per_file (list[dict]): One dict per file with keys: file, label, scores (list of floats).
        classes (list[str]): Ordered class names.
        out_path (str): Path to save the CSV.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        header = ["file", "label", "top1_label", "top1_score"] + classes
        f.write(",".join(header) + "\n")
        for row in per_file:
            scores = np.array(row["scores"])
            top1_idx = int(np.argmax(scores))
            top1_label = classes[top1_idx]
            top1_score = scores[top1_idx]
            vals = [
                row["file"],
                row["label"],
                top1_label,
                f"{top1_score:.3f}",
            ] + [f"{s:.3f}" for s in scores]
            f.write(",".join(vals) + "\n")


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate model on test audio (file-level pooling).")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .keras or quantized .tflite")
    parser.add_argument("--model_config", type=str, default="", help="Path to <checkpoint>_model_config.json (defaults to infer from model_path)")
    parser.add_argument("--data_path_test", type=str, required=True, help="Path to test dataset root (class-subfolders)")
    parser.add_argument("--max_files", type=int, default=-1, help="Optional max number of test files per class to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for chunk inference")
    parser.add_argument("--overlap", type=float, default=0.0, help="Chunk overlap between consecutive audio segments (max. chunk_duration - 0.1)")
    parser.add_argument("--pooling", type=str, default="avg", choices=["avg", "max", "lme"], help="Pooling method across chunks")
    parser.add_argument("--save_csv", type=str, default="", help="Optional path to save per-file predictions CSV")
    return parser.parse_args()


def main():
    """Evaluate a trained model on a class-structured test dataset.
    1) Load model and its JSON config (infer config path if omitted).
    2) Collect test files and ground-truth labels.
    3) Run inference and compute metrics.
    4) Save per-file predictions (optional).
    """
    args = get_args()

    # Resolve model_config
    model_cfg_path = args.model_config
    if not model_cfg_path:
        root, _ = os.path.splitext(args.model_path)
        model_cfg_path = root + "_model_config.json"
    if not os.path.isfile(model_cfg_path):
        raise FileNotFoundError(f"Model config JSON not found: {model_cfg_path}")
    with open(model_cfg_path, "r") as f:
        cfg = json.load(f)

    # Class list from training
    classes = cfg.get("class_names", [])
    if not classes:
        raise ValueError("class_names missing in model_config; cannot evaluate.")

    # Collect test files restricted to known classes
    files, _ = load_file_paths_from_directory(args.data_path_test, 
                                              classes=classes, 
                                              exts=SUPPORTED_AUDIO_EXTS, 
                                              max_samples=args.max_files)
    if len(files) == 0:
        raise RuntimeError(f"No test audio found in {args.data_path_test} with exts {SUPPORTED_AUDIO_EXTS}")

    # Load model
    runner = load_model_runner(args.model_path)
    tf_frontend_layer = None
    if isinstance(runner, KerasRunner):
        # Try to extract the frontend layer by name or type
        for layer in runner.model.layers:
            if isinstance(layer, AudioFrontendLayer) or "frontend" in layer.name:
                tf_frontend_layer = layer
                break

    # Run sanity check on a few test files with the trained frontend (Keras only)
    if tf_frontend_layer is not None:
        test_files = files[:10]
        dataset_sanity_check(
            test_files, classes,
            sample_rate=cfg["sample_rate"],
            max_duration=cfg.get("max_duration", 30),
            chunk_duration=cfg["chunk_duration"],
            spec_width=cfg["spec_width"],
            mel_bins=cfg["num_mels"],
            audio_frontend=cfg["audio_frontend"],
            fft_length=cfg["fft_length"],
            mag_scale=cfg.get("mag_scale", "none"),
            snr_threshold=0.25,
            prefix='test',
            tf_frontend_layer=tf_frontend_layer
        )

    # Evaluate
    metrics, per_file, y_true, y_scores = evaluate(
        model_runner=runner,
        files=files,
        classes=classes,
        cfg=cfg,
        pooling=args.pooling,
        batch_size=args.batch_size,
        overlap=max(0.0, min(cfg['chunk_duration'] - 0.1, args.overlap)),
    )

    # Print summary
    print(f"Evaluated {len(per_file)} files across {len(classes)} classes.")
    for k, v in metrics.items():
        if k == "ap_per_class":
            continue
        print(f"{k}: {v:.4f}")
    if "ap_per_class" in metrics and metrics["ap_per_class"]:
        # Show top 10 and bottom 10 classes by AP
        ap_list = metrics["ap_per_class"]
        ap_with_names = [(classes[i], ap_list[i]) for i in range(len(classes)) if not (ap_list[i] is None or (isinstance(ap_list[i], float) and math.isnan(ap_list[i])))]
        ap_with_names.sort(key=lambda x: x[1], reverse=True)
        print("\nTop 10 classes by AP:")
        for class_name, ap in ap_with_names[:10]:
            print(f"  {class_name}: {ap:.4f}")
        print("\nBottom 10 classes by AP:")
        for class_name, ap in ap_with_names[-10:]:
            print(f"  {class_name}: {ap:.4f}")
        print("")

    # Gather top-1 score per file
    top1_scores = np.array([np.max(row["scores"]) for row in per_file])
    print("\nHistogram of top-1 predicted scores per file:")
    print_ascii_histogram(top1_scores, bins=10, width=40)

    # Print ASCII PR curve    
    print_ascii_pr_curve(y_true, y_scores, bins=20, width=40)

    # Optional CSV
    if args.save_csv:
        save_predictions_csv(per_file, classes, args.save_csv)
        print(f"Saved predictions to {args.save_csv}")


if __name__ == "__main__":
    main()