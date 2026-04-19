# Evaluation

Evaluate a trained or quantized model on a test dataset.

## Basic usage

```bash
python -m birdnet_stm32 evaluate \
  --model_path checkpoints/my_model_quantized.tflite \
  --model_config checkpoints/my_model_model_config.json \
  --data_path_test data/test \
  --pooling lme
```

The command:

1. Loads a `.keras` or `.tflite` model.
2. Reads `_model_config.json` for frontend and chunking parameters.
3. Splits each test file into non-overlapping chunks (up to `--max_duration`).
4. Runs batched inference on all chunks.
5. Pools chunk-level scores to file-level predictions.
6. Reports metrics and per-class statistics.

## Pooling methods

Chunk scores are aggregated to file-level predictions using one of:

| Method | Formula | Use case |
|---|---|---|
| `avg` | Arithmetic mean | Balanced baseline |
| `max` | Element-wise maximum | Good when target is present in few chunks |
| `lme` | $\frac{1}{\beta} \log \left( \frac{1}{N} \sum_{i=1}^{N} e^{\beta \cdot s_i} \right)$ | Best overall — smoothly interpolates between avg and max |

LME (log-mean-exponential) uses a fixed $\beta = 10$.

## Metrics

| Metric | Description |
|---|---|
| ROC-AUC (micro) | Area under receiver operating characteristic, averaged over all class decisions |
| cmAP | Class-macro average precision — mean AP over classes that have positive examples |
| mAP | Micro average precision over all decisions |
| Precision | At threshold 0.5, file-level |
| Recall | At threshold 0.5, file-level |
| F1 | Harmonic mean of precision and recall at threshold 0.5 |

The command also prints the top-10 and bottom-10 classes ranked by average
precision.

## Confusion matrix

Use `--confusion_matrix` to print an ASCII confusion matrix to stdout. Use
`--save_cm_plot path/to/plot.png` to save a matplotlib figure.

## Threshold optimization

By default, evaluation uses a fixed threshold of 0.5. Use `--optimize_thresholds`
to find the per-class threshold that maximizes F1 via the precision-recall curve.
Optimal thresholds are printed sorted by value.

## Species-level AP report

Use `--species_report path/to/species.csv` to save a per-species average
precision report with bootstrap confidence intervals. The CSV includes columns:

| Column | Description |
|---|---|
| `class` | Species name |
| `ap` | Point-estimate average precision |
| `ci_lower` | 95% confidence interval lower bound |
| `ci_upper` | 95% confidence interval upper bound |
| `n_positive` | Number of positive test files for this class |
| `n_total` | Total number of test files |

Control the number of bootstrap resamples with `--n_bootstrap` (default 1000).
Higher values produce tighter CI estimates but take longer.

```bash
python -m birdnet_stm32 evaluate \
  --model_path checkpoints/my_model_quantized.tflite \
  --model_config checkpoints/my_model_model_config.json \
  --data_path_test data/test \
  --species_report report/species_ap.csv \
  --n_bootstrap 2000
```

## DET curve

The Detection Error Tradeoff (DET) curve plots false rejection rate (FRR)
against false acceptance rate (FAR) across thresholds — a standard metric in
bioacoustics evaluation.

- `--det_curve` — print an ASCII DET curve to stdout
- `--save_det_plot path/to/det.png` — save a matplotlib DET curve image

```bash
python -m birdnet_stm32 evaluate \
  --model_path checkpoints/my_model_quantized.tflite \
  --model_config checkpoints/my_model_model_config.json \
  --data_path_test data/test \
  --det_curve --save_det_plot report/det_curve.png
```

## Latency measurement

Use `--benchmark_latency` to measure per-chunk inference time. When enabled,
the evaluation loop wraps each `model.predict()` call with high-resolution
timing. The following statistics are added to the metrics output:

| Metric | Description |
|---|---|
| `latency_mean_ms` | Mean inference time per chunk (ms) |
| `latency_median_ms` | Median inference time per chunk (ms) |
| `latency_p95_ms` | 95th percentile latency (ms) |
| `latency_p99_ms` | 99th percentile latency (ms) |
| `total_chunks` | Total number of chunks processed |

!!! note "Host timing"
    Latency is measured on the host CPU/GPU, not on-device. For on-device
    latency, use `stedgeai validate` (see [Deployment](deployment.md)).

## Benchmark mode

Use `--benchmark path/to/benchmark.json` to save a structured JSON report
containing all metrics, per-species AP with CIs, model config, and latency
stats. This is designed for experiment tracking and automated comparison.

The JSON report contains:

```json
{
  "model_path": "checkpoints/my_model_quantized.tflite",
  "num_classes": 10,
  "num_files": 499,
  "metrics": {
    "roc-auc": 0.8521,
    "cmAP": 0.7834,
    "f1": 0.6912,
    "latency_mean_ms": 12.3,
    "latency_p95_ms": 14.1
  },
  "species": [ ... ],
  "config": { ... }
}
```

To include latency stats in the benchmark, combine with `--benchmark_latency`:

```bash
python -m birdnet_stm32 evaluate \
  --model_path checkpoints/my_model_quantized.tflite \
  --model_config checkpoints/my_model_model_config.json \
  --data_path_test data/test --pooling lme \
  --benchmark report/benchmark.json --benchmark_latency \
  --species_report report/species_ap.csv
```

## HTML report

Use `--report_html path/to/report.html` to generate a self-contained HTML
evaluation report. The report includes:

- Summary metrics table
- Per-species average precision table (if `--species_report` or `--benchmark`
  computes species data)
- Confusion matrix heatmap (uses base64-embedded matplotlib image)
- Inline CSS styling — no external dependencies needed to view

```bash
python -m birdnet_stm32 evaluate \
  --model_path checkpoints/my_model_quantized.tflite \
  --model_config checkpoints/my_model_model_config.json \
  --data_path_test data/test --pooling lme \
  --report_html report/eval_report.html
```

## Saving results

Use `--save_csv` to export per-file predictions. Evaluation run CSVs are stored
in `report/eval_runs/` with the naming convention:

```
{run_number}_{frontend}_{mag}_{alpha}_{depth}_{embed}_{batch}_{maxsamples}.csv
```

## Full argument reference

| Argument | Default | Description |
|---|---|---|
| `--model_path` | *(required)* | Path to `.keras` or `.tflite` model |
| `--model_config` | *(inferred)* | Path to `_model_config.json` |
| `--data_path_test` | *(required)* | Test data root with class subfolders |
| `--max_files` | -1 (all) | Max files per class |
| `--batch_size` | 16 | Chunk inference batch size |
| `--pooling` | avg | `avg`, `max`, or `lme` |
| `--overlap` | 0 | Chunk overlap in seconds |
| `--save_csv` | None | Path to save per-file predictions as CSV |
| `--confusion_matrix` | False | Print ASCII confusion matrix |
| `--save_cm_plot` | None | Save confusion matrix plot to image file |
| `--optimize_thresholds` | False | Find per-class optimal F1 thresholds |
| `--benchmark` | None | Save structured JSON benchmark report to this path |
| `--benchmark_latency` | False | Measure per-chunk inference latency (mean, median, p95, p99) |
| `--species_report` | None | Save per-species AP report with 95% bootstrap CI to CSV |
| `--n_bootstrap` | 1000 | Number of bootstrap resamples for CI estimation |
| `--det_curve` | False | Print ASCII DET curve |
| `--save_det_plot` | None | Save DET curve plot to image file |
| `--report_html` | None | Generate a self-contained HTML evaluation report |

## Full evaluation example

Run a comprehensive evaluation with all reporting options:

```bash
python -m birdnet_stm32 evaluate \
  --model_path checkpoints/my_model_quantized.tflite \
  --model_config checkpoints/my_model_model_config.json \
  --data_path_test data/test \
  --pooling lme \
  --confusion_matrix --save_cm_plot report/confusion_matrix.png \
  --optimize_thresholds \
  --benchmark report/benchmark.json --benchmark_latency \
  --species_report report/species_ap.csv --n_bootstrap 2000 \
  --det_curve --save_det_plot report/det_curve.png \
  --report_html report/eval_report.html \
  --save_csv report/predictions.csv
```
