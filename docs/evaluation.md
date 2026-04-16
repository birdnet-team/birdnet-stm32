# Evaluation

Evaluate a trained or quantized model on a test dataset.

## Basic usage

```bash
python test.py \
  --model_path checkpoints/my_model_quantized.tflite \
  --model_config checkpoints/my_model_model_config.json \
  --data_path_test data/test \
  --pooling lme
```

The script:

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

The script also prints the top-10 and bottom-10 classes ranked by average
precision.

## Arguments

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

## Saving results

Use `--save_csv` to export per-file predictions. Evaluation run CSVs are stored
in `report/eval_runs/` with the naming convention:

```
{run_number}_{frontend}_{mag}_{alpha}_{depth}_{embed}_{batch}_{maxsamples}.csv
```

## Confusion matrix

Use `--confusion_matrix` to print an ASCII confusion matrix to stdout. Use
`--save_cm_plot path/to/plot.png` to save a matplotlib figure.

## Threshold optimization

By default, evaluation uses a fixed threshold of 0.5. Use `--optimize_thresholds`
to find the per-class threshold that maximizes F1 via the precision-recall curve.
Optimal thresholds are printed sorted by value.
