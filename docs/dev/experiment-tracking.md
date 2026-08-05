# Experiment Tracking

Conventions for tracking training runs, evaluation results, and model
comparisons in BirdNET-STM32.

## Checkpoint naming

Training produces a set of files with a shared name prefix:

```
checkpoints/
├── {name}.keras                        # Trained model
├── {name}_model_config.json            # Model configuration
├── {name}_labels.txt                   # Class labels (one per line)
├── {name}_history.csv                  # Training loss/metrics per epoch
├── {name}_curves.png                   # Training curves plot
├── {name}_train_state.json             # LR, epoch, optimizer state metadata
├── {name}_quantized.tflite             # Quantized TFLite model
└── {name}_quantized_validation_data.npz  # Validation I/O for on-device comparison
```

The default name is `best_model`. Use `--checkpoint_path` to give each
experiment a descriptive internal name.

Public release names are separate from experiment names. Use
`BirdNET_Tiny_N6_<REGION>_<SPECIES_COUNT>_V<MAJOR.MINOR>` for release assets;
for example, `BirdNET_Tiny_N6_USNE_30_V1.0`. The species count covers bird
species only and excludes nuisance/background outputs. Precision-bearing
release files append `_FP32`, `_FP16`, or `_INT8` to that family basename.

## Evaluation run naming

Evaluation results are stored in `report/eval_runs/` with the naming convention:

```
{run_number}_{frontend}_{mag}_{alpha}_{depth}_{embed}_{batch}_{maxsamples}.csv
```

For example: `001_hybrid_pwl_1.0_1_256_32_500.csv`

This encodes the key hyperparameters in the filename for easy comparison without
opening each file.

## Benchmark JSON

Use `--benchmark` during evaluation to produce a structured JSON report:

```bash
python -m birdnet_stm32 evaluate \
  --model_path checkpoints/best_model_quantized.tflite \
  --model_config checkpoints/best_model_model_config.json \
  --data_path_test data/test --benchmark
```

The benchmark JSON includes:

- Model metadata (frontend, alpha, depth, mag_scale, num_classes)
- All metrics (ROC-AUC, cmAP, mAP, F1, precision, recall)
- Per-species AP (if `--species_report` is also set)
- Latency statistics (if `--benchmark_latency` is also set)
- Timestamp and model file hash for reproducibility

## Comparing experiments

### Manual comparison

Compare CSV files in `report/eval_runs/`:

```bash
# List all runs sorted by cmAP
head -1 report/eval_runs/001_*.csv  # Header
for f in report/eval_runs/*.csv; do
  tail -1 "$f" | awk -F, '{printf "%-60s cmAP=%.4f\n", FILENAME, $3}'
done | sort -t= -k2 -rn
```

### Optuna comparison

When using `--tune`, Optuna stores trial results in
`checkpoints/optuna_trials/` and the best hyperparameters in
`checkpoints/optuna_best_params.json`.

## Tips

- **Name experiments descriptively**: use `--checkpoint_path` to encode the
  experiment variant (for example,
  `checkpoints/hybrid_pwl_a1.0_d2_se.keras`).
- **Keep training data fixed**: changing the dataset between experiments
  invalidates comparisons. Use `--max_samples` to cap training data size
  for controlled experiments.
- **Use `--benchmark`**: structured JSON is easier to parse programmatically
  than CSV files.
- **Record the git commit**: note which commit each experiment was run on to
  enable reproduction.
