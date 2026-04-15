# Dataset Preparation

## Folder structure

BirdNET-STM32 expects audio files organized by species:

```
data/
├── train/
│   ├── species_name_1/
│   │   ├── file1.wav
│   │   ├── file2.wav
│   │   └── ...
│   ├── species_name_2/
│   └── ...
└── test/
    ├── species_name_1/
    ├── species_name_2/
    └── ...
```

Each subfolder name becomes a class label. All audio files must be `.wav` format.

## Special class names

Folders named `noise`, `silence`, `background`, or `other` are treated as
**negative classes** — they receive all-zero label vectors during training. Use
these to improve robustness against non-bird audio.

## Downloading iNatSounds

We use a subset of the [iNatSounds 2024](https://github.com/visipedia/inat_sounds/tree/main/2024)
dataset. After downloading, sort files into species folders based on the train
and test annotation CSVs.

Since the model is small, you typically train on a subset of species. Species
lists for various regions are available in `dev/`:

| File | Region |
|---|---|
| `species_list_eu.txt` | Central Europe |
| `species_list_CA.txt` | California |
| `species_list_USE.txt` | Eastern US |
| `species_list_USW.txt` | Western US |
| `species_list_brazil.txt` | Brazil |
| `species_list_sea.txt` | Southeast Asia |
| `species_list_australia.txt` | Australia |
| `species_list_africa.txt` | Sub-Saharan Africa |
| `species_list_combined.txt` | Combined subset |

## Data pipeline details

During training, the data pipeline:

1. **Discovers** all `.wav` files under `data/train/<class>/`.
2. **Upsamples** minority classes to a configurable ratio (`--upsample_ratio`,
   default 0.5) of the largest class.
3. **Caps** files per class if `--max_samples` is set.
4. **Chunks** each file into fixed-length segments (`--chunk_duration`,
   default 3 seconds) up to `--max_duration` (default 30 seconds).
5. **Computes spectrograms** according to the selected `--audio_frontend`.
6. **Splits** into train/validation (`--val_split`, default 0.2).

## Tips

- Aim for at least 50–100 files per species for reasonable training.
- Longer files contribute more chunks — balance file counts, not total
  duration.
- Add noise/background folders to make the model more robust in the field.
- The `--max_samples` flag is useful for quick experiments with balanced
  class counts.
