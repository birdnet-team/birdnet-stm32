# iNat-tiny

This repository contains code and resources for training a tiny audio classification model for bioacoustics. The model is designed to run on the [STM32N6570-DK development board](https://www.st.com/en/evaluation-tools/stm32n6570-dk.html#overview).

## Setup (Ubuntu)

Clone this repository and navigate to the project directory:

```bash
git clone https://github.com/kahst/iNat-tiny.git
cd iNat-tiny
```

We assume you have Python 3.12 installed. If not, you can install it using:

```bash
sudo apt install python3.12 python3.12-venv python3.12-dev
``` 
Then, create a virtual environment and activate it:

```bash
python3.12 -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Deplyoing a model to the STM32N6570-DK is quite involved an requires us to:

- prepare dataset
- train a model
- convert the model
- deploy the model to the STM32N6570-DK

### Download and Prepare the Dataset

We'll use a subset of the iNatSounds dataset, which is available here: [iNatSounds on GitHub](https://github.com/visipedia/inat_sounds/tree/main/2024)

After download, we'll sort files into folders with folder names as labels (i.e, species names) based on the train and test annotations. Therefore, this repo assumes that your data is structured as follows:

```
data/
├── train/
│   ├── species1/
│   ├── species2/
│   └── ...
└── test/
    ├── species1/
    ├── species2/
    └── ... 
```

Each folder contains `.wav` files of the respective species. Since we're training a tiny model, we won't be able to fit all iNatSounds classes into our model. Thus, we will use a subset of species. However, it's your decision which species to use.

### Train the Model

To train the model, run `train.py` with the desired arguments. The script will:

- Split your training data into train and validation sets (using `--val_split`).
- Split audio files into fixed-length chunks.
- Generate mel spectrograms from the audio chunks.
- Build a compact CNN model for audio spectrogram classification.
- Optionally apply mixup augmentation to the training data.
- Enable quantization-aware training for deployment on embedded hardware.
- Train the model with early stopping, learning rate scheduling, and checkpointing.
- Save the best model to the specified path.

**Example usage:**
```bash
python train.py --data_path_train path/to/my/data --val_split 0.2 --checkpoint_path checkpoints/my_tiny_model.h5
```

**Arguments:**
- `--data_path_train`: Path to your training dataset (default: `/data/train`)
- `--num_mels`: Number of mel bins for spectrograms (default: `64`)
- `--spec_width`: Spectrogram width (frames) (default: `128`)
- `--chunk_duration`: Duration (seconds) of each audio chunk (default: `3`)
- `--max_duration`: Maximum duration (seconds) per audio file (default: `30`)
- `--embeddings_size`: Size of the final embeddings layer (default: `256`)
- `--alpha`: Model width scaling factor (default: `1.0`)
- `--depth_multiplier`: Number of block repetitions in the model (default: `2`)
- `--mixup_alpha`: Beta parameter for mixup augmentation (set to `0` to disable, default: `0.2`)
- `--mixup_probability`: Fraction of batch to apply mixup (set to `0` to disable, default: `0.25`)
- `--batch_size`: Batch size for training (default: `64`)
- `--epochs`: Number of training epochs (default: `50`)
- `--learning_rate`: Initial learning rate (default: `0.001`)
- `--patience`: Early stopping patience (default: `5`)
- `--val_split`: Fraction of training data used for validation (default: `0.2`)
- `--checkpoint_path`: Path to save the best model.

The script will print progress and save the best model.

Note: The conversion script expects a `.h5` model file, so ensure you specify the correct `--checkpoint_path`.



