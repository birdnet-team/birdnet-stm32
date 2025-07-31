# birdnet-stm32

This repository contains code and resources for training a tiny audio classification model for bioacoustics. The model is designed to run on the [STM32N6570-DK development board](https://www.st.com/en/evaluation-tools/stm32n6570-dk.html#overview).

## Setup (Ubuntu)

Clone this repository and navigate to the project directory:

```bash
git clone https://github.com/kahst/birdnet-stm32.git
cd birdnet-stm32
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

Deploying a model to the STM32N6570-DK is quite involved and requires us to:

- download and prepare the dataset
- train a model
- convert the model
- deploy the model to the STM32N6570-DK

### Download and Prepare the Dataset

We'll use a subset of the iNatSounds dataset, which is available here: [iNatSounds on GitHub](https://github.com/visipedia/inat_sounds/tree/main/2024)

After downloading, sort files into folders with folder names as labels (i.e., species names) based on the train and test annotations. Therefore, this repo assumes that your data is structured as follows:

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

This repo comes with a pre-trained checkpoint (`checkpoints/birdnet_stm32_tiny.h5`) that you can use to test the model conversion and deployment process. To train a custom model, run `train.py` with the desired arguments. 

The script will:

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

### Model conversion

We'll use the STM32Cube.AI CLI to convert the trained model to a format suitable for deployment on the STM32N6570-DK. You can download the STM32Cube.AI CLI from the [STMicroelectronics website](https://www.st.com/en/embedded-software/x-cube-ai.html#get-software).

After downloading, you should have `x-cube-ai-linux-v10.2.0.zip`, unzip and locate the CLI tool which is typically found in the `Utilities/linux` directory.

```bash
unzip x-cube-ai-linux-v10.2.0.zip X-CUBE-AI.10.2.0
cd X-CUBE-AI.10.2.0
unzip stedgeai-linux-10.2.0.zip
```

This should be your directory structure after unzipping both zips:

```
X-CUBE-AI.10.2.0/
├── STMicroelectronics.X-CUBE-AI.10.2.0.pack
├── Utilities/
│   ├── linux/
│   │   └── stedgeai  <-- CLI tool
│   |   └── ...
│   ├── windows/
│   └── ...
└── Middlewares/
└── ...
```

Now, we need to run model conversion using the CLI tool. Make sure you have your trained model saved as a `.h5` file (e.g., `checkpoints/my_tiny_model.h5`).

```bash
cd Utilities/linux
./stedgeai generate \
  --model /path/to/best_model.h5 \
  --name tiny_birdnet \
  --type keras \
  --target STM32N6570-DK \
  --verbose
```

Note: After conversion, the tool will generate a `tiny_birdnet_generate_report.txt` in the output folder which you can consult to get some basic metrics on model computer requirements. If you run the command above with `analyze` instead of `generate`, it will analyze the model and provide more detailed information about its size, memory usage, and performance.

To validate the model on the STM32N6570-DK, you can use the `validate` command:

```bash
./stedgeai validate \
  --model /path/to/best_model.h5 \
  --name tiny_birdnet \
  --type keras \
  --target STM32N6570-DK \
  --mode target \
  --verbose
```

### Model deployment

TODO :)

## License

 - **Source Code & models**: The source code and models for this project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

 - **Citation**: Feel free to use the code or models in your research. If you do, please cite as:

```bibtex
@article{kahl2021birdnet,
  title={BirdNET: A deep learning solution for avian diversity monitoring},
  author={Kahl, Stefan and Wood, Connor M and Eibl, Maximilian and Klinck, Holger},
  journal={Ecological Informatics},
  volume={61},
  pages={101236},
  year={2021},
  publisher={Elsevier}
}
```

## Funding

This project is supported by Jake Holshuh (Cornell class of ´69) and The Arthur Vining Davis Foundations.
Our work in the K. Lisa Yang Center for Conservation Bioacoustics is made possible by the generosity of K. Lisa Yang to advance innovative conservation technologies to inspire and inform the conservation of wildlife and habitats.

The development of BirdNET is supported by the German Federal Ministry of Education and Research through the project “BirdNET+” (FKZ 01|S22072).
The German Federal Ministry for the Environment, Nature Conservation and Nuclear Safety contributes through the “DeepBirdDetect” project (FKZ 67KI31040E).
In addition, the Deutsche Bundesstiftung Umwelt supports BirdNET through the project “RangerSound” (project 39263/01).

## Partners

BirdNET is a joint effort of partners from academia and industry.
Without these partnerships, this project would not have been possible.
Thank you!

![Logos of all partners](https://tuc.cloud/index.php/s/KSdWfX5CnSRpRgQ/download/box_logos.png)





