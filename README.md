# birdnet-stm32

This repository contains code and resources for training a tiny audio classification model for bioacoustics. The model is designed to run on the [STM32N6570-DK development board](https://www.st.com/en/evaluation-tools/stm32n6570-dk.html#overview).

<img src="https://www.st.com/bin/ecommerce/api/image.PF275430.en.feature-description-include-personalized-no-cpn-large.jpg" alt="STM32N6570-DK board" style="width: 100%;" />

(Image source: [STMicroelectronics](https://www.st.com/en/evaluation-tools/stm32n6570-dk.html))

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
  --type keras \
  --target stm32n6 \ 
  --st-neural-art \ (only for tflite)
  --output validation/st_ai_output \
  --verbose
```

Note: After conversion, the tool will generate a `network_generate_report.txt` in the output folder which you can consult to get some basic metrics on model computer requirements. If you run the command above with `analyze` instead of `generate`, it will analyze the model and provide more detailed information about its size, memory usage, and performance.

### Validate the Model on STM32N6570-DK

Connect your STM32N6570-DK board to your computer and ensure it is recognized by running:

```bash
ls /dev/ttyACM*
```

If you see a device like `/dev/ttyUSB0`, you can proceed with the validation.

First, install STM32CubeProgrammer on your computer. You can download it from the [STMicroelectronics website](https://www.st.com/en/development-tools/stm32cubeprog.html). Unzip and run `./SetupSTM32CubeProgrammer-2.20.0.linux` to install it. This will launch a GUI installer. Follow the instructions to complete the installation.

Verify the installation by navigating to the installation directory and running the command:

```bash
<path-to-install-dir>/STM32Cube/STM32CubeProgrammer/bin/STM32_Programmer_CLI --version
```

Now, add the STM32CubeProgrammer CLI to your PATH:

```bash
export PATH=$PATH:/<path-to-install-dir>/STM32Cube/STM32CubeProgrammer/bin
```

Add your user to the plugdev and dialout groups:

```bash
sudo usermod -aG plugdev $USER
sudo usermod -aG dialout $USER
```

Install STMicroelectronics udev rules: If you haven't already, copy the rules file:

```bash
sudo cd <path-to-install-dir>/STM32Cube/STM32CubeProgrammer/Drivers/rules/
sudo cp *.* /etc/udev/rules.d
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Unplug and replug your STM32N6570-DK board to apply the new rules, reboot your computer, or log out and log back in.

Check if the board is connected and recognized by the STM32CubeProgrammer CLI:

```bash
STM32_Programmer_CLI --list
```

If everything is set up correctly, you should see your STM32N6570-DK board listed.

Now, we need to flash the pre-built firmware to the STM32N6570-DK board so the validate command can communicate with it. 
Make sure you ran the `generate` command above to generate the model files, which will be used by the `validate` command - generated files are in `X-CUBE-AI.10.2.0/Utilities/linux/st_ai_output`. 

**Install STM32CubeIDE**

If you haven't installed STM32CubeIDE yet, you can download it from the [STMicroelectronics website](https://www.st.com/en/development-tools/stm32cubeide.html). Unzip and run the installer with `./st-stm32cubeide_1.19.0_25607_20250703_0907_amd64.sh`. Follow the installation instructions for your platform.

**Setting paths for N6 loader script**

We need to set the paths to our genereated files in `X-CUBE-AI.10.2.0/scripts/N6_scripts/config_n6l.json`; comment out the `project_path` and `project_build_conf` lines if you are not using the NPU_Validation project, and set the `network.c` path to the generated `network.c` file.

```json
{
	// The 2lines below are _only used if you call n6_loader.py ALONE (memdump is optional and will be the parent dir of network.c by default)
	"network.c": "<path-to-install-dir>/X-CUBE-AI.10.2.0/Utilities/linux/st_ai_output/network.c",
	//"memdump_path": "C:/Users/foobar/CODE/stm.ai/stm32ai_output",
	// Location of the "validation" project  + build config name to be built (if applicable)
	// "project_path": "<path-to-install-dir>/X-CUBE-AI.10.2.0/Projects/STM32N6570-DK/Applications/NPU_Validation/",
	// If using the NPU_Validation project, valid build_conf names are "N6-DK", "N6-DK-USB", "N6-Nucleo", "N6-Nucleo-USB"
	// "project_build_conf": "N6-DK",
	// Skip programming weights to earn time (but lose accuracy) -- useful for performance tests
	"skip_external_flash_programming": false,
	"skip_ram_data_programming": false,
	"objcopy_binary_path": "/usr/bin/arm-none-eabi-objcopy"
}
```

Update your `config.json` for Linux and GCC as follows (replace all Windows-style paths (`C:/...`) with Linux-style paths (`/home/...` or `/opt/...`) based on your installation):

```json
{
    // Set Compiler_type to either gcc or iar
    "compiler_type": "gcc",
    // Path to gdbserver directory (ends up in bin/)
    "gdb_server_path": "<path-to-install-dir>/stm32cubeide/plugins/com.st.stm32cube.ide.mcu.externaltools.stlink-gdb-server.linux64_2.2.200.202505060755/tools/bin/",
    // Path to gcc directory (ends up in bin/)
    "gcc_binary_path": "<path-to-install-dir>/stm32cubeide/plugins/com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.13.3.rel1.linux64_1.0.0.202410170706/tools/bin/",
    // Path to IAR directory (ends up in bin/) (leave empty if not using IAR)
    "iar_binary_path": "",
    // Full path to arm-none-eabi-objcopy program
    "objcopy_binary_path": "/usr/bin/arm-none-eabi-objcopy",
    // Full path to STM32_Programmer_CLI program
    "cubeProgrammerCLI_binary_path": "<path-to-install-dir>/STM32Cube/STM32CubeProgrammer/bin/STM32_Programmer_CLI",
    // Path to CubeIDE directory (ends up in stm32cubeide)
    "cubeide_path": "<path-to-install-dir>/stm32cubeide"
}
```

**Notes:**
- Adjust `stm32cubeide/` and `stm32cubeprogrammer/` directories to match your actual installation paths.
- Use `which arm-none-eabi-objcopy` to confirm the objcopy path.
- Use `which STM32_Programmer_CLI` if you added it to your PATH, or specify the full path.
- Install the ARM toolchain if you haven't already with `sudo apt-get install gcc-arm-none-eabi`

**Example commands to find paths:**
```bash
which arm-none-eabi-objcopy
which STM32_Programmer_CLI
```

**Set board to DEV mode**

 - disconnect the board from USB
 - set BOOT0 to right
 - set BOOT1 to left
 - set JP2 to position 1-2
 - reconnect the board to USB

See the image below for reference:

![Set STM32N6570-DK to dev mode](https://community.st.com/t5/image/serverpage/image-id/108308iD2AD18EF06920D91/image-size/large/is-moderation-mode/true?v=v2&px=999)

(Image source: [ST Community](https://community.st.com/t5/stm32-mcus/how-to-debug-stm32n6-using-stm32cubeide/ta-p/800547))

**Running n6_loader.py**

Navigate to the `validation` directory in this repo and run the `n6_loader.py` script from the X-CUBE-AI *scipt* directory:

```bash
python <path-to-install-dir>/X-CUBE-AI.10.2.0/scripts/N6_scripts/n6_loader.py
```

If the build fails, check the `n6_loader.log` and `compile.log` files in the `validation` directory for errors. If you encounter issues, ensure that the paths in `config.json` and `config_n6l.json` are correct and that the necessary tools are installed.

**Run the validation command**

To validate the model on the STM32N6570-DK, you can use the `validate` command:

```bash
./stedgeai validate \
  --model /path/to/best_model.h5 \
  --type keras \
  --target STM32N6570-DK \
  --mode target \
  --verbose
```

Note: STM provides a "Getting Started" guide for the STM32N6, which you can find [here](https://stm32ai-cs.st.com/assets/embedded-docs/stneuralart_getting_started.html) in case you need more detailed instructions on setting up the board and running the validation.

### Build and deploy demo application

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





