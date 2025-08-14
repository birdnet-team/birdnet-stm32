# birdnet-stm32

This repository contains code and resources for training a tiny audio classification model for bioacoustics. The model is designed to run on the [STM32N6570-DK development board](https://www.st.com/en/evaluation-tools/stm32n6570-dk.html#overview).

<img src="https://my.avnet.com/wcm/connect/c651fc2f-a5b2-489c-9d63-d3f064753690/STMicroelectronics+STM32N6570-DK.jpg?MOD=AJPERES&CACHEID=ROOTWORKSPACE-c651fc2f-a5b2-489c-9d63-d3f064753690-phBdXih" alt="STM32N6570-DK board" style="width: 100%;" />

(Image source: [EBV Electronik](https://my.avnet.com/ebv/products/new-products/npi/2024/stmicroelectronics-stm32n6570-dk))

STM32N6570-DK user manual: [DM00570145](https://www.st.com/resource/en/user_manual/dm00570145.pdf)

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

## Training

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
- `--audio_frontend`: Audio frontend to use (`librosa` or `tf`, default: `librosa`)
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

## Model conversion & validation

Run the `convert.py` script to convert the trained model to a fully quantized TensorFlow Lite model (with float32 inputs and outputs, as required for STM32 deployment). The script will:
- Load the trained Keras model from the specified path.
- Quantize the model for deployment (weights and activations are quantized, but input/output remain float32).
- Use a representative dataset (from your training data) for optimal quantization.
- Save the converted model as a `.tflite` file.
- Optionally validate the TFLite model after conversion.

**Example usage:**

```bash
python convert.py \
  --checkpoint_path checkpoints/birdnet_stm32_tiny.h5 \
  --data_path_train data/train
```

**Arguments:**

- `--checkpoint_path`: Path to the trained model checkpoint (reuired, should be a `.h5` file)
- `--output_path`: Path to save the converted model. If not provided, it will save to the same directory as the checkpoint
- `--data_path_train`: Path to your training data directory (used for representative dataset during quantization)
- `--num_samples`: Number of samples from the training data to use for quantization (default: `100`)
- `--num_mels`: Number of mel bins for spectrograms (should match training, default: `64`)
- `--spec_width`: Spectrogram width (should match training, default: `128`)
- `--chunk_duration`: Duration (seconds) of each audio chunk (should match training, default: `3`)
- `--validate`: Whether to validate the TFLite model after conversion (default: `True`)
- `--audio_frontend`: Audio frontend to use for spectrogram generation (`librosa` or `tf`, default: `librosa`)

Note:
- Match --audio_frontend, --num_mels, --spec_width, and --chunk_duration to training.
- Provide ≥ 512 diverse representative samples for better activation calibration and higher cosine similarity.
- If you don’t pass --data_path_train, random data is used (conversion works but accuracy/fidelity may suffer).

After conversion, the script will also run a quick validation to ensure the TFLite model has float32 input/output and can run inference.

### The STM deployment process

In order to deploy the model to the STM32N6570-DK, we will use STM's X-CUBE-AI framework, which provides tools for converting and deploying machine learning models on STM32 microcontrollers. The workflow involves several steps:

1. **Generate the model files** using the STM32Cube.AI CLI tool.
2. **Load the model onto the board** using the N6 loader script.
3. **Validate the model** on the STM32N6570-DK to ensure it works as expected.
 
<img width="100%" alt="stm32_model_validation" src="https://github.com/user-attachments/assets/3dddfbd5-4c87-4e3c-ac59-6291545188af" />

(Image source: [STM32ai](https://stm32ai-cs.st.com/assets/embedded-docs/stneuralart_getting_started.html))

### Generate the Model Files

First, we'll use the STM32Cube.AI CLI to convert the trained model to a format suitable for deployment on the STM32N6570-DK. You can download the STM32Cube.AI CLI from the [STMicroelectronics website](https://www.st.com/en/embedded-software/x-cube-ai.html#get-software).

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

Now, we need to run model conversion using the CLI tool. Make sure you have your trained and converted model saved as a `.tflite` file.

Navigate to `/path/to/X-CUBE-AI.10.2.0/Utilities/linux` and run the `stedgeai` command to generate the model files for STM32N6570-DK:

```bash
cd Utilities/linux
./stedgeai generate \
  --model  /path/to/birdnet-stm32/checkpoints/birdnet-stm32-tiny.tflite \
  --target stm32n6 \ 
  --st-neural-art \
  --output /path/to/birdnet-stm32/validation/st_ai_output \
  --workspace /path/to/birdnet-stm32/validation/st_ai_ws \
  --verbose
```

If you encounter the error `arm-none-eabi-gcc: error: unrecognized -mcpu target: cortex-m55`, it means you need to install the most recent ARM toolchain. You can do this by downloading the ARM toolchain from [ARM Developer](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads) or using a package manager:

```bash
wget https://developer.arm.com/-/media/Files/downloads/gnu/14.3.rel1/binrel/arm-gnu-toolchain-14.3.rel1-x86_64-arm-none-eabi.tar.xz
tar xf arm-gnu-toolchain-14.3.rel1-x86_64-arm-none-eabi.tar.xz
export PATH=$PWD/arm-gnu-toolchain-14.3.rel1-x86_64-arm-none-eabi/bin:$PATH
```

Make sure you have the correct `arm-none-eabi-gcc` compiler installed and available in your PATH. You can check this by running:

```bash
arm-none-eabi-gcc --version
```

Note: After conversion, the tool will generate a `network_generate_report.txt` in the output folder which you can consult to get some basic metrics on model computer requirements. If you run the command above with `analyze` instead of `generate`, it will analyze the model and provide more detailed information about its size, memory usage, and performance.

### Load the Model onto the STM32N6570-DK

Connect your STM32N6570-DK board to your computer and ensure it is recognized by running:

```bash
ls /dev/ttyACM*
```

If you see a device like `/dev/ttyACM0`, you can proceed with the validation.

**Install STM32CubeProgrammer**

Next, install STM32CubeProgrammer on your computer. You can download it from the [STMicroelectronics website](https://www.st.com/en/development-tools/stm32cubeprog.html). Unzip and run `./SetupSTM32CubeProgrammer-2.20.0.linux` to install it. This will launch a GUI installer. Follow the instructions to complete the installation.

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

**Install STM32CubeIDE**

If you haven't installed STM32CubeIDE yet, you can download it from the [STMicroelectronics website](https://www.st.com/en/development-tools/stm32cubeide.html). Unzip and run the installer with `./st-stm32cubeide_1.19.0_25607_20250703_0907_amd64.sh`. Follow the installation instructions for your platform.

**Setting paths for N6 loader script**

Create a `config_n6l.json` file and copy the lines below; change the paths to point to your generated `network.c`and the `NPU_Validation` project in the `X-CUBE-AI.10.2.0/Projects/STM32N6570-DK/Applications` directory.

```json
{	
  "network.c": "/path/to/birdnet-stm32/validation/st_ai_output/network.c",
  "project_path": "/path/to/Code/X-CUBE-AI.10.2.0/Projects/STM32N6570-DK/Applications/NPU_Validation",
  "project_build_conf": "N6-DK",
  "skip_external_flash_programming": false,
  "skip_ram_data_programming": false,
  "objcopy_binary_path": "/usr/bin/arm-none-eabi-objcopy"
}
```

Update the `config.json` in the `X-CUBE-AI.10.2.0/scripts/N6_scripts` directory to point to your STM32CubeIDE installation path:

```json
{
  "compiler_type": "gcc",
  "cubeide_path":"/path/to/stm32cubeide"
}
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

Navigate to the `validation` directory in this repo and run the `n6_loader.py` script from the X-CUBE-AI *scipt* directory and pass the `config_n6l.json` file as an argument:

```bash
python <path-to-install-dir>/X-CUBE-AI.10.2.0/scripts/N6_scripts/n6_loader.py --n6-loader-config /path/to/birdnet-stm32/config_n6l.json
```

If the build fails, check the `n6_loader.log` and `compile.log` files in the `validation` directory for errors. If you encounter issues, ensure that the paths in `config.json` and `config_n6l.json` are correct and that the necessary tools are installed.

If successful, ouputs should look like this:

```
XXX  __main__ -- Preparing compiler GCC
XXX  __main__ -- Setting a breakpoint in main.c at line 137 (before the infinite loop)
XXX  __main__ -- Copying network.c to project: -> /path/to/X-CUBE-AI.10.2.0/Projects/STM32N6570-DK/Applications/NPU_Validation/X-CUBE-AI/App/network.c
XXX  __main__ -- Extracting information from the c-file
XXX  __main__ -- Converting memory files in results/<model>/generation/ to Intel-hex with proper offsets
XXX  __main__ -- arm-none-eabi-objcopy --change-addresses 0x71000000 -Ibinary -Oihex network_atonbuf.xSPI2.raw network_atonbuf.xSPI2.hex
XXX  __main__ -- Resetting the board...
XXX  __main__ -- Flashing memory xSPI2 -- 1 659.665 kB
XXX  __main__ -- Building project (conf= N6-DK)
XXX  __main__ -- Loading internal memories & Running the program
XXX  __main__ -- Start operation achieved successfully
```

### Validate the Model on STM32N6570-DK

Now, we can finally validate the model on the STM32N6570-DK, you can use the `validate` command after navigating to the `X-CUBE-AI.10.2.0/Utilities/linux` directory:

```bash
./stedgeai validate \
  --model  /path/to/birdnet-stm32/checkpoints/birdnet-stm32-tiny.h5 \
  --target stm32n6 \ 
  --mode target \
  --desc serial:921600 \
  --output /path/to/birdnet-stm32/validation/st_ai_output \
  --workspace /path/to/birdnet-stm32/validation/st_ai_ws \
  --valinput /path/to/birdnet-stm32/checkpoints/birdnet-stm32-tiny_validation_data.npz \
  --classifier \
  --verbose
```

Make sure to pass the .h5 model file you trained earlier, the validation script will validate on-device outputs vs. the reference model.

You might have to run `sudo chmod a+rw /dev/ttyACM0` to give your user permission to access the serial port.

Note: STM provides a "Getting Started" guide for the STM32N6, which you can find [here](https://stm32ai-cs.st.com/assets/embedded-docs/stneuralart_getting_started.html) in case you need more detailed instructions on setting up the board and running the validation.

If everything is set up correctly, the validate command will run inference on the STM32N6570-DK and print the results to the console. After the validation is complete, you should see a `network_validate_report.txt` file in the `validation/st_ai_output` directory with the validation results.

For more command line options, visit the [ST Edge AI documentation](https://stm32ai-cs.st.com/assets/embedded-docs/command_line_interface.html).

## Build and deploy demo application

TODO :)

## License

 - **Source Code & models**: The source code and models for this project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
 - **STM tools and scripts**: The STM tools and scripts used in this project are licensed under different licenses, please refer to the respective documentation for details.
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




