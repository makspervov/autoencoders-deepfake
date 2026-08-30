# VAE Deepfake Detection Pipeline

This repository contains a modular Python pipeline for building, training, and evaluating a Variational Autoencoder (VAE) to detect deepfakes using anomaly detection techniques. The model is trained exclusively on pristine (real) faces to learn their latent distribution. During inference, deepfakes are identified by high reconstruction errors (MSE).

This implementation provides full execution support for CPU, NVIDIA CUDA, and Intel NPU devices (via OpenVINO).

## Prerequisites

- Python 3.10+
- (Optional) A [Hugging Face](https://huggingface.co/) account if downloading restricted datasets, although the default dataset is public.

## Installation

1. **Clone the repository and install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Pipeline Execution

The pipeline is designed to be executed sequentially through independent scripts.

### 1. Download Dataset

Download the DeepFakeFace dataset into the raw data folder from the Hugging Face Hub:

```bash
python download.py --dataset OpenRL/DeepFakeFace --path ./data/raw
```

### 2. Preprocessing (Face Extraction)

Deepfake detection relies on analyzing faces. Extract and crop faces from the downloaded dataset using MTCNN. This script processes the raw images and saves the cropped faces while maintaining the directory structure.

```bash
# Run on CPU
python preprocess.py --input_dir ./data/raw --output_dir ./data/processed --device cpu

# Run on NPU
python preprocess.py --input_dir ./data/raw --output_dir ./data/processed --device npu

# Or run on CUDA (if available) for faster processing
python preprocess.py --input_dir ./data/raw --output_dir ./data/processed --device cuda
```

### 3. Training the VAE

Train the model exclusively on the *real* (pristine) images. Adjust hyperparameters like `--epochs`, `--batch_size`, and `--lr` as needed based on your hardware constraints.

*Ensure your real images are located in the designated `real_dir` path.*

```bash
python train_<ae_type>.py \
    --real_dir ./data/processed/real \
    --epochs 50 \
    --batch_size 64 \
    --device cuda \
    --save_path ./models/vae_model.pth
```

### 4. Evaluation and Inference

Evaluate the trained VAE on a mixed dataset containing both real and fake images. This script calculates the optimal threshold for classifying images as deepfakes, calculates the AUC score, generates ROC curves, plots the MSE distribution, and visualizes reconstruction differences.

**Important:** You can select your processing backend here. Selecting `--device npu` will automatically trigger a model export to ONNX and OpenVINO IR, compiling it for Intel Neural Processing Units (NPUs) before running inference.

```bash
# Evaluate using standard PyTorch CPU
python evaluate_<ae_type>.py --real_dir ./data/processed/real --fake_dir ./data/processed/fake --device cpu

# Evaluate using NVIDIA GPU
python evaluate_<ae_type>.py --real_dir ./data/processed/real --fake_dir ./data/processed/fake --device cuda

# Evaluate using Intel NPU via OpenVINO
python evaluate_<ae_type>.py --real_dir ./data/processed/real --fake_dir ./data/processed/fake --device npu
```

Outputs, including ROC curve plots and visual grids, will be saved into the `./results` directory.
