import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
from tqdm import tqdm
import openvino as ov

from dataset import get_dataloaders
from model_dae import DAE


def export_to_openvino(
    model: torch.nn.Module, dummy_input: torch.Tensor, onnx_path: str, ir_dir: str
):
    """
    Exports a PyTorch model to ONNX, then converts to OpenVINO Intermediate Representation (IR).
    """
    print("Exporting PyTorch model to ONNX...")
    model.eval()

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["recon"],
        dynamic_axes={"input": {0: "batch_size"}},
    )
    print(f"Model exported to {onnx_path}")

    print("Converting ONNX to OpenVINO IR...")
    # Convert ONNX to OpenVINO model
    ov_model = ov.convert_model(onnx_path)

    # Save the IR model
    os.makedirs(ir_dir, exist_ok=True)
    ir_path = os.path.join(ir_dir, "dae.xml")
    ov.save_model(ov_model, ir_path)
    print(f"OpenVINO IR saved to {ir_path}")

    return ov_model


def get_reconstruction_errors(
    model, dataloader, device, is_openvino=False, ov_compiled_model=None
):
    """
    Calculates reconstruction errors (MSE) for all images in the dataloader.
    Returns lists of errors, labels, original images, and reconstructed images.
    """
    errors = []
    labels = []
    originals = []
    reconstructions = []

    if not is_openvino:
        model.eval()

    with torch.no_grad():
        for data, label in tqdm(dataloader, desc="Evaluating"):

            # Keep a copy of original for plotting (first image in batch)
            orig_img = data[0].cpu().numpy().transpose(1, 2, 0)

            if is_openvino:
                # OpenVINO inference
                data_np = data.numpy()
                # OpenVINO outputs a dictionary, get the first output (reconstruction)
                result = ov_compiled_model([data_np])
                recon_batch = result[ov_compiled_model.output(0)]
                recon_batch = torch.tensor(recon_batch)
            else:
                # PyTorch inference
                data = data.to(device)
                # DAE outputs only recon_batch
                recon_batch = model(data)
                recon_batch = recon_batch.cpu()
                data = data.cpu()

            # Calculate MSE for each image in batch independently
            # Batch shape is (B, C, H, W). Compute MSE across C, H, W for each item.
            # Using reduction='none' then mean over dim [1, 2, 3]
            mse = F.mse_loss(recon_batch, data, reduction="none")
            mse_per_image = mse.view(mse.size(0), -1).mean(dim=1).numpy()

            errors.extend(mse_per_image)
            labels.extend(label.numpy())

            # Save first image of batch for visualization
            recon_img = recon_batch[0].numpy().transpose(1, 2, 0)
            originals.append(orig_img)
            reconstructions.append(recon_img)

    return np.array(errors), np.array(labels), originals, reconstructions


def evaluate(
    real_dir: str,
    fake_dir: str,
    model_path: str,
    device: str,
    batch_size: int,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    # Check device availability
    is_openvino = False
    ov_compiled_model = None

    if device == "npu":
        print("NPU requested. Will use OpenVINO.")
        is_openvino = True
        torch_device = torch.device("cpu")
    elif device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        torch_device = torch.device("cpu")
    else:
        torch_device = torch.device(device)

    print("Loading model...")
    model = DAE()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch_device))
    else:
        print(
            f"Warning: Model not found at {model_path}. Using random weights for testing."
        )

    model.to(torch_device)

    if is_openvino:
        # Initialize OpenVINO Core
        core = ov.Core()
        available_devices = core.available_devices
        print(f"Available OpenVINO devices: {available_devices}")

        target_device = "NPU" if "NPU" in available_devices else "CPU"
        if target_device == "CPU":
            print(
                "Warning: NPU not found in OpenVINO available devices. Falling back to OpenVINO CPU."
            )

        dummy_input = torch.randn(1, 3, 128, 128)
        onnx_path = os.path.join(output_dir, "dae.onnx")
        ir_dir = os.path.join(output_dir, "ir")

        ov_model = export_to_openvino(model, dummy_input, onnx_path, ir_dir)
        print(f"Compiling OpenVINO model for {target_device}...")
        ov_compiled_model = core.compile_model(ov_model, target_device)

    print("Initializing DataLoaders...")
    _, real_loader, fake_loader = get_dataloaders(
        real_dir=real_dir, fake_dir=fake_dir, batch_size=batch_size, num_workers=2
    )

    if real_loader is None or fake_loader is None:
        print("Error: Dataloaders are empty. Make sure you have processed data.")
        return

    print("\nProcessing Real Images...")
    real_errors, real_labels, real_orig, real_recon = get_reconstruction_errors(
        model, real_loader, torch_device, is_openvino, ov_compiled_model
    )

    print("\nProcessing Fake Images...")
    fake_errors, fake_labels, fake_orig, fake_recon = get_reconstruction_errors(
        model, fake_loader, torch_device, is_openvino, ov_compiled_model
    )

    # Combine results
    all_errors = np.concatenate([real_errors, fake_errors])
    all_labels = np.concatenate([real_labels, fake_labels])

    # 1. Distribution Plot
    plt.figure(figsize=(10, 6))
    plt.hist(real_errors, bins=50, alpha=0.5, label="Real (Pristine)", density=True)
    plt.hist(fake_errors, bins=50, alpha=0.5, label="Fake (Deepfake)", density=True)
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Density")
    plt.title("Reconstruction Error Distribution: Real vs Fake")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "error_distribution.png"))
    plt.close()

    # 2. ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(all_labels, all_errors)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()

    # 3. Calculate Optimal Threshold
    # Youden's J statistic
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Predict using threshold
    predictions = (all_errors >= optimal_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, predictions, average="binary"
    )

    print("\n--- Evaluation Results ---")
    print(f"AUC Score: {roc_auc:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # 4. Visual Grid (Original vs Recon)
    num_samples = min(3, len(real_orig), len(fake_orig))
    fig, axes = plt.subplots(num_samples * 2, 2, figsize=(8, 4 * num_samples))

    for i in range(num_samples):
        # Real
        axes[i * 2, 0].imshow(real_orig[i])
        axes[i * 2, 0].set_title("Real - Original")
        axes[i * 2, 0].axis("off")

        axes[i * 2, 1].imshow(real_recon[i])
        axes[i * 2, 1].set_title(f"Real - Recon (MSE: {real_errors[i]:.4f})")
        axes[i * 2, 1].axis("off")

        # Fake
        axes[i * 2 + 1, 0].imshow(fake_orig[i])
        axes[i * 2 + 1, 0].set_title("Fake - Original")
        axes[i * 2 + 1, 0].axis("off")

        axes[i * 2 + 1, 1].imshow(fake_recon[i])
        axes[i * 2 + 1, 1].set_title(f"Fake - Recon (MSE: {fake_errors[i]:.4f})")
        axes[i * 2 + 1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reconstruction_grid.png"))
    plt.close()
    print(f"\nVisualizations and metrics saved to '{output_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate DAE for Deepfake Anomaly Detection"
    )
    parser.add_argument(
        "--real_dir",
        type=str,
        default="./data/processed/real",
        help="Path to real/pristine images",
    )
    parser.add_argument(
        "--fake_dir",
        type=str,
        default="./data/processed/fake",
        help="Path to fake images",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/dae_model.pth",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "npu"],
        help="Processing backend (cpu, cuda, npu)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save plots and metrics",
    )

    args = parser.parse_args()

    evaluate(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
