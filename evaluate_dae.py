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


def get_reconstruction_errors(
    model, dataloader, device, is_openvino=False, ov_compiled_model=None
):
    errors = []
    labels = []
    originals = []
    reconstructions = []

    if not is_openvino:
        model.eval()

    with torch.no_grad():
        for data, label in tqdm(dataloader, desc="Evaluating"):
            
            if is_openvino and data.shape[0] != dataloader.batch_size:
                continue

            if is_openvino:
                result = ov_compiled_model([data.numpy()])
                recon_batch = result[ov_compiled_model.output(0)]
                recon_batch = torch.tensor(recon_batch)
            else:
                data = data.to(device)
                # У DAE возвращается только один тензор: recon_batch
                recon_batch = model(data)
                recon_batch = recon_batch.cpu()
                data = data.cpu()

            mse = F.mse_loss(recon_batch, data, reduction="none")
            mse_per_image = mse.view(mse.size(0), -1).mean(dim=1).numpy()

            errors.extend(mse_per_image)
            labels.extend(label.numpy())

            # Защита ОЗУ: сохраняем только 3 картинки для сетки
            if len(originals) < 3:
                orig_img = np.clip(data[0].numpy().transpose(1, 2, 0), 0.0, 1.0)
                recon_img = np.clip(recon_batch[0].numpy().transpose(1, 2, 0), 0.0, 1.0)
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

    is_openvino = (device == "npu")
    ov_compiled_model = None

    if is_openvino:
        print("NPU requested. Will use OpenVINO.")
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
        print(f"Warning: Model not found at {model_path}. Using random weights.")

    model.to(torch_device)
    model.eval()

    if is_openvino:
        core = ov.Core()
        target_device = "NPU" if "NPU" in core.available_devices else "CPU"
        print(f"Compiling directly to OpenVINO for {target_device}...")

        dummy_input = torch.randn(batch_size, 3, 128, 128)
        ov_model = ov.convert_model(model, example_input=dummy_input, input=[batch_size, 3, 128, 128])
        ov_model.reshape([batch_size, 3, 128, 128])
        
        ir_dir = os.path.join(output_dir, "ir")
        os.makedirs(ir_dir, exist_ok=True)
        ir_path = os.path.join(ir_dir, "dae.xml")
        ov.save_model(ov_model, ir_path, compress_to_fp16=True)
        ov_compiled_model = core.compile_model(ov_model, target_device)

    print("Initializing DataLoaders...")
    _, real_loader, fake_loader = get_dataloaders(
        real_dir=real_dir, fake_dir=fake_dir, batch_size=batch_size, num_workers=4
    )

    if real_loader is None or fake_loader is None:
        print("Error: Dataloaders are empty.")
        return

    print("\nProcessing Real Images...")
    real_errors, real_labels, real_orig, real_recon = get_reconstruction_errors(
        model, real_loader, torch_device, is_openvino, ov_compiled_model
    )

    print("\nProcessing Fake Images...")
    fake_errors, fake_labels, fake_orig, fake_recon = get_reconstruction_errors(
        model, fake_loader, torch_device, is_openvino, ov_compiled_model
    )

    # --- БАЛАНСИРОВКА 1:1 ---
    print("\nBalancing dataset to 1:1 ratio for objective metrics...")
    min_samples = min(len(real_errors), len(fake_errors))
    np.random.seed(42)

    if len(fake_errors) > min_samples:
        indices = np.random.choice(len(fake_errors), min_samples, replace=False)
        fake_errors = fake_errors[indices]
        fake_labels = fake_labels[indices]
    elif len(real_errors) > min_samples:
        indices = np.random.choice(len(real_errors), min_samples, replace=False)
        real_errors = real_errors[indices]
        real_labels = real_labels[indices]

    all_errors = np.concatenate([real_errors, fake_errors])
    all_labels = np.concatenate([real_labels, fake_labels])

    plt.figure(figsize=(10, 6))
    plt.hist(real_errors, bins=50, alpha=0.5, label="Real (Pristine)", density=True)
    plt.hist(fake_errors, bins=50, alpha=0.5, label="Fake (Deepfake)", density=True)
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Density")
    plt.title("DAE Reconstruction Error Distribution (Balanced 1:1)")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "dae_error_distribution.png"))
    plt.close()

    fpr, tpr, thresholds = roc_curve(all_labels, all_errors)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "dae_roc_curve.png"))
    plt.close()

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    predictions = (all_errors >= optimal_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, predictions, average="binary"
    )

    print("\n--- Evaluation Results ---")
    print(f"Dataset Size: {len(real_errors)} Real vs {len(fake_errors)} Fake")
    print(f"AUC Score: {roc_auc:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    num_samples = min(3, len(real_orig), len(fake_orig))
    if num_samples > 0:
        fig, axes = plt.subplots(num_samples * 2, 2, figsize=(8, 4 * num_samples))
        for i in range(num_samples):
            axes[i * 2, 0].imshow(real_orig[i])
            axes[i * 2, 0].set_title("Real - Original")
            axes[i * 2, 0].axis("off")

            axes[i * 2, 1].imshow(real_recon[i])
            axes[i * 2, 1].set_title("Real - Recon")
            axes[i * 2, 1].axis("off")

            axes[i * 2 + 1, 0].imshow(fake_orig[i])
            axes[i * 2 + 1, 0].set_title("Fake - Original")
            axes[i * 2 + 1, 0].axis("off")

            axes[i * 2 + 1, 1].imshow(fake_recon[i])
            axes[i * 2 + 1, 1].set_title("Fake - Recon")
            axes[i * 2 + 1, 1].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "dae_reconstruction_grid.png"))
        plt.close()

    print(f"\nVisualizations and metrics saved to '{output_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DAE for Deepfake Anomaly Detection")
    parser.add_argument("--real_dir", type=str, default="./data/processed2/real")
    parser.add_argument("--fake_dir", type=str, default="./data/processed2/fake")
    parser.add_argument("--model_path", type=str, default="./models/dae_model.pth")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "npu"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="./results_dae")

    args = parser.parse_args()

    evaluate(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )