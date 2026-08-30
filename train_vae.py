import os
import argparse
from tqdm import tqdm
import torch
import torch.optim as optim

from dataset import get_dataloaders
from model_vae import VAE, vae_loss


def train_vae(
    real_dir: str,
    fake_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    save_path: str,
):
    """
    Trains the VAE only on real/pristine images.
    """
    # Force CPU if requested, otherwise check CUDA availability
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    device_obj = torch.device(device)
    print(f"Using device: {device_obj}")

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Initialize dataloaders
    print("Initializing DataLoaders...")
    train_loader, _, _ = get_dataloaders(
        real_dir=real_dir, fake_dir=fake_dir, batch_size=batch_size, num_workers=2
    )

    if train_loader is None:
        print("Error: Training dataset is empty. Please check your data directory.")
        return

    # Initialize Model and Optimizer
    model = VAE().to(device_obj)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Starting training for {epochs} epochs...")
    model.train()

    # Training Loop
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_recon = 0.0
        total_kld = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(device_obj)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            recon_batch, mu, log_var = model(data)

            # Compute loss
            loss, recon_loss, kld_loss = vae_loss(recon_batch, data, mu, log_var)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld_loss.item()

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "Loss": f"{loss.item():.2f}",
                    "Recon": f"{recon_loss.item():.2f}",
                    "KLD": f"{kld_loss.item():.2f}",
                }
            )

        # Calculate epoch averages
        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon / len(train_loader)
        avg_kld = total_kld / len(train_loader)

        print(
            f"Epoch {epoch} Summary: Avg Loss: {avg_loss:.4f} | Avg Recon: {avg_recon:.4f} | Avg KLD: {avg_kld:.4f}"
        )

    # Save model
    print(f"Training complete. Saving model to {save_path}...")
    torch.save(model.state_dict(), save_path)
    print("Model saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train VAE for Deepfake Anomaly Detection"
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
        help="Path to fake images (used only for dataloader initialization here, not training)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use (cpu or cuda)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./models/vae_model.pth",
        help="Path to save the trained model",
    )

    args = parser.parse_args()

    train_vae(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        save_path=args.save_path,
    )
