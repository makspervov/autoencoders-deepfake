import os
import argparse
from tqdm import tqdm
import torch
import torch.optim as optim

from dataset_sae import get_dataloaders_sae
from model_sae import SAE, sae_loss


def train_sae(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    save_path: str,
    sparsity_param: float = 0.05,
    beta: float = 0.5,
):
    """
    Trains the SAE only on real/pristine images enforcing a sparsity constraint on the latent space.
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
    train_loader, _, _ = get_dataloaders_sae(
        batch_size=batch_size, num_workers=2
    )

    if train_loader is None:
        print("Error: Training dataset is empty. Please check your dataset class.")
        return

    # Initialize Model and Optimizer
    model = SAE().to(device_obj)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Starting training for {epochs} epochs...")
    model.train()

    # Training Loop
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_recon = 0.0
        total_sparsity = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(device_obj)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass: model reconstructs data and outputs latent representations
            recon_batch, latent = model(data)

            # Compute loss
            loss, recon_loss, sparsity_loss = sae_loss(
                recon_batch, data, latent, sparsity_param=sparsity_param, beta=beta
            )

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_sparsity += sparsity_loss.item()

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Recon": f"{recon_loss.item():.4f}",
                    "Sparsity": f"{sparsity_loss.item():.4f}",
                }
            )

        # Calculate epoch averages
        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon / len(train_loader)
        avg_sparsity = total_sparsity / len(train_loader)

        print(
            f"Epoch {epoch} Summary: Avg Loss: {avg_loss:.4f} | Avg Recon: {avg_recon:.4f} | Avg Sparsity: {avg_sparsity:.4f}"
        )

    # Save model
    print(f"Training complete. Saving model to {save_path}...")
    torch.save(model.state_dict(), save_path)
    print("Model saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SAE for Deepfake Anomaly Detection"
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
        default="./models/sae_model.pth",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--sparsity_param",
        type=float,
        default=0.05,
        help="Target average activation level for hidden units (rho)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Weight of the sparsity penalty in the total loss function",
    )

    args = parser.parse_args()

    train_sae(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        save_path=args.save_path,
        sparsity_param=args.sparsity_param,
        beta=args.beta,
    )
