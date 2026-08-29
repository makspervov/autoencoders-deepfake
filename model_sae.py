import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderSAE(nn.Module):
    """
    Encoder module of the Sparse Autoencoder.
    Compresses the input image into a latent space representation, where sparsity will be enforced.
    """

    def __init__(self, latent_dim: int = 128, in_channels: int = 3):
        super(EncoderSAE, self).__init__()

        # Expected input: (B, 3, 128, 128)
        self.conv1 = nn.Conv2d(
            in_channels, 32, kernel_size=4, stride=2, padding=1
        )  # -> (B, 32, 64, 64)
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=4, stride=2, padding=1
        )  # -> (B, 64, 32, 32)
        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=4, stride=2, padding=1
        )  # -> (B, 128, 16, 16)
        self.conv4 = nn.Conv2d(
            128, 256, kernel_size=4, stride=2, padding=1
        )  # -> (B, 256, 8, 8)

        self.fc = nn.Linear(256 * 8 * 8, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = torch.flatten(x, start_dim=1)

        # We use ReLU or Sigmoid on the latent space to enforce positive activations
        # which is standard for KL-divergence sparsity or L1 sparsity
        latent = torch.sigmoid(self.fc(x))

        return latent


class DecoderSAE(nn.Module):
    """
    Decoder module of the Sparse Autoencoder.
    Reconstructs the image from the sparse latent representation.
    """

    def __init__(self, latent_dim: int = 128, out_channels: int = 3):
        super(DecoderSAE, self).__init__()

        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)

        # Expected input shape after unflattening: (B, 256, 8, 8)
        self.deconv1 = nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1
        )  # -> (B, 128, 16, 16)
        self.deconv2 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )  # -> (B, 64, 32, 32)
        self.deconv3 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1
        )  # -> (B, 32, 64, 64)
        self.deconv4 = nn.ConvTranspose2d(
            32, out_channels, kernel_size=4, stride=2, padding=1
        )  # -> (B, 3, 128, 128)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(-1, 256, 8, 8)

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(
            self.deconv4(x)
        )  # Sigmoid to ensure pixel values are in [0, 1]

        return x


from typing import Tuple

class SAE(nn.Module):
    """
    Complete Sparse Autoencoder network.
    Combines Encoder and Decoder and exposes latent representation for loss calculation.
    """

    def __init__(self, latent_dim: int = 128, in_channels: int = 3):
        super(SAE, self).__init__()
        self.encoder = EncoderSAE(latent_dim, in_channels)
        self.decoder = DecoderSAE(latent_dim, in_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


def sae_loss(recon_x: torch.Tensor, x: torch.Tensor, latent: torch.Tensor, sparsity_param: float = 0.05, beta: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the SAE loss function (Mean Squared Error + KL Divergence Sparsity Penalty).

    Args:
        recon_x: Reconstructed images from decoder.
        x: Original (clean) input images.
        latent: Latent space representation (activations) of the batch.
        sparsity_param: The target average activation level for the hidden units.
        beta: The weight of the sparsity penalty.

    Returns:
        total_loss, recon_loss, sparsity_loss
    """
    # 1. Reconstruction loss (Mean Squared Error)
    recon_loss = F.mse_loss(recon_x, x)

    # 2. Sparsity penalty (KL Divergence between target sparsity and average batch activation)
    # Average activation of each hidden unit over the batch
    rho_hat = torch.mean(latent, dim=0)

    # Avoid numerical instability with log
    rho_hat = torch.clamp(rho_hat, min=1e-5, max=1-1e-5)

    # Target sparsity parameter (rho)
    rho = torch.tensor([sparsity_param], device=latent.device)

    # KL Divergence: sum(rho * log(rho/rho_hat) + (1-rho) * log((1-rho)/(1-rho_hat)))
    kl_div = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    sparsity_loss = torch.sum(kl_div)

    # Alternatively, L1 regularization could be used:
    # sparsity_loss = torch.mean(torch.abs(latent))

    # Total loss
    total_loss = recon_loss + beta * sparsity_loss

    return total_loss, recon_loss, sparsity_loss
