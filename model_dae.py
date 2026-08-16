import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderDAE(nn.Module):
    """
    Encoder module of the Denoising Autoencoder.
    Compresses the input image into a latent space representation.
    """

    def __init__(self, latent_dim: int = 128, in_channels: int = 3):
        super(EncoderDAE, self).__init__()

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
        latent = self.fc(x)

        return latent


class DecoderDAE(nn.Module):
    """
    Decoder module of the Denoising Autoencoder.
    Reconstructs the image from the latent space representation.
    """

    def __init__(self, latent_dim: int = 128, out_channels: int = 3):
        super(DecoderDAE, self).__init__()

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


class DAE(nn.Module):
    """
    Complete Denoising Autoencoder network.
    Combines Encoder and Decoder.
    """

    def __init__(self, latent_dim: int = 128, in_channels: int = 3):
        super(DAE, self).__init__()
        self.encoder = EncoderDAE(latent_dim, in_channels)
        self.decoder = DecoderDAE(latent_dim, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


def inject_noise(x: torch.Tensor, noise_factor: float = 0.2) -> torch.Tensor:
    """
    Injects Gaussian noise into the input tensor.

    Args:
        x: Input image tensor (usually in range [0, 1]).
        noise_factor: Multiplier for the standard normal noise.

    Returns:
        Noisy image tensor, clamped to [0, 1].
    """
    noise = torch.randn_like(x) * noise_factor
    noisy_x = x + noise
    # Assuming the input is normalized to [0, 1]
    return torch.clamp(noisy_x, 0.0, 1.0)


def dae_loss(recon_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Computes the DAE loss function (Mean Squared Error).

    Args:
        recon_x: Reconstructed images from decoder.
        x: Original (clean) input images.

    Returns:
        Reconstruction loss (MSE).
    """
    # Reconstruction loss (Mean Squared Error)
    # Using mean reduction is standard for DAE
    return F.mse_loss(recon_x, x)
