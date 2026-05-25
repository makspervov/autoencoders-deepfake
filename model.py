import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encoder module of the Variational Autoencoder.
    Compresses the input image into a latent space representation (mu and log_var).
    """

    def __init__(self, latent_dim: int = 128, in_channels: int = 3):
        super(Encoder, self).__init__()

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

        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = torch.flatten(x, start_dim=1)

        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)

        return mu, log_var


class Decoder(nn.Module):
    """
    Decoder module of the Variational Autoencoder.
    Reconstructs the image from the latent space representation.
    """

    def __init__(self, latent_dim: int = 128, out_channels: int = 3):
        super(Decoder, self).__init__()

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


class VAE(nn.Module):
    """
    Complete Variational Autoencoder network.
    Combines Encoder and Decoder.
    """

    def __init__(self, latent_dim: int = 128, in_channels: int = 3):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim, in_channels)
        self.decoder = Decoder(latent_dim, in_channels)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) while allowing backpropagation.
        z = mu + std * eps
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decoder(z)
        return reconstructed, mu, log_var


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    kld_weight: float = 0.00025,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the VAE loss function.
    Loss = Reconstruction Loss (MSE) + KL Divergence.

    Args:
        recon_x: Reconstructed images from decoder.
        x: Original input images.
        mu: Mean from the latent space.
        log_var: Log variance from the latent space.
        kld_weight: Weighting factor for the KL divergence term to balance with reconstruction loss.
                    Value depends heavily on image size and batch size.

    Returns:
        total_loss, recon_loss, kld_loss
    """
    # Reconstruction loss (Mean Squared Error)
    # Using sum reduction to be independent of batch size scale issues, then averaging at the end
    recon_loss = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)

    # KL Divergence
    # KL(N(mu, var) || N(0, 1)) = -0.5 * sum(1 + log(var) - mu^2 - var)
    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    )

    # Total loss
    total_loss = recon_loss + kld_weight * kld_loss

    return total_loss, recon_loss, kld_loss
