import os
import glob
from typing import Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class DeepFakeFaceDataset(Dataset):
    """
    Dataset class for loading pre-cropped face images from the DeepFakeFace dataset.
    Expects the directory structure to separate real and fake images.
    """

    def __init__(self, root_dir: str, image_size: int = 128, label: int = 0):
        """
        Args:
            root_dir (str): Path to the directory containing processed images.
            image_size (int): Size to resize the images to (default: 128).
            label (int): Label to assign to all images in this directory (0 for pristine, 1 for fake).
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.label = label

        # Define image transformations
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                # Normalization typically maps [0, 1] to [-1, 1] for VAE/GANs,
                # but for standard BCE reconstruction, [0, 1] is often preferred.
                # We will just stick to ToTensor() which normalizes to [0, 1].
            ]
        )

        # Load all image paths
        self.image_paths = []
        for ext in ["**/*.png", "**/*.jpg", "**/*.jpeg"]:
            self.image_paths.extend(
                glob.glob(os.path.join(root_dir, ext), recursive=True)
            )

        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {root_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            # Fallback to returning a zero tensor if image is corrupted to avoid crashing DataLoader
            print(f"Error loading {img_path}: {e}")
            image = torch.zeros((3, self.image_size, self.image_size))

        return image, self.label


def get_dataloaders(
    real_dir: str,
    fake_dir: str,
    batch_size: int = 64,
    image_size: int = 128,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoaders for training (only real data), and evaluation (mixed data).

    Args:
        real_dir: Path to directory containing real (pristine) face images.
        fake_dir: Path to directory containing fake (manipulated) face images.
        batch_size: Batch size for DataLoader.
        image_size: Size to resize images to.
        num_workers: Number of DataLoader workers.

    Returns:
        train_loader: DataLoader with only real images (for VAE training).
        test_real_loader: DataLoader with real images for evaluation.
        test_fake_loader: DataLoader with fake images for evaluation.
    """

    # Create datasets
    # Usually, we'd split the real dataset into train/val/test splits.
    # For simplicity, we assume real_dir contains the train split.
    real_dataset = DeepFakeFaceDataset(
        root_dir=real_dir, image_size=image_size, label=0
    )
    fake_dataset = DeepFakeFaceDataset(
        root_dir=fake_dir, image_size=image_size, label=1
    )

    # We create a simple random split for real data: 80% train, 20% test
    # If the dataset is too small, we just use it entirely for both for demo purposes.
    train_size = int(0.8 * len(real_dataset))
    test_real_size = len(real_dataset) - train_size

    if len(real_dataset) > 1:
        import torch

        # Use a generator with a fixed seed to prevent data leakage
        # between independent train.py and evaluate.py runs.
        generator = torch.Generator().manual_seed(42)
        train_dataset, test_real_dataset = torch.utils.data.random_split(
            real_dataset, [train_size, test_real_size], generator=generator
        )
    else:
        # Fallback if empty or only 1 item
        train_dataset = real_dataset
        test_real_dataset = real_dataset

    # Create DataLoaders
    # If the dataset is empty, skip dataloader creation to prevent ValueError
    train_loader = (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )
        if len(train_dataset) > 0
        else None
    )
    test_real_loader = (
        DataLoader(
            test_real_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        if len(test_real_dataset) > 0
        else None
    )
    test_fake_loader = (
        DataLoader(
            fake_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        if len(fake_dataset) > 0
        else None
    )

    return train_loader, test_real_loader, test_fake_loader
