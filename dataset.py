import os
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image

class DeepFakeFaceDataset(Dataset):
    """
    Dataset class for loading face images directly from the Hugging Face datasets library.
    It can be used by both the VAE and DAE pipelines.
    """

    def __init__(self, is_fake: bool, image_size: int = 128, max_samples: int = 5000):
        """
        Args:
            is_fake (bool): True for fake images, False for real images.
            image_size (int): Size to resize the images to.
            max_samples (int): Max samples to load to memory.
        """
        self.image_size = image_size
        self.label = 1 if is_fake else 0

        # Define image transformations
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

        print(f"Loading Hugging Face dataset 'OpenRL/DeepFakeFace' (is_fake={is_fake})...")

        # The OpenRL dataset on HF has 120,000 images in 'train' split.
        # The first 90k are fakes.
        # The last 30k are wiki (reals).

        self.samples = []
        try:
            # We use non-streaming load_dataset but only index into what we need.
            # Loading the Hugging Face dataset lazily maps the parquet files via pyarrow,
            # which is incredibly efficient and fast for random access without loading everything to RAM.
            self.ds = load_dataset('OpenRL/DeepFakeFace', split='train')

            self.start_idx = 0 if is_fake else 90000
            self.end_idx = min(self.start_idx + max_samples, len(self.ds)) if max_samples else (90000 if is_fake else len(self.ds))
            self.num_samples = self.end_idx - self.start_idx

        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.ds = None
            self.num_samples = 0

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.ds is None:
            return torch.zeros((3, self.image_size, self.image_size)), self.label

        real_idx = self.start_idx + idx
        try:
            image = self.ds[real_idx]['image']
            image = image.convert("RGB")
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {real_idx}: {e}")
            image = torch.zeros((3, self.image_size, self.image_size))

        return image, self.label


def get_dataloaders(
    real_dir: str,
    fake_dir: str,
    batch_size: int = 64,
    image_size: int = 128,
    num_workers: int = 4,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Creates DataLoaders for training (only real data), and evaluation (mixed data).

    Args:
        real_dir: Ignored, uses Hugging Face dataset.
        fake_dir: Ignored, uses Hugging Face dataset.
        batch_size: Batch size for DataLoader.
        image_size: Size to resize images to.
        num_workers: Number of DataLoader workers.

    Returns:
        train_loader: DataLoader with only real images (for VAE/DAE training).
        test_real_loader: DataLoader with real images for evaluation.
        test_fake_loader: DataLoader with fake images for evaluation.
    """

    # Create datasets using Hugging Face
    # Provide a reasonable limit (e.g. 5000 images each) for demonstration,
    # to avoid overly long epochs on constrained environments.
    real_dataset = DeepFakeFaceDataset(
        is_fake=False, image_size=image_size, max_samples=5000
    )
    fake_dataset = DeepFakeFaceDataset(
        is_fake=True, image_size=image_size, max_samples=5000
    )

    # We create a simple random split for real data: 80% train, 20% test
    # If the dataset is too small, we just use it entirely for both for demo purposes.
    train_size = int(0.8 * len(real_dataset))
    test_real_size = len(real_dataset) - train_size

    if len(real_dataset) > 1:
        # Use a generator with a fixed seed to prevent data leakage
        generator = torch.Generator().manual_seed(42)
        train_dataset, test_real_dataset = torch.utils.data.random_split(
            real_dataset, [train_size, test_real_size], generator=generator
        )
    else:
        # Fallback if empty or only 1 item
        train_dataset = real_dataset
        test_real_dataset = real_dataset

    # Create DataLoaders
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
