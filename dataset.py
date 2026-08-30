import os
import glob
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image

class LocalFaceDataset(Dataset):
    """
    Loads locally processed images from disk.
    """
    def __init__(self, data_dir: str, label: int, image_size: int = 128):
        self.data_dir = data_dir
        self.label = label
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

        # Collect all image paths with common image extensions
        self.image_paths = []
        for ext in ["**/*.png", "**/*.jpg", "**/*.jpeg"]:
            self.image_paths.extend(glob.glob(os.path.join(data_dir, ext), recursive=True))

        if not self.image_paths:
            print(f"WARNING: Directory {data_dir} is empty or does not exist!")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            print(f"ERROR: Failed to read {img_path}: {e}")
            image = torch.zeros((3, self.image_size, self.image_size))

        return image, self.label


def get_dataloaders(
    real_dir: str,
    fake_dir: str,
    batch_size: int = 64,
    image_size: int = 128,
    num_workers: int = 4,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    
    print(f"Reading real images from: {real_dir}")
    real_dataset = LocalFaceDataset(data_dir=real_dir, label=0, image_size=image_size)
    
    print(f"Reading fake images from: {fake_dir}")
    fake_dataset = LocalFaceDataset(data_dir=fake_dir, label=1, image_size=image_size)

    # Split real dataset into training and testing sets (80% train, 20% test)
    train_size = int(0.8 * len(real_dataset))
    test_real_size = len(real_dataset) - train_size

    if len(real_dataset) > 1:
        generator = torch.Generator().manual_seed(42)
        train_dataset, test_real_dataset = torch.utils.data.random_split(
            real_dataset, [train_size, test_real_size], generator=generator
        )
    else:
        train_dataset = real_dataset
        test_real_dataset = real_dataset

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True if torch.cuda.is_available() else False
    ) if len(train_dataset) > 0 else None

    test_real_loader = DataLoader(
        test_real_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    ) if len(test_real_dataset) > 0 else None

    test_fake_loader = DataLoader(
        fake_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    ) if len(fake_dataset) > 0 else None

    return train_loader, test_real_loader, test_fake_loader