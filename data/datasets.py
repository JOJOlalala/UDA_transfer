"""Unified dataset classes for CycleGAN training."""
import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets as dsets, transforms as T
from PIL import Image


DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")


# ---------------------------------------------------------------------------
# Wrapper to extract only images (drop labels) from torchvision datasets
# ---------------------------------------------------------------------------
class ImageOnlyDataset(Dataset):
    """Wraps a torchvision dataset to return only the image (no label)."""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img


# ---------------------------------------------------------------------------
# Folder-based dataset for Office-31 / PACS
# ---------------------------------------------------------------------------
class FolderImageDataset(Dataset):
    """Load all images from a folder (recursively), ignoring class structure."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.paths = sorted(
            p for p in Path(root).rglob("*")
            if p.suffix.lower() in self.EXTENSIONS
        )
        assert len(self.paths) > 0, f"No images found in {root}"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


# ---------------------------------------------------------------------------
# Unpaired dataset for CycleGAN
# ---------------------------------------------------------------------------
class UnpairedDataset(Dataset):
    """Returns one image from domain A and one from domain B (unpaired)."""

    def __init__(self, dataset_A, dataset_B):
        self.A = dataset_A
        self.B = dataset_B

    def __len__(self):
        return max(len(self.A), len(self.B))

    def __getitem__(self, idx):
        img_A = self.A[idx % len(self.A)]
        img_B = self.B[random.randint(0, len(self.B) - 1)]
        return img_A, img_B


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_transform_small(img_size=32):
    """Transform for small grayscale datasets (MNIST, USPS, SVHN)."""
    return T.Compose([
        T.Resize(img_size),
        T.ToTensor(),                    # [0, 1]
        T.Normalize([0.5] * 3, [0.5] * 3),  # -> [-1, 1]
    ])


def grayscale_to_rgb(img):
    """Convert 1-channel tensor to 3-channel by repeating."""
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    return img


def get_transform_small_gray(img_size=32):
    """Transform for grayscale datasets, converting to 3-ch."""
    return T.Compose([
        T.Resize(img_size),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1)),  # 1ch -> 3ch
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])


def get_transform_large(img_size=256):
    """Transform for large RGB datasets (Office-31, PACS)."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])


def get_transform_large_test(img_size=256):
    """Test-time transform (no augmentation)."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------
def build_dataset(config, split="train"):
    """Build an UnpairedDataset from config dict.

    Args:
        config: dict with keys 'dataset.name', 'dataset.img_size', etc.
        split: 'train' or 'test'
    """
    ds_cfg = config["dataset"]
    name = ds_cfg["name"]
    img_size = ds_cfg["img_size"]
    root = ds_cfg.get("data_root", DATA_ROOT)
    is_train = (split == "train")

    if name == "mnist_usps":
        # MNIST (grayscale) -> USPS (grayscale), both to 3ch
        tf = get_transform_small_gray(img_size)
        ds_A = ImageOnlyDataset(
            dsets.MNIST(root=root, train=is_train, download=True), transform=tf
        )
        ds_B = ImageOnlyDataset(
            dsets.USPS(root=root, train=is_train, download=True), transform=tf
        )

    elif name == "svhn_mnist":
        # SVHN (RGB) -> MNIST (grayscale->3ch)
        tf_svhn = get_transform_small(img_size)
        tf_mnist = get_transform_small_gray(img_size)
        ds_A = ImageOnlyDataset(
            dsets.SVHN(root=root, split="train" if is_train else "test", download=True),
            transform=tf_svhn,
        )
        ds_B = ImageOnlyDataset(
            dsets.MNIST(root=root, train=is_train, download=True),
            transform=tf_mnist,
        )

    elif name == "amazon_webcam":
        tf = get_transform_large(img_size) if is_train else get_transform_large_test(img_size)
        ds_A = FolderImageDataset(os.path.join(root, "office31", "amazon"), transform=tf)
        ds_B = FolderImageDataset(os.path.join(root, "office31", "webcam"), transform=tf)

    elif name == "photo_sketch":
        tf = get_transform_large(img_size) if is_train else get_transform_large_test(img_size)
        ds_A = FolderImageDataset(os.path.join(root, "pacs", "photo"), transform=tf)
        ds_B = FolderImageDataset(os.path.join(root, "pacs", "sketch"), transform=tf)

    else:
        raise ValueError(f"Unknown dataset: {name}")

    return UnpairedDataset(ds_A, ds_B)


def build_dataloader(config, split="train"):
    """Build DataLoader from config."""
    dataset = build_dataset(config, split)
    batch_size = config["training"]["batch_size"] if split == "train" else 1
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=4,
        pin_memory=True,
        drop_last=(split == "train"),
    )
