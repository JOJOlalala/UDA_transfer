"""Labeled datasets for Task II classification UDA.

Unlike data/datasets.py (unpaired, label-free — for CycleGAN), these loaders
return (image, label) for the source and target domains of each of the 5 pairs.

Normalization: symmetric [-1, 1] per channel, matching Task I CycleGAN output,
so translated images from M2/M3 drop into the same classifier without a
re-normalization step.

Per-dataset num_classes:
    mnist_usps    → 10
    svhn_mnist    → 10
    amazon_webcam → 31
    art_realworld → 65
    photo_sketch  →  7
"""
import os
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets as dsets, transforms as T
from PIL import Image


DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw"
)

NUM_CLASSES = {
    "mnist_usps": 10,
    "svhn_mnist": 10,
    "amazon_webcam": 31,
    "art_realworld": 65,
    "photo_sketch": 7,
}

IMG_SIZE = {
    "mnist_usps": 32,
    "svhn_mnist": 32,
    "amazon_webcam": 64,
    "art_realworld": 64,
    "photo_sketch": 64,
}

ARCH = {
    "mnist_usps": "smallcnn",
    "svhn_mnist": "smallcnn",
    "amazon_webcam": "resnet18",
    "art_realworld": "resnet18",
    "photo_sketch": "resnet18",
}


# ----- transforms ---------------------------------------------------------

def _transform_small_gray(img_size: int, train: bool) -> T.Compose:
    """32x32 grayscale → 3-channel, [-1, 1]."""
    ops = [
        T.Resize(img_size),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1)),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ]
    return T.Compose(ops)


def _transform_small_rgb(img_size: int, train: bool) -> T.Compose:
    """32x32 RGB (SVHN), [-1, 1]."""
    ops = [
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ]
    return T.Compose(ops)


CLASSIFIER_RES = int(os.environ.get("CLASSIFIER_RES", "224"))


def _transform_large(img_size: int, train: bool) -> T.Compose:
    # Office/PACS classifier input size. Default 224 (ResNet-18 native) for M0;
    # override to a matched mid-res (e.g. 128) via CLASSIFIER_RES env var when
    # training on translated 64×64 sources so the train/test blur is symmetric.
    del img_size
    r = CLASSIFIER_RES
    load = int(round(r * 1.143))  # 224 → 256, 128 → 146
    if train:
        ops = [
            T.Resize((load, load)),
            T.RandomCrop(r),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ]
    else:
        ops = [
            T.Resize((r, r)),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ]
    return T.Compose(ops)


# ----- folder dataset for pre-translated images --------------------------

class TranslatedFolderDataset(Dataset):
    """Mirror of torchvision.ImageFolder but reads translated-source PNGs.

    Expects a directory of the form:
        <root>/<class_name>/<id>.png

    Labels come from the parent class directory, matching the training-time
    source ImageFolder so class indices are consistent.
    """

    EXTENSIONS = {".png", ".jpg", ".jpeg"}

    def __init__(self, root: str, class_to_idx: dict, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.samples = []
        for cls, idx in class_to_idx.items():
            cdir = self.root / cls
            if not cdir.is_dir():
                continue
            for p in sorted(cdir.iterdir()):
                if p.suffix.lower() in self.EXTENSIONS:
                    self.samples.append((str(p), idx))
        assert len(self.samples) > 0, f"No images in {root}"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ----- per-dataset builders -----------------------------------------------

def _build_mnist_usps(root: str, domain: str, train: bool) -> Dataset:
    img_size = IMG_SIZE["mnist_usps"]
    tf = _transform_small_gray(img_size, train)
    if domain == "source":
        return dsets.MNIST(root=root, train=train, download=True, transform=tf)
    return dsets.USPS(root=root, train=train, download=True, transform=tf)


def _build_svhn_mnist(root: str, domain: str, train: bool) -> Dataset:
    img_size = IMG_SIZE["svhn_mnist"]
    if domain == "source":
        tf = _transform_small_rgb(img_size, train)
        split = "train" if train else "test"
        return dsets.SVHN(root=root, split=split, download=True, transform=tf)
    tf = _transform_small_gray(img_size, train)
    return dsets.MNIST(root=root, train=train, download=True, transform=tf)


def _image_folder_root(name: str, domain: str) -> Path:
    """Return the directory that ImageFolder should point at."""
    root = Path(DATA_ROOT)
    if name == "amazon_webcam":
        sub = "amazon" if domain == "source" else "webcam"
        return root / "office31" / sub / "images"
    if name == "art_realworld":
        sub = "art" if domain == "source" else "real_world"
        return root / "office_home" / sub
    if name == "photo_sketch":
        sub = "photo" if domain == "source" else "sketch"
        return root / "pacs" / sub
    raise ValueError(name)


def _build_large(name: str, domain: str, train: bool) -> dsets.ImageFolder:
    img_size = IMG_SIZE[name]
    tf = _transform_large(img_size, train)
    folder = _image_folder_root(name, domain)
    return dsets.ImageFolder(str(folder), transform=tf)


# ----- public API ---------------------------------------------------------

def build_dataset(name: str, domain: str, train: bool) -> Dataset:
    """Return a labeled Dataset for (dataset_name, domain, train/test).

    For MNIST/USPS/SVHN the torchvision train/test splits are honored.
    For Office-31/Office-Home/PACS there is no natural split — we expose the
    full folder in both `train=True` and `train=False`; see `split_train_val`
    for a deterministic 80/20 partition.
    """
    if name == "mnist_usps":
        return _build_mnist_usps(DATA_ROOT, domain, train)
    if name == "svhn_mnist":
        return _build_svhn_mnist(DATA_ROOT, domain, train)
    return _build_large(name, domain, train)


def split_train_val(ds: Dataset, val_frac: float = 0.1, seed: int = 0
                    ) -> Tuple[Subset, Subset]:
    """Deterministic index split (same seed → same split)."""
    n = len(ds)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(round(n * val_frac)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return Subset(ds, train_idx), Subset(ds, val_idx)


def build_translated_dataset(name: str, root: str, source_ds: Dataset = None
                             ) -> TranslatedFolderDataset:
    """Labeled dataset over bulk-translated source images.

    Digits: labels are saved under `<root>/<int>/…` (class_to_idx is synthetic,
    digits transform applied). Office/PACS: reuses the source ImageFolder's
    class_to_idx so labels stay aligned with the M0 classifier.
    """
    img_size = IMG_SIZE[name]
    if name in ("mnist_usps", "svhn_mnist"):
        # Target domain is grayscale (USPS / MNIST), so route translated
        # images through the same grayscale transform used for source/target
        # in M0 — keeps the classifier input distribution consistent.
        tf = _transform_small_gray(img_size, train=True)
        class_to_idx = {str(i): i for i in range(10)}
    else:
        tf = _transform_large(img_size, train=True)
        class_to_idx = source_ds.class_to_idx  # ImageFolder only
    return TranslatedFolderDataset(root, class_to_idx, transform=tf)


def build_loader(ds: Dataset, batch_size: int, shuffle: bool,
                 num_workers: int = 4) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
        persistent_workers=(num_workers > 0),
    )
