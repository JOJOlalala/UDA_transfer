"""Dataset download utilities for UDA experiments."""
import os
import torchvision.datasets as dsets
import torchvision.transforms as T


DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")


def download_mnist(root=DATA_ROOT):
    dsets.MNIST(root=root, train=True, download=True)
    dsets.MNIST(root=root, train=False, download=True)
    print("MNIST downloaded.")


def download_usps(root=DATA_ROOT):
    dsets.USPS(root=root, train=True, download=True)
    dsets.USPS(root=root, train=False, download=True)
    print("USPS downloaded.")


def download_svhn(root=DATA_ROOT):
    dsets.SVHN(root=root, split="train", download=True)
    dsets.SVHN(root=root, split="test", download=True)
    print("SVHN downloaded.")


def download_all_torchvision(root=DATA_ROOT):
    """Download all torchvision-available datasets."""
    os.makedirs(root, exist_ok=True)
    download_mnist(root)
    download_usps(root)
    download_svhn(root)
    print(f"\nAll torchvision datasets saved to {root}")
    print("NOTE: Office-31 and PACS must be downloaded manually.")
    print("  Office-31: place under data/raw/office31/{amazon,webcam,dslr}/")
    print("  PACS: place under data/raw/pacs/{photo,sketch,art_painting,cartoon}/")


if __name__ == "__main__":
    download_all_torchvision()
