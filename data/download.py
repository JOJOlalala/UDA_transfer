"""Dataset download utilities for UDA experiments.

Torchvision datasets (MNIST, USPS, SVHN) are auto-downloaded.
Office-31 requires gdown (pip install gdown).
PACS uses HuggingFace datasets (pip install datasets).

Download links (verified 2026-04-13):
  Office-31: gdown "https://drive.google.com/uc?id=0B4IapRTv9pJ1WGZVd1VDMmhwdlE"
      ~77MB tar.gz, extracts to {amazon,dslr,webcam}/images/<class>/
      Alt: https://wjdcloud.blob.core.windows.net/dataset/OFFICE31.zip
      Alt: kaggle datasets download -d xixuhu/office31
  PACS: HuggingFace flwrlabs/pacs (~191MB parquet)
      Google Drive folder 401-restricted — do not use gdown.
  DEAD: DomainBed PACS file ID 1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd (404)
"""
import os
import sys
import subprocess
import shutil
import torchvision.datasets as dsets


DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
GDOWN = os.path.join(os.path.dirname(sys.executable), "gdown")


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


def download_office31(root=DATA_ROOT):
    """Download Office-31 via gdown from Google Drive (~77MB)."""
    dest = os.path.join(root, "office31")
    if os.path.isdir(dest) and len(os.listdir(dest)) >= 2:
        print(f"Office-31 already exists at {dest}, skipping.")
        return

    print("Downloading Office-31 from Google Drive...")
    tar_path = os.path.join(root, "OFFICE31.tar.gz")
    subprocess.run(
        [GDOWN, "https://drive.google.com/uc?id=0B4IapRTv9pJ1WGZVd1VDMmhwdlE",
         "-O", tar_path],
        check=True,
    )
    print("Extracting Office-31...")
    subprocess.run(["tar", "xzf", tar_path, "-C", root], check=True)
    # Rename extracted dir to office31
    extracted = os.path.join(root, "domain_adaptation_images")
    if os.path.isdir(extracted):
        os.rename(extracted, dest)
    if os.path.isfile(tar_path):
        os.remove(tar_path)
    print(f"Office-31 saved to {dest}")


def download_pacs(root=DATA_ROOT):
    """Download PACS (photo + sketch) from HuggingFace flwrlabs/pacs."""
    dest = os.path.join(root, "pacs")
    if os.path.isdir(dest) and len(os.listdir(dest)) >= 2:
        print(f"PACS already exists at {dest}, skipping.")
        return

    from datasets import load_dataset

    print("Loading PACS from HuggingFace (flwrlabs/pacs)...")
    ds = load_dataset("flwrlabs/pacs", split="train")
    print(f"Loaded {len(ds)} samples")

    for domain in ["photo", "sketch"]:
        subset = [x for x in ds if x["domain"] == domain]
        print(f"  {domain}: {len(subset)} images")
        for i, item in enumerate(subset):
            label = ds.features["label"].int2str(item["label"])
            save_dir = os.path.join(dest, domain, label)
            os.makedirs(save_dir, exist_ok=True)
            item["image"].save(os.path.join(save_dir, f"{i:04d}.jpg"))
        print(f"  {domain} saved.")

    print(f"PACS saved to {dest}")


def download_office_home(root=DATA_ROOT):
    """Download Office-Home (art + real_world) from HuggingFace flwrlabs/office-home."""
    dest = os.path.join(root, "office_home")
    if os.path.isdir(dest) and len(os.listdir(dest)) >= 2:
        print(f"Office-Home already exists at {dest}, skipping.")
        return

    from datasets import load_dataset

    print("Loading Office-Home from HuggingFace (flwrlabs/office-home)...")
    ds = load_dataset("flwrlabs/office-home", split="train")
    print(f"Loaded {len(ds)} samples")

    for domain in ["art", "real_world"]:
        subset = [x for x in ds if x["domain"] == domain]
        print(f"  {domain}: {len(subset)} images")
        for i, item in enumerate(subset):
            label = ds.features["label"].int2str(item["label"])
            save_dir = os.path.join(dest, domain, label)
            os.makedirs(save_dir, exist_ok=True)
            item["image"].save(os.path.join(save_dir, f"{i:04d}.jpg"))
        print(f"  {domain} saved.")

    print(f"Office-Home saved to {dest}")


def download_all_torchvision(root=DATA_ROOT):
    """Download all torchvision-available datasets."""
    os.makedirs(root, exist_ok=True)
    download_mnist(root)
    download_usps(root)
    download_svhn(root)
    print(f"\nAll torchvision datasets saved to {root}")


def download_all(root=DATA_ROOT):
    """Download all datasets."""
    os.makedirs(root, exist_ok=True)
    download_mnist(root)
    download_usps(root)
    download_svhn(root)
    download_office31(root)
    download_pacs(root)
    download_office_home(root)
    print(f"\nAll datasets saved to {root}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=["all", "torchvision", "mnist", "usps", "svhn",
                                 "office31", "pacs", "office_home"])
    parser.add_argument("--root", default=DATA_ROOT)
    args = parser.parse_args()
    os.makedirs(args.root, exist_ok=True)

    if args.dataset == "all":
        download_all(args.root)
    elif args.dataset == "torchvision":
        download_all_torchvision(args.root)
    else:
        globals()[f"download_{args.dataset}"](args.root)
