"""Bulk-translate a source train split through a trained CycleGAN G_AB.

Outputs one PNG per source image at the CycleGAN training resolution
(32 for digits, 64 for Office/PACS) under:
    <out>/<class_or_label>/<index>.png

Layout matches `data_labeled.build_translated_dataset` — the M2/M3 classifier
retrain stage loads these PNGs and upsamples through the standard transform
(224 for Office/PACS, 32 + grayscale for digits).
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets as dsets, transforms as T
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.generator import ResNetGenerator
from task2 import data_labeled as DL


def _source_transform(name: str, img_size: int = None) -> T.Compose:
    """Match the preprocessing that G_AB was trained on (see data/datasets.py).
    img_size defaults to DL.IMG_SIZE[name] but can be overridden for
    checkpoints trained at a different resolution (e.g. 256).
    """
    if img_size is None:
        img_size = DL.IMG_SIZE[name]
    if name == "mnist_usps":
        return T.Compose([
            T.Resize(img_size),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ])
    if name == "svhn_mnist":
        # SVHN source fed as RGB at CycleGAN train time.
        return T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ])
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])


def _build_labeled_source(name: str, img_size: int = None):
    """Return (dataset, idx_to_dirname) — deterministic, labels preserved."""
    tf = _source_transform(name, img_size)
    root = DL.DATA_ROOT
    if name == "mnist_usps":
        ds = dsets.MNIST(root=root, train=True, download=True, transform=tf)
        return ds, {i: str(i) for i in range(10)}
    if name == "svhn_mnist":
        ds = dsets.SVHN(root=root, split="train", download=True, transform=tf)
        return ds, {i: str(i) for i in range(10)}
    folder = DL._image_folder_root(name, "source")
    ds = dsets.ImageFolder(str(folder), transform=tf)
    idx_to_dirname = {v: k for k, v in ds.class_to_idx.items()}
    return ds, idx_to_dirname


def _load_G_AB(ckpt_path: str, config_yaml: str, device: torch.device):
    with open(config_yaml) as f:
        cfg = yaml.safe_load(f)
    mc = cfg["model"]
    gen_type = mc.get("generator", "resnet")
    if gen_type == "unet":
        from models.generator_unet import UNetGenerator
        G = UNetGenerator(
            input_nc=3, output_nc=3,
            ngf=mc["ngf"], n_downsample=mc.get("n_downsample", 4),
        ).to(device)
    else:
        G = ResNetGenerator(
            input_nc=3, output_nc=3,
            ngf=mc["ngf"], n_blocks=mc["n_blocks"],
            n_downsample=mc.get("n_downsample", 2),
        ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    G.load_state_dict(ckpt["G_AB"])
    G.eval()
    return G


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DL.NUM_CLASSES))
    p.add_argument("--ckpt", required=True, help="CycleGAN checkpoint (*.pth)")
    p.add_argument("--config", required=True, help="CycleGAN config YAML used for training")
    p.add_argument("--out", required=True, help="Output root dir")
    p.add_argument("--mode", default="pixel", choices=["pixel", "spectral", "posthoc"],
                   help="pixel: G_AB(x). spectral: G_AB(low)+high. posthoc: low(G_AB(x))+high(x).")
    p.add_argument("--beta", type=float, default=0.2,
                   help="FFT low-pass radius fraction (only for --mode spectral)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  dataset={args.dataset}  mode={args.mode}")
    print(f"ckpt={args.ckpt}")

    G = _load_G_AB(args.ckpt, args.config, device)
    # Read img_size from the CycleGAN config so translations match training res.
    with open(args.config) as f:
        cyclegan_img_size = yaml.safe_load(f)["dataset"]["img_size"]
    ds, idx_to_dirname = _build_labeled_source(args.dataset, img_size=cyclegan_img_size)
    print(f"source: n={len(ds)}  classes={len(idx_to_dirname)}  img_size={cyclegan_img_size}")

    if args.mode in ("spectral", "posthoc"):
        from utils.spectral import fft_decompose, fft_recombine
        print(f"{args.mode} recombination: beta={args.beta}")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    for name in idx_to_dirname.values():
        (out_root / name).mkdir(exist_ok=True)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    per_class = {k: 0 for k in idx_to_dirname}
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            if args.mode == "spectral":
                low, high = fft_decompose(x, args.beta)
                translated_low = G(low).clamp(-1, 1)
                fake = fft_recombine(translated_low, high)
            elif args.mode == "posthoc":
                translated_full = G(x).clamp(-1, 1)
                translated_low, _ = fft_decompose(translated_full, args.beta)
                _, source_high = fft_decompose(x, args.beta)
                fake = fft_recombine(translated_low, source_high)
            else:
                fake = G(x).clamp(-1, 1)
            fake01 = (fake + 1) / 2
            for i in range(fake01.size(0)):
                label = int(y[i].item())
                dirname = idx_to_dirname[label]
                idx = per_class[label]
                per_class[label] += 1
                save_image(fake01[i], str(out_root / dirname / f"{idx:06d}.png"))
            total += fake01.size(0)
            if total % 1024 == 0 or total == len(ds):
                print(f"  wrote {total}/{len(ds)}")
    print(f"DONE  wrote {total} images to {out_root}")


if __name__ == "__main__":
    main()
