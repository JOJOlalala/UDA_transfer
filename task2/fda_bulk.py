"""Bulk FDA adaptation: swap low-freq amplitude of source with target.

No generator needed — pure FFT operation. Outputs FDA-adapted source images
with source labels preserved.
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets as dsets, transforms as T
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task2 import data_labeled as DL
from utils.spectral import fda_transfer


def _build_source_and_target(name, img_size):
    """Build labeled source dataset + unlabeled target dataset at img_size."""
    tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])
    if name == "mnist_usps":
        tf_gray = T.Compose([
            T.Resize(img_size), T.Grayscale(1), T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
            T.Normalize([0.5]*3, [0.5]*3),
        ])
        src = dsets.MNIST(root=DL.DATA_ROOT, train=True, download=True, transform=tf_gray)
        tgt = dsets.USPS(root=DL.DATA_ROOT, train=True, download=True, transform=tf_gray)
        idx_to_dir = {i: str(i) for i in range(10)}
    elif name == "svhn_mnist":
        tf_svhn = T.Compose([T.Resize(img_size), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
        tf_mnist = T.Compose([
            T.Resize(img_size), T.Grayscale(1), T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
            T.Normalize([0.5]*3, [0.5]*3),
        ])
        src = dsets.SVHN(root=DL.DATA_ROOT, split="train", download=True, transform=tf_svhn)
        tgt = dsets.MNIST(root=DL.DATA_ROOT, train=True, download=True, transform=tf_mnist)
        idx_to_dir = {i: str(i) for i in range(10)}
    else:
        folder_src = DL._image_folder_root(name, "source")
        folder_tgt = DL._image_folder_root(name, "target")
        src = dsets.ImageFolder(str(folder_src), transform=tf)
        tgt = dsets.ImageFolder(str(folder_tgt), transform=tf)
        idx_to_dir = {v: k for k, v in src.class_to_idx.items()}
    return src, tgt, idx_to_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DL.NUM_CLASSES))
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"FDA: dataset={args.dataset} beta={args.beta} img_size={args.img_size} device={device}")

    src_ds, tgt_ds, idx_to_dir = _build_source_and_target(args.dataset, args.img_size)
    print(f"source: {len(src_ds)}  target: {len(tgt_ds)}  classes: {len(idx_to_dir)}")

    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    # Preload a pool of target images for amplitude sampling
    tgt_loader = DataLoader(tgt_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)
    tgt_iter = iter(tgt_loader)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    for d in idx_to_dir.values():
        (out_root / d).mkdir(exist_ok=True)

    per_class = {k: 0 for k in idx_to_dir}
    total = 0
    with torch.no_grad():
        for src_x, src_y in src_loader:
            src_x = src_x.to(device)
            # Get a batch of target images for amplitude reference
            try:
                tgt_x, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tgt_x, _ = next(tgt_iter)
            tgt_x = tgt_x.to(device)
            # Match batch sizes (repeat target if smaller)
            if tgt_x.size(0) < src_x.size(0):
                reps = (src_x.size(0) // tgt_x.size(0)) + 1
                tgt_x = tgt_x.repeat(reps, 1, 1, 1)[:src_x.size(0)]
            else:
                tgt_x = tgt_x[:src_x.size(0)]

            adapted = fda_transfer(src_x, tgt_x, args.beta)
            adapted01 = (adapted + 1) / 2

            for i in range(adapted01.size(0)):
                label = int(src_y[i].item()) if isinstance(src_y[i], torch.Tensor) else src_y[i]
                dirname = idx_to_dir[label]
                idx = per_class[label]
                per_class[label] += 1
                save_image(adapted01[i], str(out_root / dirname / f"{idx:06d}.png"))
            total += adapted01.size(0)
            if total % 1024 == 0 or total == len(src_ds):
                print(f"  wrote {total}/{len(src_ds)}")

    print(f"DONE  wrote {total} images to {out_root}")


if __name__ == "__main__":
    main()
