"""Generic classifier trainer for Task II.

Supports three training-data sources:
    --data source     : labeled source train split (M0, M1-oracle variant)
    --data target     : labeled target train split (M1 target-only oracle)
    --data translated : pre-translated source images from M2/M3/M5

Evaluation at the end always runs on the labeled target test split.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task2.classifier import build_classifier
from task2 import data_labeled as DL


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=list(DL.NUM_CLASSES.keys()))
    p.add_argument("--data", default="source",
                   choices=["source", "target", "translated"])
    p.add_argument("--translated-dir", default=None,
                   help="Root dir for --data translated (class subfolders).")
    p.add_argument("--arch", default=None,
                   help="Override auto arch for the dataset.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True,
                   help="Output dir for checkpoint + metrics.json.")
    p.add_argument("--tag", default=None,
                   help="Optional label written into metrics.json.")
    return p.parse_args()


def default_hparams(arch: str) -> dict:
    if arch == "smallcnn":
        return {"epochs": 20, "batch_size": 256, "lr": 1e-3,
                "wd": 5e-4, "optimizer": "adam"}
    if arch == "resnet18":
        return {"epochs": 30, "batch_size": 64, "lr": 1e-4,
                "wd": 1e-4, "optimizer": "adam"}
    if arch == "resnet50":
        return {"epochs": 30, "batch_size": 32, "lr": 1e-4,
                "wd": 1e-4, "optimizer": "adam"}
    if arch == "resnet18_cifar":
        return {"epochs": 30, "batch_size": 64, "lr": 1e-4,
                "wd": 1e-4, "optimizer": "adam"}
    raise ValueError(arch)


def build_optimizer(model: nn.Module, hp: dict) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=hp["lr"],
                            weight_decay=hp["wd"])


def evaluate(model: nn.Module, loader, device) -> dict:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return {"acc": correct / total, "loss": loss_sum / total, "n": total}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    arch = args.arch or DL.ARCH[args.dataset]
    num_classes = DL.NUM_CLASSES[args.dataset]
    hp = default_hparams(arch)
    if args.epochs is not None:      hp["epochs"] = args.epochs
    if args.batch_size is not None:  hp["batch_size"] = args.batch_size
    if args.lr is not None:          hp["lr"] = args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  arch={arch}  num_classes={num_classes}")
    print(f"hparams={hp}")

    # --- datasets ---
    if args.data == "source":
        train_ds = DL.build_dataset(args.dataset, domain="source", train=True)
    elif args.data == "target":
        train_ds = DL.build_dataset(args.dataset, domain="target", train=True)
    else:
        assert args.translated_dir, "--translated-dir required for --data translated"
        source_full = DL.build_dataset(args.dataset, domain="source", train=True)
        train_ds = DL.build_translated_dataset(
            args.dataset, args.translated_dir, source_full
        )
    target_test = DL.build_dataset(args.dataset, domain="target", train=False)

    train_loader = DL.build_loader(train_ds, hp["batch_size"], shuffle=True)
    target_loader = DL.build_loader(target_test, 256, shuffle=False)
    print(f"train: n={len(train_ds)}  target_test: n={len(target_test)}")

    # --- model ---
    model = build_classifier(arch, num_classes).to(device)
    opt = build_optimizer(model, hp)
    sched = CosineAnnealingLR(opt, T_max=hp["epochs"])

    # --- train ---
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    best_target_acc = -1.0
    log = []
    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        running = 0.0
        n_seen = 0
        correct = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item() * y.numel()
            n_seen += y.numel()
            correct += (logits.argmax(1) == y).sum().item()
        sched.step()

        tgt = evaluate(model, target_loader, device)
        train_acc = correct / n_seen
        train_loss = running / n_seen
        elapsed = time.time() - start
        print(f"[Ep {epoch:02d}/{hp['epochs']}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"tgt_acc={tgt['acc']:.4f} tgt_loss={tgt['loss']:.4f} "
              f"lr={sched.get_last_lr()[0]:.2e}  {elapsed:.0f}s")
        log.append({"epoch": epoch, "train_loss": train_loss,
                    "train_acc": train_acc, "target_acc": tgt["acc"],
                    "target_loss": tgt["loss"]})
        if tgt["acc"] > best_target_acc:
            best_target_acc = tgt["acc"]
            torch.save({"model": model.state_dict(),
                        "arch": arch,
                        "num_classes": num_classes,
                        "epoch": epoch,
                        "target_acc": tgt["acc"]},
                       out_dir / "best.pth")

    torch.save({"model": model.state_dict(),
                "arch": arch,
                "num_classes": num_classes,
                "epoch": hp["epochs"],
                "target_acc": tgt["acc"]},
               out_dir / "latest.pth")

    metrics = {
        "dataset": args.dataset,
        "arch": arch,
        "data": args.data,
        "tag": args.tag,
        "hparams": hp,
        "final_target_acc": tgt["acc"],
        "best_target_acc": best_target_acc,
        "train_log": log,
        "wall_seconds": time.time() - start,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"DONE  final_tgt={tgt['acc']:.4f}  best_tgt={best_target_acc:.4f}  "
          f"out={out_dir}")


if __name__ == "__main__":
    main()
