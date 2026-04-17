"""Evaluate a saved classifier on the target test split."""
import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task2.classifier import build_classifier
from task2 import data_labeled as DL
from task2.train_classifier import evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", required=True,
                   choices=list(DL.NUM_CLASSES.keys()))
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(args.ckpt, map_location=device)
    arch = state["arch"]
    num_classes = state["num_classes"]
    model = build_classifier(arch, num_classes).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    target = DL.build_dataset(args.dataset, domain="target", train=False)
    loader = DL.build_loader(target, args.batch_size, shuffle=False)
    m = evaluate(model, loader, device)
    m.update({"ckpt": args.ckpt, "dataset": args.dataset, "arch": arch})
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
