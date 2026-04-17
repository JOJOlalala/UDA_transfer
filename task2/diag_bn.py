"""BN-stats diagnostic: evaluate the saved M0 checkpoint on the target test
set in both eval() and train() modes. If train-mode dramatically outperforms
eval-mode, the gap is caused by stored BN running stats rather than a real
domain gap in the learned features.

Also prints per-channel mean/std over a batch of MNIST (source) vs USPS
(target) after the same transform, to check if preprocessing itself diverges.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task2.classifier import build_classifier
from task2 import data_labeled as DL


@torch.no_grad()
def acc_loss(model, loader, device):
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total, loss_sum / total


def tensor_stats(name, ds, n=512):
    loader = DL.build_loader(ds, batch_size=n, shuffle=False, num_workers=2)
    x, y = next(iter(loader))
    c_mean = x.mean(dim=(0, 2, 3))
    c_std = x.std(dim=(0, 2, 3))
    print(f"{name:8s}  n={n}  y.unique={torch.unique(y).tolist()}  "
          f"mean={c_mean.tolist()}  std={c_std.tolist()}  "
          f"range=({x.min():+.3f},{x.max():+.3f})", flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_name = "mnist_usps"
    ckpt_path = PROJECT_ROOT / "checkpoints_task2" / "m0_source_only" / ds_name / "latest.pth"
    state = torch.load(ckpt_path, map_location=device)
    print(f"loaded {ckpt_path}  arch={state['arch']}  saved_acc={state.get('target_acc')}", flush=True)

    num_classes = DL.NUM_CLASSES[ds_name]
    model = build_classifier(state["arch"], num_classes).to(device)
    model.load_state_dict(state["model"])

    target = DL.build_dataset(ds_name, domain="target", train=False)
    loader = DL.build_loader(target, batch_size=256, shuffle=False, num_workers=2)

    model.eval()
    a_eval, l_eval = acc_loss(model, loader, device)
    print(f"EVAL  mode: acc={a_eval:.4f}  loss={l_eval:.4f}", flush=True)

    model.train()
    a_train, l_train = acc_loss(model, loader, device)
    print(f"TRAIN mode: acc={a_train:.4f}  loss={l_train:.4f}", flush=True)

    src_ds = DL.build_dataset(ds_name, "source", train=True)
    tgt_ds = DL.build_dataset(ds_name, "target", train=False)
    tensor_stats("source", src_ds)
    tensor_stats("target", tgt_ds)

    verdict = "BN_STATS" if (a_train - a_eval) > 0.15 else "NOT_BN"
    print(f"VERDICT: {verdict}  (train_mode − eval_mode = {a_train - a_eval:+.4f})", flush=True)


if __name__ == "__main__":
    main()
