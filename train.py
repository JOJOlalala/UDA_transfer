"""Main training script for CycleGAN (pixel and spectral modes)."""
import argparse
import os
import sys
import time
from collections import defaultdict

import yaml
import torch

from data.datasets import build_dataloader
from models.cyclegan import CycleGANTrainer
from models.spectral_cyclegan import SpectralCycleGANTrainer
from utils.visualization import save_comparison, plot_losses


def parse_args():
    parser = argparse.ArgumentParser(description="CycleGAN Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--mode", choices=["pixel", "spectral"], default="pixel")
    parser.add_argument("--beta", type=float, default=0.05, help="Spectral cutoff (spectral mode)")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    ds_name = config["dataset"]["name"]
    tc = config["training"]
    total_epochs = tc["epochs"]

    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    exp_name = f"{ds_name}_{args.mode}"
    if args.mode == "spectral":
        exp_name += f"_beta{args.beta}"
    ckpt_dir = os.path.join(base_dir, "checkpoints", exp_name)
    result_dir = os.path.join(base_dir, "results", exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # Build dataloader
    dataloader = build_dataloader(config, split="train")
    print(f"Dataset: {ds_name} | Mode: {args.mode} | Samples: {len(dataloader.dataset)}")

    # Build trainer
    if args.mode == "pixel":
        trainer = CycleGANTrainer(config)
    else:
        trainer = SpectralCycleGANTrainer(config, beta=args.beta)

    # Resume
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume) + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    loss_history = defaultdict(list)
    vis_freq = tc.get("vis_freq", 10)
    save_freq = tc.get("save_freq", 20)

    for epoch in range(start_epoch, total_epochs):
        trainer.set_train()
        epoch_losses = defaultdict(float)
        n_steps = 0
        t0 = time.time()

        for real_A, real_B in dataloader:
            losses = trainer.train_step(real_A, real_B)
            for k, v in losses.items():
                epoch_losses[k] += v
            n_steps += 1

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= max(n_steps, 1)
            loss_history[k].append(epoch_losses[k])

        elapsed = time.time() - t0
        lr = trainer.opt_G.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch+1}/{total_epochs}] "
            f"G={epoch_losses['G']:.4f} D_A={epoch_losses['D_A']:.4f} "
            f"D_B={epoch_losses['D_B']:.4f} cyc={epoch_losses['cyc_A']+epoch_losses['cyc_B']:.4f} "
            f"lr={lr:.6f} ({elapsed:.1f}s)"
        )

        # Step LR
        trainer.step_schedulers()

        # Save visualizations
        if (epoch + 1) % vis_freq == 0 or epoch == 0:
            trainer.set_eval()
            # Grab a batch for visualization
            vis_A, vis_B = next(iter(dataloader))
            vis_A = vis_A.to(trainer.device)
            vis_B = vis_B.to(trainer.device)
            fake_B = trainer.translate_A2B(vis_A)
            fake_A = trainer.translate_B2A(vis_B)
            save_comparison(
                vis_A, fake_B, vis_B, fake_A,
                os.path.join(result_dir, f"epoch{epoch+1:04d}.png"),
            )

        # Save checkpoint
        if (epoch + 1) % save_freq == 0 or epoch == total_epochs - 1:
            trainer.save_checkpoint(epoch + 1, ckpt_dir)

    # Save loss curves
    plot_losses(dict(loss_history), os.path.join(result_dir, "loss_curves.png"))
    print(f"\nTraining complete. Results: {result_dir}")


if __name__ == "__main__":
    main()
