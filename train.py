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
from utils.visualization import save_comparison, plot_losses, denormalize


def parse_args():
    parser = argparse.ArgumentParser(description="CycleGAN Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--mode", choices=["pixel", "spectral"], default="pixel")
    parser.add_argument("--beta", type=float, default=0.05, help="Spectral cutoff (spectral mode)")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--suffix", type=str, default=None, help="Append to exp_name (e.g. '_unet')")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="UDA-CycleGAN", help="wandb project name")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI overrides
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    ds_name = config["dataset"]["name"]
    tc = config["training"]
    total_epochs = tc["epochs"]

    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    exp_name = f"{ds_name}_{args.mode}"
    if args.mode == "spectral":
        exp_name += f"_beta{args.beta}"
    if args.suffix:
        exp_name += args.suffix
    ckpt_dir = os.path.join(base_dir, "checkpoints", exp_name)
    result_dir = os.path.join(base_dir, "results", exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # --- wandb init ---
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            os.environ.setdefault("WANDB_INIT_TIMEOUT", "30")
            os.environ.setdefault("WANDB_MODE", "offline")
            wandb.init(
                project=args.wandb_project,
                name=exp_name,
                config={
                    "dataset": ds_name,
                    "mode": args.mode,
                    "beta": args.beta if args.mode == "spectral" else None,
                    **config["model"],
                    **config["training"],
                },
                tags=[ds_name, args.mode],
            )
        except (ImportError, Exception) as e:
            print(f"wandb init failed ({e}), logging disabled.")
            use_wandb = False

    # Log file on shared storage (readable without SSH to compute node)
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    log_fh = open(log_file, "a")

    def log(msg):
        print(msg, flush=True)
        log_fh.write(msg + "\n")
        log_fh.flush()

    # Build dataloader
    dataloader = build_dataloader(config, split="train")
    log(f"Dataset: {ds_name} | Mode: {args.mode} | Samples: {len(dataloader.dataset)}")

    # Build trainer
    if args.mode == "pixel":
        trainer = CycleGANTrainer(config)
    else:
        trainer = SpectralCycleGANTrainer(config, beta=args.beta)

    # Resume
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume) + 1
        trainer.sched_G.last_epoch = start_epoch - 1
        trainer.sched_D.last_epoch = start_epoch - 1
        log(f"Resumed from epoch {start_epoch}")

    # Training loop — step-based logging/vis/checkpoint
    loss_history = defaultdict(list)
    log_step_freq = 10       # log loss every N steps
    vis_step_freq = 500      # vis + checkpoint every N steps
    global_step = 0
    steps_per_epoch = len(dataloader)

    # Pre-fetch a fixed vis batch (same images throughout training)
    vis_A, vis_B = next(iter(dataloader))
    vis_A = vis_A[:8].to(trainer.device)
    vis_B = vis_B[:8].to(trainer.device)

    log(f"Steps/epoch: {steps_per_epoch} | Total steps: {steps_per_epoch * total_epochs}")

    for epoch in range(start_epoch, total_epochs):
        trainer.set_train()
        t0 = time.time()

        for real_A, real_B in dataloader:
            losses = trainer.train_step(real_A, real_B)
            global_step += 1

            # Loss logging every 10 steps
            if global_step % log_step_freq == 0:
                lr = trainer.opt_G.param_groups[0]["lr"]
                log(
                    f"[step {global_step}] "
                    f"G={losses['G']:.4f} D_A={losses['D_A']:.4f} "
                    f"D_B={losses['D_B']:.4f} cyc={losses['cyc_A']+losses['cyc_B']:.4f} "
                    f"lr={lr:.6f}"
                )
                if use_wandb:
                    log_dict = {f"loss/{k}": v for k, v in losses.items()}
                    log_dict["lr"] = lr
                    wandb.log(log_dict, step=global_step)

            # Record for loss curves
            for k, v in losses.items():
                loss_history[k].append(v)

            # Vis + checkpoint every 500 steps
            if global_step % vis_step_freq == 0:
                trainer.set_eval()
                fake_B = trainer.translate_A2B(vis_A)
                fake_A = trainer.translate_B2A(vis_B)
                save_path = os.path.join(result_dir, f"step{global_step:06d}.png")
                save_comparison(vis_A, fake_B, vis_B, fake_A, save_path)

                # Overwrite single checkpoint
                ckpt_path = os.path.join(ckpt_dir, "latest.pth")
                torch.save({
                    "global_step": global_step,
                    "epoch": epoch,
                    "G_AB": trainer.G_AB.state_dict(),
                    "G_BA": trainer.G_BA.state_dict(),
                    "D_A": trainer.D_A.state_dict(),
                    "D_B": trainer.D_B.state_dict(),
                    "opt_G": trainer.opt_G.state_dict(),
                    "opt_D": trainer.opt_D.state_dict(),
                }, ckpt_path)

                log(f"  [step {global_step}] vis + checkpoint saved")

                if use_wandb:
                    import torchvision.utils as vutils
                    grid_AB = vutils.make_grid(
                        torch.cat([denormalize(vis_A[:4].cpu()), denormalize(fake_B[:4].cpu())]),
                        nrow=4, padding=2,
                    )
                    wandb.log({
                        "images/A_to_B": wandb.Image(grid_AB),
                    }, step=global_step)

                trainer.set_train()

        # Step LR at end of epoch
        elapsed = time.time() - t0
        trainer.step_schedulers()
        log(f"[Epoch {epoch+1}/{total_epochs}] done ({elapsed:.1f}s)")

    # Save final checkpoint + loss curves
    torch.save({
        "global_step": global_step,
        "epoch": total_epochs,
        "G_AB": trainer.G_AB.state_dict(),
        "G_BA": trainer.G_BA.state_dict(),
        "D_A": trainer.D_A.state_dict(),
        "D_B": trainer.D_B.state_dict(),
        "opt_G": trainer.opt_G.state_dict(),
        "opt_D": trainer.opt_D.state_dict(),
    }, os.path.join(ckpt_dir, "latest.pth"))

    # Subsample loss history for plot (every 10th entry to keep it clean)
    plot_history = {k: v[::10] for k, v in loss_history.items()}
    plot_losses(plot_history, os.path.join(result_dir, "loss_curves.png"))
    log(f"\nTraining complete. {global_step} steps. Results: {result_dir}")
    log_fh.close()

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
