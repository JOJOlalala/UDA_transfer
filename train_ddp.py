"""DDP training script for CycleGAN (pixel and spectral modes).

Launch with: torchrun --nproc_per_node=N train_ddp.py --config ...
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.datasets import build_dataset
from models.cyclegan import CycleGANTrainer
from models.spectral_cyclegan import SpectralCycleGANTrainer
from utils.visualization import save_comparison, plot_losses, denormalize


def parse_args():
    parser = argparse.ArgumentParser(description="CycleGAN DDP Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", choices=["pixel", "spectral"], default="pixel")
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="UDA-CycleGAN")
    return parser.parse_args()


def main():
    args = parse_args()

    # DDP setup
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    is_main = (rank == 0)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

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
    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
    dist.barrier()

    # Logging (rank 0 only)
    log_fh = None
    if is_main:
        log_dir = os.path.join(base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{exp_name}.log")
        log_fh = open(log_file, "a")

    def log(msg):
        if is_main:
            print(msg, flush=True)
            log_fh.write(msg + "\n")
            log_fh.flush()

    log(f"DDP: {world_size} GPUs, rank={rank}, device={device}")

    # Build dataset + distributed sampler
    dataset = build_dataset(config, split="train")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=tc["batch_size"],
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    log(f"Dataset: {ds_name} | Mode: {args.mode} | Samples: {len(dataset)} | "
        f"BS/GPU: {tc['batch_size']} | Effective BS: {tc['batch_size'] * world_size}")

    # Build trainer (models on local device)
    if args.mode == "pixel":
        trainer = CycleGANTrainer(config)
    else:
        trainer = SpectralCycleGANTrainer(config, beta=args.beta)

    # Move models to correct device and wrap with DDP
    trainer.device = device
    for name in ["G_AB", "G_BA", "D_A", "D_B"]:
        model = getattr(trainer, name).to(device)
        ddp_model = DDP(model, device_ids=[local_rank])
        setattr(trainer, name, ddp_model)

    # Rebuild optimizers after DDP wrapping (parameters changed)
    import itertools
    trainer.opt_G = torch.optim.Adam(
        itertools.chain(trainer.G_AB.parameters(), trainer.G_BA.parameters()),
        lr=tc["lr"], betas=(0.5, 0.999),
    )
    trainer.opt_D = torch.optim.Adam(
        itertools.chain(trainer.D_A.parameters(), trainer.D_B.parameters()),
        lr=tc["lr"], betas=(0.5, 0.999),
    )

    # Rebuild LR schedulers
    decay_start = tc["decay_epoch"]
    def lr_lambda(epoch):
        if epoch < decay_start:
            return 1.0
        return 1.0 - (epoch - decay_start) / (total_epochs - decay_start + 1)
    trainer.sched_G = torch.optim.lr_scheduler.LambdaLR(trainer.opt_G, lr_lambda)
    trainer.sched_D = torch.optim.lr_scheduler.LambdaLR(trainer.opt_D, lr_lambda)

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        for name in ["G_AB", "G_BA", "D_A", "D_B"]:
            getattr(trainer, name).module.load_state_dict(ckpt[name])
        if "opt_G" in ckpt:
            trainer.opt_G.load_state_dict(ckpt["opt_G"])
            trainer.opt_D.load_state_dict(ckpt["opt_D"])
        start_epoch = ckpt.get("epoch", 0) + 1
        trainer.sched_G.last_epoch = start_epoch - 1
        trainer.sched_D.last_epoch = start_epoch - 1
        log(f"Resumed from epoch {start_epoch}")

    # Training loop
    loss_history = defaultdict(list)
    log_step_freq = 10
    vis_step_freq = 500
    global_step = 0
    steps_per_epoch = len(dataloader)

    # Vis batch (rank 0 only, fixed)
    vis_iter = iter(dataloader)
    vis_A, vis_B = next(vis_iter)
    vis_A = vis_A[:8].to(device)
    vis_B = vis_B[:8].to(device)

    log(f"Steps/epoch: {steps_per_epoch} | Total steps: {steps_per_epoch * total_epochs}")

    for epoch in range(start_epoch, total_epochs):
        sampler.set_epoch(epoch)
        trainer.set_train()
        t0 = time.time()

        for real_A, real_B in dataloader:
            losses = trainer.train_step(real_A, real_B)
            global_step += 1

            if global_step % log_step_freq == 0:
                lr = trainer.opt_G.param_groups[0]["lr"]
                log(
                    f"[step {global_step}] "
                    f"G={losses['G']:.4f} D_A={losses['D_A']:.4f} "
                    f"D_B={losses['D_B']:.4f} cyc={losses['cyc_A']+losses['cyc_B']:.4f} "
                    f"lr={lr:.6f}"
                )

            for k, v in losses.items():
                loss_history[k].append(v)

            # Vis + checkpoint (rank 0 only)
            if global_step % vis_step_freq == 0 and is_main:
                trainer.set_eval()
                with torch.no_grad():
                    fake_B = trainer.G_AB(vis_A)
                    fake_A = trainer.G_BA(vis_B)
                save_path = os.path.join(result_dir, f"step{global_step:06d}.png")
                save_comparison(vis_A, fake_B, vis_B, fake_A, save_path)

                ckpt_path = os.path.join(ckpt_dir, "latest.pth")
                torch.save({
                    "global_step": global_step,
                    "epoch": epoch,
                    "G_AB": trainer.G_AB.module.state_dict(),
                    "G_BA": trainer.G_BA.module.state_dict(),
                    "D_A": trainer.D_A.module.state_dict(),
                    "D_B": trainer.D_B.module.state_dict(),
                    "opt_G": trainer.opt_G.state_dict(),
                    "opt_D": trainer.opt_D.state_dict(),
                }, ckpt_path)
                log(f"  [step {global_step}] vis + checkpoint saved")
                trainer.set_train()

        elapsed = time.time() - t0
        trainer.step_schedulers()
        log(f"[Epoch {epoch+1}/{total_epochs}] done ({elapsed:.1f}s)")

    # Final save (rank 0)
    if is_main:
        torch.save({
            "global_step": global_step,
            "epoch": total_epochs,
            "G_AB": trainer.G_AB.module.state_dict(),
            "G_BA": trainer.G_BA.module.state_dict(),
            "D_A": trainer.D_A.module.state_dict(),
            "D_B": trainer.D_B.module.state_dict(),
            "opt_G": trainer.opt_G.state_dict(),
            "opt_D": trainer.opt_D.state_dict(),
        }, os.path.join(ckpt_dir, "latest.pth"))

        plot_history = {k: v[::10] for k, v in loss_history.items()}
        plot_losses(plot_history, os.path.join(result_dir, "loss_curves.png"))
        log(f"\nTraining complete. {global_step} steps. Results: {result_dir}")
        log_fh.close()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
