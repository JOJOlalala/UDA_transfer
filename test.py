"""Generate translated images from a trained CycleGAN checkpoint."""
import argparse
import os

import yaml
import torch

from data.datasets import build_dataloader
from models.cyclegan import CycleGANTrainer
from models.spectral_cyclegan import SpectralCycleGANTrainer
from utils.visualization import save_comparison, save_spectral_decomposition, denormalize
from utils.spectral import fft_decompose
import torchvision.utils as vutils


def parse_args():
    parser = argparse.ArgumentParser(description="CycleGAN Inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", choices=["pixel", "spectral"], default="pixel")
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--n_images", type=int, default=16, help="Number of images to generate")
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ds_name = config["dataset"]["name"]
    exp_name = f"{ds_name}_{args.mode}"
    if args.mode == "spectral":
        exp_name += f"_beta{args.beta}"

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(base_dir, "results", exp_name, "test")
    os.makedirs(output_dir, exist_ok=True)

    # Build trainer and load checkpoint
    if args.mode == "pixel":
        trainer = CycleGANTrainer(config)
    else:
        trainer = SpectralCycleGANTrainer(config, beta=args.beta)

    trainer.load_checkpoint(args.checkpoint)
    trainer.set_eval()

    # Build test dataloader
    dataloader = build_dataloader(config, split="test")

    # Collect images
    all_A, all_B, all_fake_B, all_fake_A = [], [], [], []
    count = 0

    for real_A, real_B in dataloader:
        fake_B = trainer.translate_A2B(real_A)
        fake_A = trainer.translate_B2A(real_B)

        all_A.append(real_A)
        all_B.append(real_B)
        all_fake_B.append(fake_B.cpu())
        all_fake_A.append(fake_A.cpu())

        count += real_A.shape[0]
        if count >= args.n_images:
            break

    all_A = torch.cat(all_A)[:args.n_images]
    all_B = torch.cat(all_B)[:args.n_images]
    all_fake_B = torch.cat(all_fake_B)[:args.n_images]
    all_fake_A = torch.cat(all_fake_A)[:args.n_images]

    # Save comparison grid
    save_comparison(
        all_A, all_fake_B, all_B, all_fake_A,
        os.path.join(output_dir, "comparison.png"),
        n=args.n_images,
    )

    # Save individual translated images for FID
    fid_dir_B = os.path.join(output_dir, "fake_B")
    fid_dir_A = os.path.join(output_dir, "fake_A")
    os.makedirs(fid_dir_B, exist_ok=True)
    os.makedirs(fid_dir_A, exist_ok=True)

    for i in range(len(all_fake_B)):
        vutils.save_image(denormalize(all_fake_B[i]), os.path.join(fid_dir_B, f"{i:04d}.png"))
        vutils.save_image(denormalize(all_fake_A[i]), os.path.join(fid_dir_A, f"{i:04d}.png"))

    # For spectral mode: save decomposition visualization
    if args.mode == "spectral":
        real_A_dev = all_A[:4].to(trainer.device)
        low_A, high_A = fft_decompose(real_A_dev, args.beta)
        fake_low_B = trainer.G_AB(low_A)
        from utils.spectral import fft_recombine
        recombined = fft_recombine(fake_low_B, high_A)

        save_spectral_decomposition(
            real_A_dev.cpu(), low_A.cpu(), high_A.cpu(),
            fake_low_B.cpu(), recombined.cpu(),
            os.path.join(output_dir, "spectral_decomposition.png"),
        )

    print(f"Results saved to {output_dir}")
    print(f"  Comparison: {os.path.join(output_dir, 'comparison.png')}")
    if args.mode == "spectral":
        print(f"  Decomposition: {os.path.join(output_dir, 'spectral_decomposition.png')}")


if __name__ == "__main__":
    main()
