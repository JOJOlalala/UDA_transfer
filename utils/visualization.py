"""Visualization utilities for CycleGAN results."""
import os
import torch
import torchvision.utils as vutils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def denormalize(tensor):
    """Convert from [-1, 1] to [0, 1] for display."""
    return (tensor.clamp(-1, 1) + 1) / 2


def save_image_grid(images_dict, save_path, nrow=8):
    """Save a grid of named image sets.

    Args:
        images_dict: OrderedDict of {name: (B, C, H, W) tensor}
        save_path: path to save the figure
        nrow: images per row in each sub-grid
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n_sets = len(images_dict)
    fig, axes = plt.subplots(1, n_sets, figsize=(5 * n_sets, 5))
    if n_sets == 1:
        axes = [axes]

    for ax, (name, imgs) in zip(axes, images_dict.items()):
        imgs = denormalize(imgs.cpu())
        grid = vutils.make_grid(imgs[:nrow], nrow=min(nrow, 4), padding=2)
        ax.imshow(grid.permute(1, 2, 0).numpy())
        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_comparison(real_A, fake_B, real_B, fake_A, save_path, n=8):
    """Save side-by-side comparison: real_A | fake_B | real_B | fake_A."""
    from collections import OrderedDict
    images_dict = OrderedDict([
        ("Source (A)", real_A[:n]),
        ("Translated (A→B)", fake_B[:n]),
        ("Target (B)", real_B[:n]),
        ("Translated (B→A)", fake_A[:n]),
    ])
    save_image_grid(images_dict, save_path)


def save_spectral_decomposition(real, low, high, translated_low, recombined, save_path, n=4):
    """Save spectral decomposition visualization."""
    from collections import OrderedDict
    images_dict = OrderedDict([
        ("Original", real[:n]),
        ("Low-freq", low[:n]),
        ("High-freq", high[:n]),
        ("Translated low", translated_low[:n]),
        ("Recombined", recombined[:n]),
    ])
    save_image_grid(images_dict, save_path)


def plot_losses(loss_history, save_path):
    """Plot training loss curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(loss_history["G"]) + 1)

    ax1.plot(epochs, loss_history["G"], label="G total")
    ax1.plot(epochs, loss_history["G_AB"], label="G_AB")
    ax1.plot(epochs, loss_history["G_BA"], label="G_BA")
    ax1.plot(epochs, loss_history["cyc_A"], label="cyc_A")
    ax1.plot(epochs, loss_history["cyc_B"], label="cyc_B")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Generator Losses")
    ax1.legend()

    ax2.plot(epochs, loss_history["D_A"], label="D_A")
    ax2.plot(epochs, loss_history["D_B"], label="D_B")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Discriminator Losses")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
