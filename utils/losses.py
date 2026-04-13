"""Loss functions for CycleGAN."""
import torch
import torch.nn.functional as F


def lsgan_loss_D(real_pred, fake_pred):
    """LSGAN discriminator loss."""
    loss_real = torch.mean((real_pred - 1.0) ** 2)
    loss_fake = torch.mean(fake_pred ** 2)
    return 0.5 * (loss_real + loss_fake)


def lsgan_loss_G(fake_pred):
    """LSGAN generator loss (fool discriminator)."""
    return 0.5 * torch.mean((fake_pred - 1.0) ** 2)


def cycle_consistency_loss(real, reconstructed):
    """L1 cycle-consistency loss."""
    return F.l1_loss(real, reconstructed)


def identity_loss(real, same):
    """L1 identity loss: G_A2B(B) should ≈ B."""
    return F.l1_loss(real, same)
