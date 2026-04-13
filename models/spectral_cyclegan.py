"""Spectral CycleGAN — applies CycleGAN to low-frequency spectral bands."""
import torch
from models.cyclegan import CycleGANTrainer
from utils.spectral import fft_decompose, fft_recombine
from utils.losses import lsgan_loss_D, lsgan_loss_G, cycle_consistency_loss, identity_loss


class SpectralCycleGANTrainer(CycleGANTrainer):
    """CycleGAN that operates only on low-frequency spectral content.

    Pipeline:
      1. FFT decompose input → low-freq (spatial) + high-freq (spatial)
      2. CycleGAN generators translate only the low-freq images
      3. At inference: recombine translated low-freq with original high-freq
    """

    def __init__(self, config, beta=0.05):
        super().__init__(config)
        self.beta = beta

    def train_step(self, real_A, real_B):
        """Training step on low-frequency decomposed images."""
        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)

        # Decompose into low and high frequency
        low_A, high_A = fft_decompose(real_A, self.beta)
        low_B, high_B = fft_decompose(real_B, self.beta)

        # ---- Generator forward (on low-freq only) ----
        fake_low_B = self.G_AB(low_A)
        fake_low_A = self.G_BA(low_B)
        rec_low_A = self.G_BA(fake_low_B)
        rec_low_B = self.G_AB(fake_low_A)
        idt_low_A = self.G_BA(low_A)
        idt_low_B = self.G_AB(low_B)

        # Generator losses
        loss_G_AB = lsgan_loss_G(self.D_B(fake_low_B))
        loss_G_BA = lsgan_loss_G(self.D_A(fake_low_A))
        loss_cyc_A = cycle_consistency_loss(low_A, rec_low_A)
        loss_cyc_B = cycle_consistency_loss(low_B, rec_low_B)
        loss_idt_A = identity_loss(low_A, idt_low_A)
        loss_idt_B = identity_loss(low_B, idt_low_B)

        loss_G = (
            loss_G_AB + loss_G_BA
            + self.lambda_cycle * (loss_cyc_A + loss_cyc_B)
            + self.lambda_idt * (loss_idt_A + loss_idt_B)
        )

        self.opt_G.zero_grad()
        loss_G.backward()
        self.opt_G.step()

        # ---- Discriminator forward ----
        fake_low_A_pool = self.pool_A.query(fake_low_A.detach())
        loss_D_A = lsgan_loss_D(self.D_A(low_A), self.D_A(fake_low_A_pool))

        fake_low_B_pool = self.pool_B.query(fake_low_B.detach())
        loss_D_B = lsgan_loss_D(self.D_B(low_B), self.D_B(fake_low_B_pool))

        loss_D = loss_D_A + loss_D_B

        self.opt_D.zero_grad()
        loss_D.backward()
        self.opt_D.step()

        return {
            "G": loss_G.item(),
            "G_AB": loss_G_AB.item(),
            "G_BA": loss_G_BA.item(),
            "cyc_A": loss_cyc_A.item(),
            "cyc_B": loss_cyc_B.item(),
            "idt_A": loss_idt_A.item(),
            "idt_B": loss_idt_B.item(),
            "D_A": loss_D_A.item(),
            "D_B": loss_D_B.item(),
        }

    def translate_A2B(self, real_A):
        """Translate A -> B: decompose, translate low-freq, recombine."""
        with torch.no_grad():
            real_A = real_A.to(self.device)
            low_A, high_A = fft_decompose(real_A, self.beta)
            fake_low_B = self.G_AB(low_A)
            return fft_recombine(fake_low_B, high_A)

    def translate_B2A(self, real_B):
        """Translate B -> A: decompose, translate low-freq, recombine."""
        with torch.no_grad():
            real_B = real_B.to(self.device)
            low_B, high_B = fft_decompose(real_B, self.beta)
            fake_low_A = self.G_BA(low_B)
            return fft_recombine(fake_low_A, high_B)
