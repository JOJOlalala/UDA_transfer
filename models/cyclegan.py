"""CycleGAN trainer for pixel-space style transfer."""
import os
import itertools

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from models.generator import ResNetGenerator
from models.discriminator import PatchGANDiscriminator
from utils.losses import lsgan_loss_D, lsgan_loss_G, cycle_consistency_loss, identity_loss
from utils.image_pool import ImagePool


def init_weights(m):
    """Initialize weights with Normal(0, 0.02)."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("InstanceNorm") != -1:
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


class CycleGANTrainer:
    """Manages CycleGAN training in pixel space."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mc = config["model"]
        tc = config["training"]

        # Build models
        self.G_AB = ResNetGenerator(
            input_nc=3, output_nc=3,
            ngf=mc["ngf"], n_blocks=mc["n_blocks"],
            n_downsample=mc.get("n_downsample", 2),
        ).to(self.device)
        self.G_BA = ResNetGenerator(
            input_nc=3, output_nc=3,
            ngf=mc["ngf"], n_blocks=mc["n_blocks"],
            n_downsample=mc.get("n_downsample", 2),
        ).to(self.device)
        self.D_A = PatchGANDiscriminator(
            input_nc=3, ndf=mc["ndf"], n_layers=mc["n_layers_D"],
        ).to(self.device)
        self.D_B = PatchGANDiscriminator(
            input_nc=3, ndf=mc["ndf"], n_layers=mc["n_layers_D"],
        ).to(self.device)

        # Init weights
        self.G_AB.apply(init_weights)
        self.G_BA.apply(init_weights)
        self.D_A.apply(init_weights)
        self.D_B.apply(init_weights)

        # Optimizers
        self.opt_G = Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()),
            lr=tc["lr"], betas=(0.5, 0.999),
        )
        self.opt_D = Adam(
            itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
            lr=tc["lr"], betas=(0.5, 0.999),
        )

        # LR schedulers: constant for first half, linear decay to 0
        total_epochs = tc["epochs"]
        decay_start = tc["decay_epoch"]

        def lr_lambda(epoch):
            if epoch < decay_start:
                return 1.0
            return 1.0 - (epoch - decay_start) / (total_epochs - decay_start + 1)

        self.sched_G = LambdaLR(self.opt_G, lr_lambda)
        self.sched_D = LambdaLR(self.opt_D, lr_lambda)

        # Loss weights
        self.lambda_cycle = tc["lambda_cycle"]
        self.lambda_idt = tc["lambda_identity"]

        # Image pools
        self.pool_A = ImagePool(tc.get("pool_size", 50))
        self.pool_B = ImagePool(tc.get("pool_size", 50))

    def train_step(self, real_A, real_B):
        """One training step. Returns loss dict."""
        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)

        # ---- Generator forward ----
        fake_B = self.G_AB(real_A)
        fake_A = self.G_BA(real_B)
        rec_A = self.G_BA(fake_B)
        rec_B = self.G_AB(fake_A)
        idt_A = self.G_BA(real_A)
        idt_B = self.G_AB(real_B)

        # Generator losses
        loss_G_AB = lsgan_loss_G(self.D_B(fake_B))
        loss_G_BA = lsgan_loss_G(self.D_A(fake_A))
        loss_cyc_A = cycle_consistency_loss(real_A, rec_A)
        loss_cyc_B = cycle_consistency_loss(real_B, rec_B)
        loss_idt_A = identity_loss(real_A, idt_A)
        loss_idt_B = identity_loss(real_B, idt_B)

        loss_G = (
            loss_G_AB + loss_G_BA
            + self.lambda_cycle * (loss_cyc_A + loss_cyc_B)
            + self.lambda_idt * (loss_idt_A + loss_idt_B)
        )

        self.opt_G.zero_grad()
        loss_G.backward()
        self.opt_G.step()

        # ---- Discriminator forward ----
        # D_A: distinguish real A from fake A
        fake_A_pool = self.pool_A.query(fake_A.detach())
        loss_D_A = lsgan_loss_D(self.D_A(real_A), self.D_A(fake_A_pool))

        # D_B: distinguish real B from fake B
        fake_B_pool = self.pool_B.query(fake_B.detach())
        loss_D_B = lsgan_loss_D(self.D_B(real_B), self.D_B(fake_B_pool))

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

    def step_schedulers(self):
        """Step LR schedulers at end of epoch."""
        self.sched_G.step()
        self.sched_D.step()

    def translate_A2B(self, real_A):
        """Translate domain A -> B."""
        with torch.no_grad():
            return self.G_AB(real_A.to(self.device))

    def translate_B2A(self, real_B):
        """Translate domain B -> A."""
        with torch.no_grad():
            return self.G_BA(real_B.to(self.device))

    def save_checkpoint(self, epoch, save_dir):
        """Save model checkpoint."""
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "G_AB": self.G_AB.state_dict(),
            "G_BA": self.G_BA.state_dict(),
            "D_A": self.D_A.state_dict(),
            "D_B": self.D_B.state_dict(),
            "opt_G": self.opt_G.state_dict(),
            "opt_D": self.opt_D.state_dict(),
        }, os.path.join(save_dir, f"ckpt_epoch{epoch:04d}.pth"))

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.G_AB.load_state_dict(ckpt["G_AB"])
        self.G_BA.load_state_dict(ckpt["G_BA"])
        self.D_A.load_state_dict(ckpt["D_A"])
        self.D_B.load_state_dict(ckpt["D_B"])
        if "opt_G" in ckpt:
            self.opt_G.load_state_dict(ckpt["opt_G"])
            self.opt_D.load_state_dict(ckpt["opt_D"])
        return ckpt.get("epoch", 0)

    def set_train(self):
        self.G_AB.train()
        self.G_BA.train()
        self.D_A.train()
        self.D_B.train()

    def set_eval(self):
        self.G_AB.eval()
        self.G_BA.eval()
        self.D_A.eval()
        self.D_B.eval()
