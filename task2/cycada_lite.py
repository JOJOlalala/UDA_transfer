"""CyCADA: fine-tune a pretrained CycleGAN with semantic + feature-level losses.

Full CyCADA (Hoffman et al., ICML 2018):
  1. Pixel-level GAN loss (CycleGAN)
  2. Cycle consistency loss
  3. Semantic consistency: CE(f(G_AB(x_s)), y_s) — preserve class predictions
  4. Feature-level adaptation: discriminator on classifier features aligns
     translated-source feature distribution with real-target features

Pipeline:
  1. Load pretrained CycleGAN (pixel mode) + frozen M0 classifier
  2. Fine-tune with all 4 losses
  3. Save fine-tuned checkpoint for downstream translate + classify
"""
import argparse
import itertools
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from models.cyclegan import CycleGANTrainer
from task2.classifier import build_classifier
from task2 import data_labeled as DL
from data.datasets import build_dataset
from utils.losses import lsgan_loss_D, lsgan_loss_G, cycle_consistency_loss
from utils.image_pool import ImagePool


class FeatureDiscriminator(nn.Module):
    """Small MLP discriminator on classifier features.
    Distinguishes translated-source features from real-target features.
    """
    def __init__(self, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x)


class FeatureExtractor(nn.Module):
    """Wrapper to extract penultimate features from a frozen classifier."""
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
        self._features = None
        # Hook into avgpool output (works for ResNet-18/50)
        self.classifier.avgpool.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self._features = out.flatten(1)

    def forward(self, x):
        logits = self.classifier(x)
        return logits, self._features


class CyCADATrainer:
    """Full CyCADA: CycleGAN + semantic loss + feature-level discriminator."""

    def __init__(self, cyclegan_trainer, classifier, classifier_res,
                 lambda_sem=1.0, lambda_feat=0.1, lambda_gan=1.0):
        self.trainer = cyclegan_trainer
        self.device = cyclegan_trainer.device
        self.lambda_gan = lambda_gan

        # Frozen classifier with feature extraction
        classifier = classifier.to(self.device)
        classifier.eval()
        for p in classifier.parameters():
            p.requires_grad_(False)
        self.feat_extractor = FeatureExtractor(classifier)

        # Feature discriminator
        # ResNet-18: feat_dim=512, ResNet-50: feat_dim=2048
        with torch.no_grad():
            dummy = torch.randn(1, 3, classifier_res, classifier_res, device=self.device)
            _, dummy_feat = self.feat_extractor(dummy)
            feat_dim = dummy_feat.shape[1]
        self.D_feat = FeatureDiscriminator(feat_dim).to(self.device)
        self.opt_D_feat = torch.optim.Adam(self.D_feat.parameters(), lr=1e-4, betas=(0.5, 0.999))

        self.lambda_sem = lambda_sem
        self.lambda_feat = lambda_feat
        self.classifier_res = classifier_res
        self._needs_resize = True

    def _prep_for_classifier(self, images):
        if self._needs_resize:
            return F.interpolate(images, size=self.classifier_res, mode='bilinear',
                                 align_corners=False)
        return images

    def train_step(self, real_A, real_B, labels_A):
        """One training step with semantic + feature-level losses.

        Args:
            real_A: source images (B, 3, H, W) in [-1, 1]
            real_B: target images (B, 3, H, W) in [-1, 1]
            labels_A: source labels (B,) — integer class indices
        """
        t = self.trainer
        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)
        labels_A = labels_A.to(self.device)

        # ---- Generator forward ----
        fake_B = t.G_AB(real_A)
        fake_A = t.G_BA(real_B)
        rec_A = t.G_BA(fake_B)
        rec_B = t.G_AB(fake_A)

        # Pixel GAN losses
        loss_G_AB = lsgan_loss_G(t.D_B(fake_B))
        loss_G_BA = lsgan_loss_G(t.D_A(fake_A))
        loss_cyc_A = cycle_consistency_loss(real_A, rec_A)
        loss_cyc_B = cycle_consistency_loss(real_B, rec_B)

        # Semantic loss: classifier(G_AB(source)) should predict source label
        fake_B_cls = self._prep_for_classifier(fake_B)
        logits, fake_feat = self.feat_extractor(fake_B_cls)
        loss_sem = F.cross_entropy(logits, labels_A)

        # Feature-level GAN loss: fool D_feat (make translated-source look like target)
        loss_feat_G = lsgan_loss_G(self.D_feat(fake_feat))

        loss_G = (
            self.lambda_gan * (loss_G_AB + loss_G_BA
                               + t.lambda_cycle * (loss_cyc_A + loss_cyc_B))
            + self.lambda_sem * loss_sem
            + self.lambda_feat * loss_feat_G
        )

        t.opt_G.zero_grad()
        loss_G.backward()
        t.opt_G.step()

        # ---- Pixel Discriminator forward ----
        if self.lambda_gan > 0:
            fake_A_pool = t.pool_A.query(fake_A.detach())
            loss_D_A = lsgan_loss_D(t.D_A(real_A), t.D_A(fake_A_pool))
            fake_B_pool = t.pool_B.query(fake_B.detach())
            loss_D_B = lsgan_loss_D(t.D_B(real_B), t.D_B(fake_B_pool))
            loss_D = loss_D_A + loss_D_B

            t.opt_D.zero_grad()
            loss_D.backward()
            t.opt_D.step()
        else:
            loss_D_A = loss_D_B = torch.tensor(0.0)

        # ---- Feature Discriminator forward ----
        real_B_cls = self._prep_for_classifier(real_B)
        with torch.no_grad():
            _, real_feat = self.feat_extractor(real_B_cls)
            _, fake_feat_d = self.feat_extractor(fake_B_cls.detach())
        loss_D_feat = lsgan_loss_D(self.D_feat(real_feat), self.D_feat(fake_feat_d))

        self.opt_D_feat.zero_grad()
        loss_D_feat.backward()
        self.opt_D_feat.step()

        # Semantic accuracy for logging
        with torch.no_grad():
            sem_acc = (logits.argmax(1) == labels_A).float().mean().item()

        return {
            "G": loss_G.item(),
            "G_AB": loss_G_AB.item(),
            "G_BA": loss_G_BA.item(),
            "cyc": (loss_cyc_A + loss_cyc_B).item(),
            "sem": loss_sem.item(),
            "sem_acc": sem_acc,
            "feat_G": loss_feat_G.item(),
            "D_feat": loss_D_feat.item(),
            "D_A": loss_D_A.item(),
            "D_B": loss_D_B.item(),
        }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DL.NUM_CLASSES))
    p.add_argument("--cyclegan-ckpt", required=True, help="Pretrained CycleGAN checkpoint")
    p.add_argument("--cyclegan-config", required=True, help="CycleGAN config YAML")
    p.add_argument("--classifier-ckpt", required=True, help="Frozen M0 classifier checkpoint")
    p.add_argument("--classifier-arch", default="resnet50")
    p.add_argument("--classifier-res", type=int, default=224)
    p.add_argument("--lambda-sem", type=float, default=1.0)
    p.add_argument("--lambda-feat", type=float, default=0.1)
    p.add_argument("--lambda-gan", type=float, default=1.0,
                   help="Scale pixel GAN + cycle losses (0 = semantic-only CyCADA)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.0001, help="Fine-tuning LR (lower than initial)")
    p.add_argument("--out", required=True, help="Output dir for fine-tuned checkpoint")
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CycleGAN config + trainer
    with open(args.cyclegan_config) as f:
        config = yaml.safe_load(f)
    # Override training params for fine-tuning
    config["training"]["lr"] = args.lr
    config["training"]["epochs"] = args.epochs
    config["training"]["decay_epoch"] = args.epochs  # no decay during fine-tuning
    config["training"]["lambda_identity"] = 0.0

    trainer = CycleGANTrainer(config)
    trainer.load_checkpoint(args.cyclegan_ckpt)
    print(f"Loaded CycleGAN from {args.cyclegan_ckpt}")

    # Reset optimizer with fine-tuning LR
    trainer.opt_G = torch.optim.Adam(
        itertools.chain(trainer.G_AB.parameters(), trainer.G_BA.parameters()),
        lr=args.lr, betas=(0.5, 0.999),
    )
    trainer.opt_D = torch.optim.Adam(
        itertools.chain(trainer.D_A.parameters(), trainer.D_B.parameters()),
        lr=args.lr, betas=(0.5, 0.999),
    )

    # Load frozen classifier
    num_classes = DL.NUM_CLASSES[args.dataset]
    classifier = build_classifier(args.classifier_arch, num_classes)
    cls_ckpt = torch.load(args.classifier_ckpt, map_location=device, weights_only=False)
    cls_state = cls_ckpt["model"] if "model" in cls_ckpt else cls_ckpt
    classifier.load_state_dict(cls_state)
    print(f"Loaded frozen classifier ({args.classifier_arch}, {num_classes} classes)")

    # Build CyCADA trainer (full: semantic + feature discriminator)
    cycada = CyCADATrainer(trainer, classifier, args.classifier_res,
                           args.lambda_sem, args.lambda_feat, args.lambda_gan)

    # Build data: unpaired CycleGAN data + labeled source
    cyclegan_img_size = config["dataset"]["img_size"]
    unpaired_loader = DataLoader(
        build_dataset(config, "train"),
        batch_size=config["training"]["batch_size"],
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
    )
    # Labeled source at CycleGAN resolution for semantic loss
    from torchvision import transforms as T, datasets as dsets
    if args.dataset in ("mnist_usps", "svhn_mnist"):
        # Digits: use CycleGAN-resolution labeled source
        if args.dataset == "mnist_usps":
            tf = T.Compose([T.Resize(cyclegan_img_size), T.Grayscale(1), T.ToTensor(),
                            T.Lambda(lambda x: x.repeat(3,1,1)), T.Normalize([.5]*3,[.5]*3)])
            label_ds = dsets.MNIST(root=DL.DATA_ROOT, train=True, download=True, transform=tf)
        else:
            tf = T.Compose([T.Resize(cyclegan_img_size), T.ToTensor(), T.Normalize([.5]*3,[.5]*3)])
            label_ds = dsets.SVHN(root=DL.DATA_ROOT, split="train", download=True, transform=tf)
    else:
        tf = T.Compose([T.Resize((cyclegan_img_size, cyclegan_img_size)), T.RandomHorizontalFlip(),
                         T.ToTensor(), T.Normalize([.5]*3,[.5]*3)])
        folder = DL._image_folder_root(args.dataset, "source")
        label_ds = dsets.ImageFolder(str(folder), transform=tf)
    label_loader = DataLoader(label_ds, batch_size=config["training"]["batch_size"],
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    # Output
    os.makedirs(args.out, exist_ok=True)
    log_file = os.path.join(args.out, "train.log")
    log_fh = open(log_file, "w")
    def log(msg):
        print(msg, flush=True)
        log_fh.write(msg + "\n")
        log_fh.flush()

    log(f"CyCADA-lite: dataset={args.dataset} lambda_sem={args.lambda_sem} "
        f"lambda_gan={args.lambda_gan} epochs={args.epochs} lr={args.lr} "
        f"img_size={cyclegan_img_size}")

    # Fine-tuning loop
    global_step = 0
    for epoch in range(args.epochs):
        trainer.set_train()
        cycada.feat_extractor.eval()
        cycada.D_feat.train()
        t0 = time.time()
        label_iter = iter(label_loader)

        for real_A, real_B in unpaired_loader:
            # Get labeled source batch (cycle independently)
            try:
                src_labeled, src_labels = next(label_iter)
            except StopIteration:
                label_iter = iter(label_loader)
                src_labeled, src_labels = next(label_iter)

            # Use labeled source as real_A for semantic loss
            losses = cycada.train_step(src_labeled, real_B, src_labels)
            global_step += 1

            if global_step % 10 == 0:
                log(f"[step {global_step}] G={losses['G']:.4f} sem={losses['sem']:.4f} "
                    f"sem_acc={losses['sem_acc']:.3f} feat_G={losses.get('feat_G',0):.4f} "
                    f"D_feat={losses.get('D_feat',0):.4f} cyc={losses['cyc']:.4f} "
                    f"D_A={losses['D_A']:.4f} D_B={losses['D_B']:.4f}")

        elapsed = time.time() - t0
        log(f"[Epoch {epoch+1}/{args.epochs}] done ({elapsed:.1f}s)")

    # Save fine-tuned checkpoint
    ckpt_path = os.path.join(args.out, "latest.pth")
    torch.save({
        "epoch": args.epochs,
        "global_step": global_step,
        "G_AB": trainer.G_AB.state_dict(),
        "G_BA": trainer.G_BA.state_dict(),
        "D_A": trainer.D_A.state_dict(),
        "D_B": trainer.D_B.state_dict(),
    }, ckpt_path)
    log(f"Saved fine-tuned checkpoint to {ckpt_path}")
    log_fh.close()


if __name__ == "__main__":
    main()
