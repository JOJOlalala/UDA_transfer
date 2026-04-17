"""UNet-based Generator for CycleGAN.

Skip connections let fine spatial detail (edges, class-discriminative features)
bypass the bottleneck. The bottleneck only needs to encode the style change.
Uses concat-first-then-upsample pattern (standard pix2pix UNet).

For 64×64: 4 encoder stages → 2×2 bottleneck → 4 decoder stages.
"""
import torch
import torch.nn as nn


class UNetDown(nn.Module):
    """Encoder: Conv(4×4, s2) → [Norm] → LeakyReLU. Halves spatial."""

    def __init__(self, in_ch, out_ch, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """Decoder: ConvTranspose(4×4, s2) → Norm → ReLU [→ Dropout].
    Input is pre-concatenated (skip + previous decoder output)."""

    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetGenerator(nn.Module):
    """UNet generator with skip connections.

    Channel progression: ngf → ngf*2 → ngf*4 → ngf*8 (capped).

    For 64×64, n_downsample=4, ngf=64:
      Enc: 3→64(32) → 128(16) → 256(8) → 512(4)
      Bot: 512(2) → 512(4)
      Dec: [512+512]=1024→256(8) → [256+256]=512→128(16) → [128+128]=256→64(32)
      Out: [64+64]=128→3(64) + Tanh
    """

    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_downsample=4):
        super().__init__()
        self.n_downsample = n_downsample

        # Compute encoder channel sizes
        enc_chs = []  # output channels per encoder stage
        ch = ngf
        for _ in range(n_downsample):
            enc_chs.append(min(ch, ngf * 8))
            ch *= 2

        # Encoder
        self.encoders = nn.ModuleList()
        in_ch = input_nc
        for i, out_ch in enumerate(enc_chs):
            self.encoders.append(UNetDown(in_ch, out_ch, normalize=(i > 0)))
            in_ch = out_ch

        # Bottleneck: down to smallest spatial, then up (no norm on down)
        bot_ch = enc_chs[-1]
        self.bot_down = nn.Sequential(
            nn.Conv2d(bot_ch, bot_ch, 4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.bot_up = nn.Sequential(
            nn.ConvTranspose2d(bot_ch, bot_ch, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(bot_ch),
            nn.ReLU(inplace=True),
        )

        # Decoder: concat skip first, then ConvTranspose to upsample
        self.decoders = nn.ModuleList()
        prev_ch = bot_ch  # output from bottleneck_up
        for i in range(n_downsample - 1, 0, -1):
            skip_ch = enc_chs[i]
            cat_ch = prev_ch + skip_ch
            out_ch = enc_chs[i - 1]
            dropout = (i >= n_downsample - 1)
            self.decoders.append(UNetUp(cat_ch, out_ch, dropout=dropout))
            prev_ch = out_ch

        # Final output: concat with first encoder skip, upsample to input res
        self.final = nn.Sequential(
            nn.ConvTranspose2d(prev_ch + enc_chs[0], output_nc, 4,
                               stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        # Encode
        skips = []
        h = x
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)

        # Bottleneck
        h = self.bot_down(h)
        h = self.bot_up(h)

        # Decode with skip connections (reverse order)
        for i, dec in enumerate(self.decoders):
            skip = skips[-(i + 1)]  # last encoder skip first
            h = dec(torch.cat([h, skip], dim=1))

        # Final with first encoder skip
        h = torch.cat([h, skips[0]], dim=1)
        return self.final(h)
