"""PatchGAN Discriminator for CycleGAN."""
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator (no sigmoid — used with LSGAN).

    For 256x256: ndf=64, n_layers=3 (70x70 receptive field)
    For  32x32: ndf=32, n_layers=2
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super().__init__()
        layers = [
            nn.Conv2d(input_nc, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        ch = ndf
        for i in range(1, n_layers):
            ch_next = min(ch * 2, 512)
            layers += [
                nn.Conv2d(ch, ch_next, 4, stride=2, padding=1),
                nn.InstanceNorm2d(ch_next),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = ch_next

        # Second-to-last layer (stride 1)
        ch_next = min(ch * 2, 512)
        layers += [
            nn.Conv2d(ch, ch_next, 4, stride=1, padding=1),
            nn.InstanceNorm2d(ch_next),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Output layer (1-channel prediction map)
        layers.append(nn.Conv2d(ch_next, 1, 4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
