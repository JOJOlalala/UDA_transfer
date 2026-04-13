"""ResNet-based Generator for CycleGAN."""
import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    """Residual block with reflection padding and instance norm."""

    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """ResNet generator with configurable depth and width.

    For 256x256: ngf=64, n_blocks=9, n_downsample=2
    For  32x32: ngf=32, n_blocks=6, n_downsample=1
    """

    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9, n_downsample=2):
        super().__init__()
        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        ch = ngf
        for i in range(n_downsample):
            model += [
                nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(ch * 2),
                nn.ReLU(inplace=True),
            ]
            ch *= 2

        # ResNet blocks
        for _ in range(n_blocks):
            model.append(ResNetBlock(ch))

        # Upsampling
        for i in range(n_downsample):
            model += [
                nn.ConvTranspose2d(ch, ch // 2, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(ch // 2),
                nn.ReLU(inplace=True),
            ]
            ch //= 2

        # Output convolution
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
