"""Classifier architectures for Task II UDA evaluation.

Two families:
- SmallCNN: 3-block conv net for 32x32 digits, ~200K params, trained from scratch.
- ResNet-18: ImageNet-pretrained, FC replaced, fine-tuned for 64x64 office datasets.

Both take 3-channel input in the [-1, 1] range (matching CycleGAN normalization).
"""
import torch
import torch.nn as nn
from torchvision import models


def _gn(ch: int, groups: int = 8) -> nn.GroupNorm:
    # GroupNorm with a fixed group count — removes the stored-BN-stats failure
    # mode that sinks source-only UDA baselines on MNIST→USPS and similar pairs.
    return nn.GroupNorm(num_groups=min(groups, ch), num_channels=ch)


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int, in_ch: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=1, padding=1),
            _gn(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            _gn(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            _gn(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            _gn(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            _gn(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            _gn(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return self.fc(h)


def get_resnet18(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet18(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    # Use batch statistics at both train and eval time so target-domain
    # inputs are not measured against source-only running means/vars.
    # Matches the BN_STATS diagnostic finding on SmallCNN.
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return net


def get_resnet50(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet50(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return net


def get_resnet18_cifar(num_classes: int, pretrained: bool = True) -> nn.Module:
    """CIFAR-style ResNet-18: small stem (3×3/s1, no maxpool) for 64×64 input.

    At 64×64: layer4 feature map is 8×8 (vs 2×2 with standard stem).
    Loads ImageNet pretrained weights for layers 1-4; stem is reinitialized.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet18(weights=weights)
    # Replace aggressive stem: 7×7/s2 + maxpool/s2 → 3×3/s1, no maxpool
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    nn.init.kaiming_normal_(net.conv1.weight, mode="fan_out", nonlinearity="relu")
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return net


def build_classifier(arch: str, num_classes: int) -> nn.Module:
    if arch == "smallcnn":
        return SmallCNN(num_classes=num_classes)
    if arch == "resnet18":
        return get_resnet18(num_classes=num_classes, pretrained=True)
    if arch == "resnet18_cifar":
        return get_resnet18_cifar(num_classes=num_classes, pretrained=True)
    if arch == "resnet50":
        return get_resnet50(num_classes=num_classes, pretrained=True)
    raise ValueError(f"unknown arch: {arch}")
