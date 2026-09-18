"""Model architectures for MRI tumor classification.

Two options:
  - SimpleCNN: a small custom CNN trained from scratch.
  - ResNetTransfer: a torchvision ResNet18 backbone, fine-tuned.
"""

import torch.nn as nn
from torchvision import models

import config


class SimpleCNN(nn.Module):
    """A compact 4-block CNN. Good for CPU training / learning the basics."""

    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(3, 32),     # 224 -> 112
            conv_block(32, 64),    # 112 -> 56
            conv_block(64, 128),   # 56 -> 28
            conv_block(128, 256),  # 28 -> 14
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ResNetTransfer(nn.Module):
    """ResNet18 pretrained on ImageNet, with a new classification head.

    By default, the convolutional backbone is left trainable (full
    fine-tuning) since MRI images differ a lot from ImageNet photos, but
    you can freeze it via `freeze_backbone=True` for a lighter/faster run.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES,
                 freeze_backbone: bool = False):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def build_model(name: str = "resnet18", num_classes: int = config.NUM_CLASSES):
    """Factory function used by train.py / evaluate.py / predict.py."""
    name = name.lower()
    if name == "resnet18":
        return ResNetTransfer(num_classes=num_classes)
    if name == "simple":
        return SimpleCNN(num_classes=num_classes)
    raise ValueError(f"Unknown model name: {name}. Choose 'resnet18' or 'simple'.")
