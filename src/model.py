"""
SignFrame AI - Models
Two models, pick one to start:

1. LandmarkMLP (RECOMMENDED to start)
   Input: 63 numbers = 21 MediaPipe hand landmarks x (x, y, z)
   Fast, light, robust to background / lighting.
   Trains in minutes on CPU.

2. SimpleCNN / MobileNetV2 (image model)
   Input: 128x128 hand crop image
   More accurate on clean data, more sensitive to domain shift.
   Use this after the landmark model works.
"""
import torch
import torch.nn as nn
from torchvision import models


# 24 static ASL letters - J and Z need motion, so they are excluded
LETTERS_24 = list("ABCDEFGHIKLMNOPQRSTUVWXY")
# A B C D E F G H I K L M N O P Q R S T U V W X Y  (no J, no Z)
LETTER_TO_IDX = {ch: i for i, ch in enumerate(LETTERS_24)}
IDX_TO_LETTER = {i: ch for ch, i in LETTER_TO_IDX.items()}


class LandmarkMLP(nn.Module):
    """63 -> 128 -> 64 -> 24. Tiny, trains fast."""
    def __init__(self, num_classes=24, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    """Small custom CNN for 128x128 grayscale / RGB hand crops."""
    def __init__(self, num_classes=24, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_mobilenet(num_classes=24, pretrained=True):
    """Transfer-learning model - best accuracy if you have a GPU.
    This is the closest to your VGG-16 rice work: same idea, lighter backbone.
    """
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    m = models.mobilenet_v2(weights=weights)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m
