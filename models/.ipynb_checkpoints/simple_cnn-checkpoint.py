"""models/simple_cnn.py — Simple CNN for CIFAR-10"""
import torch.nn as nn


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.01)
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unknown activation '{name}'. Choose 'leakyrelu' or 'tanh'.")


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10, activation: str = "leakyrelu"):
        super(SimpleCNN, self).__init__()
        act = activation
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32),
            _make_activation(act), nn.MaxPool2d(2, 2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
            _make_activation(act), nn.MaxPool2d(2, 2),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
            _make_activation(act), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 4 * 4, 512), _make_activation(act),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256), _make_activation(act),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
