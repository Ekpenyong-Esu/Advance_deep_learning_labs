"""models/mnist_cnn.py — CNN for MNIST and Transfer to SVHN"""
import torch.nn as nn


class MnistCNN(nn.Module):
    def __init__(self, num_classes: int = 10, input_size: int = 28):
        super(MnistCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )
        flat_size = 64 * (input_size // 4) * (input_size // 4)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(flat_size, 512), nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def freeze_features(self):
        for param in self.features.parameters():
            param.requires_grad = False
        print("Feature extractor FROZEN — only the classifier will be updated.")

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
        print("All layers UNFROZEN.")
