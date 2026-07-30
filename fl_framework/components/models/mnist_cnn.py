import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTCNN(nn.Module):
    """
    A CNN model suitable for MNIST/FedMnist (28x28 grayscale images, 10 classes).

    Architecture:
        Conv1(1, 32, 5, padding=2) -> ReLU -> MaxPool(2)
        Conv2(32, 64, 5, padding=2) -> ReLU -> MaxPool(2)
        FC1(64*7*7, 512) -> ReLU
        FC2(512, 10)

    Total parameters: ~1.66M, gradient dimension d=1663370.
    This dimension is divisible by m=410 (n=4057), making it compatible
    with the compress4_clip aggregator.

    Args:
        in_channels (int): Number of input channels. Default 1 for grayscale.
        num_classes (int): Number of output classes. Default 10.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
