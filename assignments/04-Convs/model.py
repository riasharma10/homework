import torch
import torch.nn as nn


class Model(torch.nn.Module):
    """
    Creates new CNN for CIFAR10 input.
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """
        Init.
        """
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc3 = nn.Linear(16 * 16 * 16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.pool1(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.fc3(x)

        return x
