import torch
import torch.nn as nn


class Model(torch.nn.Module):
    """
    Creates new CNN for CIFAR10 input.
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """
        Initialize the model.
        """
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(64, num_classes)

        self.fc3 = nn.Linear(32 * 16 * 16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.pool1(self.relu2(self.norm2(self.conv2(x))))

        x = x.view(x.size(0), -1)
        x = self.fc3(x)

        return x
