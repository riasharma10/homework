from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import (
    Compose,
    ToTensor,
    RandomCrop,
    RandomHorizontalFlip,
    Normalize,
)


class CONFIG:
    batch_size = 8
    num_epochs = 8

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(model.parameters(), lr=1e-3)

    transforms = Compose(
        [
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
