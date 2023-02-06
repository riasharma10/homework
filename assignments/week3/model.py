import torch
from typing import Callable


class MLP(torch.nn.Module):
    """
    Class that defines the MLP for the MNIST dataset.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()
        self.actv = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_size, hidden_size))

        for i in range(hidden_count):
            self.layers += [torch.nn.Linear(hidden_size, hidden_size)]

        self.out = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """

        x = x.flatten(start_dim=1)
        # Get activations of each layer
        for layer in self.layers:
            x = self.actv(layer(x))

        # Get outputs
        x = self.out(x)

        return x
