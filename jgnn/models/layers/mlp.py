
from typing import Optional, List, Callable, Any
import torch.nn as nn


class MLP(nn.Module):
    """MLP with optional batch normalization and dropout.

    Parameters
    ----------
    input_size : int
        Size of the input features
    output_size : int
        Size of the output features
    hidden_sizes : list of int
        Sizes of hidden layers
    act : callable
        Activation function class (not instance), e.g., nn.ReLU
    batch_norm : bool
        Whether to use batch normalization
    dropout : float
        Dropout probability
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int] = [512],
        act: Callable = nn.ReLU(),
        batch_norm: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.act = act
        self.batch_norm = batch_norm
        self.dropout = dropout

        # Create a list of all layer sizes: input, hidden, and output
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        # Create layers dynamically
        layers = []
        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(act)
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)