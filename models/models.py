
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch_geometric.transforms as T


class GNNBlock(nn.Module):
    def __init__(self,
        input_size: int, output_size: int, layer_name: str,
        layer_params: Dict[str, Any] = None, activation_fn: callable = nn.ReLU(),
        layer_norm: bool = False, norm_first: bool = False
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layer_name = layer_name
        self.layer_params = layer_params or {}
        self.activation_fn = activation_fn
        self.layer_norm = layer_norm
        self.norm_first = norm_first
        self.has_edge_attr = False
        self.has_edge_weight = False
        self.graph_layer = None
        self.norm = None

        self._setup_model()

    def _setup_model(self):
        if self.layer_name == "ChebConv":
            self.has_edge_attr = False
            self.has_edge_weight = True
            self.graph_layer =  gnn.ChebConv(
                self.input_size, self.output_size, **self.layer_params)
        elif self.layer_name == "GCNConv":
            self.has_edge_attr = False
            self.has_edge_weight = True
            self.graph_layer =  gnn.GCNConv(
                self.input_size, self.output_size, **self.layer_params)
        elif self.layer_name == "SAGEConv":
            self.has_edge_attr = False
            self.has_edge_weight = False
            self.graph_layer =  gnn.SAGEConv(
                self.input_size, self.output_size, **self.layer_params)
        elif self.layer_name == "GATConv":
            self.has_edge_attr = True
            self.has_edge_weight = False
            self.layer_params['concat'] = False  # only works when False
            self.graph_layer =  gnn.GATConv(
                self.input_size, self.output_size, **self.layer_params)
        else:
            raise ValueError(f"Unknown graph layer: {layer_name}")

        if self.layer_norm:
            self.norm = gnn.norm.LayerNorm(self.output_size)

    def forward(self, x, edge_index, edge_attr=None, edge_weight=None):
        if self.has_edge_attr:
            x = self.graph_layer(x, edge_index, edge_attr)
        elif self.has_edge_weight:
            x = self.graph_layer(x, edge_index, edge_weight)
        else:
            x = self.graph_layer(x, edge_index)
        if self.norm_first and self.norm is not None:
            x = self.norm(x)
            x = self.activation_fn(x)
        elif self.norm is not None:
            x = self.activation_fn(x)
            x = self.norm(x)
        else:
            x = self.activation_fn(x)
        return x


class GNN(nn.Module):
    """ Graph Neural Network model

    Attributes
    ----------
    layers: nn.ModuleList
        List of graph layers
    activation_fn: callable
        Activation function
    """
    def __init__(
        self, input_size: int, hidden_sizes: List[int], projection_size: int = None,
        graph_layer: str = "ChebConv", graph_layer_params: Dict[str, Any] = None,
        activation_fn: callable = nn.ReLU(), pooling: str = "mean",
        layer_norm: bool = False, norm_first: bool = False
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.projection_size = projection_size
        self.graph_layer = graph_layer
        self.graph_layer_params = graph_layer_params or {}
        self.activation_fn = activation_fn
        self.pooling = pooling
        self.layer_norm = layer_norm
        self.norm_first = norm_first
        self.layers = None
        self.has_edge_attr = None
        self.has_edge_weight = None

        # setup the model
        if self.projection_size:
            self.projection_layer = nn.Linear(
                self.input_size, self.projection_size)
            input_size = self.projection_size
        else:
            self.projection_layer = None
            input_size = self.input_size

        self.layers = nn.ModuleList()
        layer_sizes = [input_size] + self.hidden_sizes
        for i in range(1, len(layer_sizes)):
            layer = GNNBlock(
                layer_sizes[i-1], layer_sizes[i], self.graph_layer,
                self.graph_layer_params, self.activation_fn,
                self.layer_norm, self.norm_first
            )
            self.layers.append(layer)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor,
        batch: torch.Tensor, edge_attr: torch.Tensor = None,
        edge_weight: torch.Tensor = None
    ) -> torch.Tensor:
        if self.projection_layer:
            x = self.projection_layer(x)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, edge_weight)

        # global pooling
        if self.pooling == "mean":
            return gnn.global_mean_pool(x, batch)
        elif self.pooling == "max":
            return gnn.global_max_pool(x, batch)
        elif self.pooling == "sum":
            return gnn.global_add_pool(x, batch)
        else:
            return x


class MLP(nn.Module):
    """
    MLP with a variable number of hidden layers.

    Attributes
    ----------
    layers : nn.ModuleList
        The layers of the MLP.
    activation_fn : callable
        The activation function to use.
    """
    def __init__(self, input_size, output_size, hidden_sizes=[512],
                 activation_fn=nn.ReLU()):
        """
        Parameters
        ----------
        input_size : int
            The size of the input
        output_size : int
            The number of classes
        hidden_sizes : list of int, optional
            The sizes of the hidden layers. Default: [512]
        activation_fn : callable, optional
            The activation function to use. Default: nn.ReLU()
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.activation_fn = activation_fn

        # Create a list of all layer sizes: input, hidden, and output
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        # Create layers dynamically
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        # Store the activation function
        self.activation_fn = activation_fn

    def forward(self, x):
        # Apply layers and activation function
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Apply activation function to all but last layer
                x = self.activation_fn(x)
        return x
