
from typing import List, Dict, Any, Tuple, Callable

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch_geometric.transforms as T


class GNNBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        layer_name: str,
        layer_params: Dict[str, Any] = None,
        act: Callable = nn.ReLU(),
        layer_norm: bool = False,
        norm_first: bool = False
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layer_name = layer_name
        self.layer_params = layer_params or {}
        self.act = act
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
        elif self.layer_name == "APPNP":
            self.has_edge_attr = False
            self.has_edge_weight = True
            self.graph_layer = gnn.APPNP(
                self.layer_params.get('K', 10),
                self.layer_params.get('alpha', 0.1)
            )
        elif self.layer_name == "SGConv":
            self.has_edge_attr = False
            self.has_edge_weight = True
            self.graph_layer = gnn.SGConv(
                self.input_size, self.output_size,
                K=self.layer_params.get('K', 2),
                cached=self.layer_params.get('cached', False)
            )

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

        if self.norm is None:
            x = self.act(x)
        else:
            x = self.norm(self.act(x)) if self.norm_first else self.act(self.norm(x))
        return x


class GNN(nn.Module):
    """Graph Neural Network model.

    Parameters
    ----------
    input_size : int
        Size of input features
    hidden_sizes : List[int]
        List of hidden layer sizes
    projection_size : int, optional
        Size of initial projection layer
    graph_layer : str
        Type of graph convolution layer
    graph_layer_params : Dict[str, Any], optional
        Parameters for the graph layer
    act : callable
        Activation function class (not instance), e.g., nn.ReLU
    pooling : str
        Type of global pooling ('mean', 'max', 'sum', or None)
    layer_norm : bool
        Whether to use layer normalization
    norm_first : bool
        Whether to apply normalization before activation
    """
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        projection_size: int = None,
        graph_layer: str = "ChebConv",
        graph_layer_params: Dict[str, Any] = None,
        act: Callable = nn.ReLU(),
        pooling: str = "mean",
        layer_norm: bool = False,
        norm_first: bool = False
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.projection_size = projection_size
        self.graph_layer = graph_layer
        self.graph_layer_params = graph_layer_params or {}
        self.act = act
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
                self.graph_layer_params, self.act,
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

