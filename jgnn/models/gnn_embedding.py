"""GNN-based embedding models with PyTorch Lightning."""

from typing import Dict, Any

import torch.nn as nn
import pytorch_lightning as pl
from ml_collections import ConfigDict

from .layers import GNN, MLP
from .utils import get_activation, configure_optimizers


class GNNEmbedding(pl.LightningModule):
    """GNN-based embedding model with optional conditional MLP.

    This model consists of:
    1. GNN featurizer that processes graph inputs
    2. MLP that projects GNN outputs to embedding space
    3. Optional conditional MLP for additional conditioning inputs

    Parameters
    ----------
    input_size : int
        Size of input node features
    gnn_args : ConfigDict
        Configuration for GNN (hidden_sizes, projection_size, graph_layer, etc.)
    mlp_args : ConfigDict
        Configuration for MLP (hidden_sizes, output_size, etc.)
    conditional_mlp_args : ConfigDict, optional
        Configuration for conditional MLP if additional conditioning is needed
    optimizer_args : ConfigDict, optional
        Optimizer configuration
    scheduler_args : ConfigDict, optional
        Scheduler configuration
    """

    def __init__(
        self,
        input_size: int,
        gnn_args: ConfigDict,
        mlp_args: ConfigDict,
        conditional_mlp_args: ConfigDict = None,
        optimizer_args: ConfigDict = None,
        scheduler_args: ConfigDict = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.gnn_args = gnn_args
        self.mlp_args = mlp_args
        self.conditional_mlp_args = conditional_mlp_args
        self.optimizer_args = optimizer_args or {}
        self.scheduler_args = scheduler_args or {}
        self.save_hyperparameters()

        self._setup_model()

    def _setup_model(self):
        """Initialize GNN, MLP, and optional conditional MLP."""

        # Create GNN featurizer
        gnn_activation_fn = get_activation(self.gnn_args.activation)
        self.gnn = GNN(
            input_size=self.input_size,
            hidden_sizes=self.gnn_args.hidden_sizes,
            projection_size=self.gnn_args.get('projection_size', None),
            graph_layer=self.gnn_args.graph_layer,
            graph_layer_params=self.gnn_args.get('graph_layer_params', {}),
            activation_fn=gnn_activation_fn,
            pooling=self.gnn_args.get('pooling', 'mean'),
            layer_norm=self.gnn_args.get('layer_norm', False),
            norm_first=self.gnn_args.get('norm_first', False),
        )

        # Create MLP
        mlp_activation_fn = get_activation(self.mlp_args.activation)
        self.mlp = MLP(
            input_size=self.gnn_args.hidden_sizes[-1],
            hidden_sizes=self.mlp_args.hidden_sizes,
            output_size=self.mlp_args.output_size,
            activation_fn=mlp_activation_fn,
            batch_norm=self.mlp_args.get('batch_norm', False),
            dropout=self.mlp_args.get('dropout', 0.0),
        )

        # Create conditional MLP if specified
        if self.conditional_mlp_args is not None:
            cond_activation_fn = get_activation(self.conditional_mlp_args.activation)
            self.conditional_mlp = MLP(
                input_size=self.conditional_mlp_args.input_size,
                hidden_sizes=self.conditional_mlp_args.hidden_sizes,
                output_size=self.conditional_mlp_args.output_size,
                activation_fn=cond_activation_fn,
                batch_norm=self.conditional_mlp_args.get('batch_norm', False),
                dropout=self.conditional_mlp_args.get('dropout', 0.0),
            )
        else:
            self.conditional_mlp = None

    def forward(self, x, edge_index, batch, edge_attr=None, edge_weight=None, cond=None):
        """Forward pass through GNN -> MLP [+ CondMLP].

        Parameters
        ----------
        x : torch.Tensor
            Node features [num_nodes, input_size]
        edge_index : torch.Tensor
            Edge indices [2, num_edges]
        batch : torch.Tensor
            Batch assignment for nodes [num_nodes]
        edge_attr : torch.Tensor, optional
            Edge attributes [num_edges, edge_attr_dim]
        edge_weight : torch.Tensor, optional
            Edge weights [num_edges]
        cond : torch.Tensor, optional
            Conditional inputs [batch_size, cond_dim]

        Returns
        -------
        torch.Tensor
            Embedding [batch_size, output_size]
        """
        # GNN featurizer
        embedding = self.gnn(
            x, edge_index, batch=batch,
            edge_attr=edge_attr, edge_weight=edge_weight
        )

        # MLP projection
        embedding = self.mlp(embedding)

        # Add conditional features if provided
        if self.conditional_mlp is not None and cond is not None:
            cond_embedding = self.conditional_mlp(cond)
            embedding = embedding + cond_embedding

        return embedding

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        return configure_optimizers(
            self.parameters(), self.optimizer_args, self.scheduler_args
        )
