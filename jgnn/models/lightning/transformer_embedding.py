"""Transformer-based embedding models with PyTorch Lightning."""

from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl

from ..layers import Transformer, MLP
from ..utils import get_activation, configure_optimizers
from ..utils import build_embedding_loss


class TransformerEmbedding(pl.LightningModule):
    """Transformer-based embedding model for graph data.

    This model converts PyTorch Geometric graph batches to padded sequences,
    processes them through a transformer, and produces embeddings.

    This model consists of:
    1. Transformer that processes sequential/set inputs (with pooling)
    2. MLP that projects transformer outputs to embedding space

    The model expects PyTorch Geometric Data batches and automatically converts
    them to padded sequences for the transformer.

    Parameters
    ----------
    input_size : int
        Size of input node/element features
    transformer_args : Dict[str, Any]
        Configuration for Transformer (d_model, n_layers, n_heads, pooling, etc.)
    mlp_args : Dict[str, Any]
        Configuration for MLP (hidden_sizes, output_size, etc.)
    loss_type : str
        Type of loss function ('mse' or 'flow')
    loss_args : Dict[str, Any], optional
        Configuration for loss function
        For 'flow' loss: features, context_features, num_transforms, hidden_features, num_bins, activation
    optimizer_args : Dict[str, Any], optional
        Optimizer configuration
    scheduler_args : Dict[str, Any], optional
        Scheduler configuration
    pre_transforms : callable, optional
        Data transformations to apply before processing
    norm_dict : dict, optional
        Normalization parameters
    """

    def __init__(
        self,
        input_size: int,
        transformer_args: Dict[str, Any],
        loss_type: str = 'mse',
        loss_args: Optional[Dict[str, Any]] = None,
        mlp_args: Dict[str, Any] = None,
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        pre_transforms=None,
        norm_dict=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = None
        self.transformer_args = transformer_args
        self.mlp_args = mlp_args
        self.loss_type = loss_type
        self.loss_args = loss_args or {}
        self.optimizer_args = optimizer_args or {}
        self.scheduler_args = scheduler_args or {}
        self.pre_transforms = pre_transforms
        self.norm_dict = norm_dict
        self.save_hyperparameters(ignore=['pre_transforms'])

        self._setup_model()

    def _setup_model(self):
        """Initialize Transformer, MLP, and loss function."""
        # Create Transformer
        self.transformer = Transformer(**self.transformer_args)

        # Create MLP
        if self.mlp_args is not None:
            mlp_config = dict(self.mlp_args)
            mlp_config['input_size'] = self.input_size
            mlp_config['act'] = get_activation(
                mlp_config.pop('act_name'), mlp_config.pop('act_args', {})
            )
            self.mlp = MLP(**mlp_config)
            self.output_size = self.mlp_args['output_size']
        else:
            self.mlp = None
            self.output_size = self.transformer_args.get('d_model', self.input_size)

        # Initialize loss function
        loss_config = dict(self.loss_args)
        if self.loss_type == 'flow' and 'context_features' not in loss_config:
            loss_config['context_features'] = self.mlp_args['output_size']

        self.loss_fn, self.flow = build_embedding_loss(self.loss_type, loss_config)

    def _convert_pyg_to_sequences(self, batch):
        """Convert PyTorch Geometric batch to padded sequences.

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Batched PyG Data object

        Returns
        -------
        tuple
            (x_padded, mask, num_nodes_per_graph)
            - x_padded: (batch_size, max_nodes, features) padded sequences
            - mask: (batch_size, max_nodes) boolean mask (True = padding)
            - num_nodes_per_graph: (batch_size,) number of nodes per graph
        """
        # Get batch size and unique batch indices
        batch_indices = batch.batch
        num_graphs = batch.num_graphs if hasattr(batch, 'num_graphs') else batch_indices.max().item() + 1

        # Count nodes per graph
        num_nodes_per_graph = torch.bincount(batch_indices, minlength=num_graphs)

        max_nodes = num_nodes_per_graph.max().item()

        # Create padded tensor
        batch_size = num_graphs
        feature_dim = batch.x.size(1)
        x_padded = torch.zeros(
            batch_size, max_nodes, feature_dim,
            dtype=batch.x.dtype, device=batch.x.device
        )

        # Create mask (True for padding positions)
        mask = torch.ones(batch_size, max_nodes, dtype=torch.bool, device=batch.x.device)

        # Fill in the actual data
        node_idx = 0
        for graph_idx in range(num_graphs):
            n_nodes = num_nodes_per_graph[graph_idx].item()
            x_padded[graph_idx, :n_nodes] = batch.x[node_idx:node_idx + n_nodes]
            mask[graph_idx, :n_nodes] = False
            node_idx += n_nodes

        return x_padded, mask, num_nodes_per_graph

    def forward(self, batch_dict):
        """Forward pass through Transformer -> MLP.

        Parameters
        ----------
        batch_dict : dict
            Dictionary containing:
            - 'x': Padded sequences (batch_size, seq_len, features)
            - 'mask': Padding mask (batch_size, seq_len)
            - 'pos_enc': Optional positional encoding
            - 'cond': Optional conditioning variables

        Returns
        -------
        torch.Tensor
            Embedding of shape (batch_size, output_size)
        """
        # Transformer forward pass (includes pooling if specified in transformer_args)
        embedding = self.transformer(
            batch_dict['x'],
            conditioning=batch_dict.get('cond', None),
            mask=batch_dict['mask'],
            pos_enc=batch_dict.get('pos_enc', None)
        )

        # MLP projection
        if self.mlp is not None:
            embedding = self.mlp(embedding)

        return embedding

    def _prepare_batch(self, batch):
        """Prepare batch data for training/validation.

        Converts PyTorch Geometric Data batch to padded sequences for transformer.

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Raw batch from dataloader (PyG Data format)

        Returns
        -------
        dict
            Dictionary containing:
            - 'x': Padded sequences (batch_size, max_nodes, features)
            - 'mask': Padding mask (batch_size, max_nodes), True for padding
            - 'target': Target values for loss computation
            - 'pos_enc': Optional positional encoding
            - 'cond': Optional conditioning variables
            - 'batch_size': Batch size
        """
        batch = self.pre_transforms(batch) if self.pre_transforms else batch
        batch = batch.to(self.device)

        # Convert PyG batch to padded sequences
        x_padded, mask, num_nodes = self._convert_pyg_to_sequences(batch)

        # Prepare batch dictionary
        batch_dict = {
            'x': x_padded,
            'mask': mask,
            'target': batch.theta if hasattr(batch, 'theta') else None,
            'pos_enc': None,
            'cond': batch.cond if hasattr(batch, 'cond') else None,
            'batch_size': x_padded.size(0),
        }

        return batch_dict

    def training_step(self, batch, batch_idx):
        """Training step for PyTorch Lightning.

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Training batch
        batch_idx : int
            Batch index

        Returns
        -------
        torch.Tensor
            Training loss
        """
        batch_dict = self._prepare_batch(batch)
        embedding = self.forward(batch_dict)

        # Compute loss
        loss = self.loss_fn(embedding, batch_dict['target'])

        # Log metrics
        self.log(
            'train/loss', loss, on_step=True, on_epoch=True, prog_bar=True,
            logger=True, batch_size=batch_dict['batch_size'], sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for PyTorch Lightning.

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Validation batch
        batch_idx : int
            Batch index

        Returns
        -------
        torch.Tensor
            Validation loss
        """
        batch_dict = self._prepare_batch(batch)
        embedding = self.forward(batch_dict)

        # Compute loss
        loss = self.loss_fn(embedding, batch_dict['target'])

        # Log metrics
        self.log(
            'val/loss', loss, on_step=False, on_epoch=True, prog_bar=True,
            logger=True, batch_size=batch_dict['batch_size'], sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        return configure_optimizers(
            self.parameters(), self.optimizer_args, self.scheduler_args
        )
