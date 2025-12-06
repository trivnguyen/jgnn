"""Transformer-based embedding models with PyTorch Lightning."""

from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl

from .layers import Transformer
from .utils import configure_optimizers


class TransformerEmbedding(pl.LightningModule):
    """Transformer-based embedding model for graph data.

    This model converts PyTorch Geometric graph batches to padded sequences,
    processes them through a transformer, and produces embeddings.

    The model expects PyTorch Geometric Data batches and automatically converts
    them to padded sequences for the transformer.

    Parameters
    ----------
    input_size : int
        Size of input node/element features (also the output embedding size)
    d_model : int
        Dimension of the model embedding space
    d_mlp : int
        Dimension of the feed-forward MLP in transformer blocks
    n_layers : int
        Number of transformer layers
    n_heads : int
        Number of attention heads
    d_pos : int, optional
        Dimension of positional encoding features
    d_cond : int, optional
        Dimension of conditioning features
    concat_conditioning : bool
        Whether to concatenate conditioning to the input
    use_pos_enc : bool
        Whether to use positional encoding
    pooling : str
        Type of pooling to aggregate sequence ('mean', 'max', 'sum', or 'cls')
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
        d_model: int = 128,
        d_mlp: int = 512,
        n_layers: int = 4,
        n_heads: int = 4,
        d_pos: Optional[int] = None,
        d_cond: Optional[int] = None,
        concat_conditioning: bool = False,
        use_pos_enc: bool = False,
        pooling: str = 'mean',
        loss_type: str = 'mse',
        loss_args: Optional[Dict[str, Any]] = None,
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        pre_transforms=None,
        norm_dict=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_pos = d_pos
        self.d_cond = d_cond
        self.concat_conditioning = concat_conditioning
        self.use_pos_enc = use_pos_enc
        self.pooling = pooling
        self.loss_type = loss_type
        self.loss_args = loss_args or {}
        self.optimizer_args = optimizer_args or {}
        self.scheduler_args = scheduler_args or {}
        self.pre_transforms = pre_transforms
        self.norm_dict = norm_dict
        self.save_hyperparameters(ignore=['pre_transforms'])

        self._setup_model()

    def _setup_model(self):
        """Initialize Transformer and loss function."""
        from .utils import build_embedding_loss

        # Create Transformer
        self.transformer = Transformer(
            d_in=self.input_size,
            d_model=self.d_model,
            d_mlp=self.d_mlp,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_pos=self.d_pos,
            d_cond=self.d_cond,
            concat_conditioning=self.concat_conditioning,
            use_pos_enc=self.use_pos_enc
        )

        # Initialize loss function
        loss_config = dict(self.loss_args)
        if self.loss_type == 'flow' and 'context_features' not in loss_config:
            loss_config['context_features'] = self.input_size

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
            mask[graph_idx, :n_nodes] = False  # False = valid data for torch
            node_idx += n_nodes

        return x_padded, mask, num_nodes_per_graph

    def _pool_sequence(self, x, mask):
        """Pool sequence representations to single vector per graph.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, seq_len, features)
        mask : torch.Tensor
            Shape (batch_size, seq_len), True for padding positions

        Returns
        -------
        torch.Tensor
            Shape (batch_size, features)
        """
        if self.pooling == 'mean':
            # Masked mean pooling
            x_masked = x.masked_fill(mask.unsqueeze(-1), 0)
            lengths = (~mask).sum(dim=1, keepdim=True).clamp(min=1)
            return x_masked.sum(dim=1) / lengths.float()
        elif self.pooling == 'max':
            # Masked max pooling
            x_masked = x.masked_fill(mask.unsqueeze(-1), float('-inf'))
            return x_masked.max(dim=1)[0]
        elif self.pooling == 'sum':
            # Masked sum pooling
            x_masked = x.masked_fill(mask.unsqueeze(-1), 0)
            return x_masked.sum(dim=1)
        elif self.pooling == 'cls':
            # Use first token (assumes CLS token)
            return x[:, 0]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

    def forward(self, batch_dict):
        """Forward pass through Transformer -> Pooling.

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
            Embedding of shape (batch_size, input_size)
        """
        # Transformer forward pass
        x = self.transformer(
            batch_dict['x'],
            conditioning=batch_dict.get('cond', None),
            mask=batch_dict['mask'],
            pos_enc=batch_dict.get('pos_enc', None)
        )

        # Pool sequence to single vector per graph
        embedding = self._pool_sequence(x, batch_dict['mask'])

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
