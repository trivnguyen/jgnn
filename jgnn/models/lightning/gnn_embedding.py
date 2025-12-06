"""GNN-based embedding models with PyTorch Lightning."""

from typing import Dict, Any

import torch.nn as nn
import pytorch_lightning as pl

from ..layers import GNN, MLP
from ..utils import get_activation, configure_optimizers
from ..utils import build_embedding_loss


class GNNEmbedding(pl.LightningModule):
    """GNN-based embedding model with optional conditional MLP.

    This model consists of:
    1. GNN featurizer that processes graph inputs
    2. MLP that projects GNN outputs to embedding space
    3. Optional conditional MLP for additional conditioning inputs
    4. Loss function (MSE or Flow-based variational loss)

    Parameters
    ----------
    input_size : int
        Size of input node features
    gnn_args : ConfigDict
        Configuration for GNN (hidden_sizes, projection_size, graph_layer, etc.)
    mlp_args : ConfigDict
        Configuration for MLP (hidden_sizes, output_size, etc.)
    loss_type : str
        Type of loss function ('mse' or 'flow')
    loss_args : ConfigDict, optional
        Configuration for loss function
        For 'flow' loss: features, context_features, num_transforms, hidden_features, num_bins, activation
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
        gnn_args: Dict[str, Any],
        mlp_args: Dict[str, Any],
        loss_type: str = 'mse',
        loss_args: Dict[str, Any] = None,
        conditional_mlp_args: Dict[str, Any] = None,
        optimizer_args: Dict[str, Any] = None,
        scheduler_args: Dict[str, Any] = None,
        pre_transforms=None,
        norm_dict=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = None  # to be defined by mlp_args
        self.gnn_args = gnn_args
        self.mlp_args = mlp_args
        self.loss_type = loss_type
        self.loss_args = loss_args or {}
        self.conditional_mlp_args = conditional_mlp_args
        self.optimizer_args = optimizer_args or {}
        self.scheduler_args = scheduler_args or {}
        self.pre_transforms = pre_transforms
        self.norm_dict = norm_dict
        self.save_hyperparameters(ignore=['pre_transforms'])

        self._setup_model()

    def _setup_model(self):
        """Initialize GNN, MLP, optional conditional MLP, and loss function."""

        # Create GNN featurizer
        gnn_config = dict(self.gnn_args)
        gnn_config['input_size'] = self.input_size
        gnn_config['act'] = get_activation(
            gnn_config.pop('act_name'), gnn_config.pop('act_args', {}))
        self.gnn = GNN(**gnn_config)

        # Create MLP
        mlp_config = dict(self.mlp_args)
        mlp_config['input_size'] = self.gnn_args.hidden_sizes[-1]
        mlp_config['act'] = get_activation(
            mlp_config.pop('act_name'), mlp_config.pop('act_args', {}))
        self.mlp = MLP(**mlp_config)

        # Create conditional MLP if specified
        if self.conditional_mlp_args is not None:
            cond_config = dict(self.conditional_mlp_args)
            cond_config['act'] = get_activation(
                cond_config.pop('act_name'), cond_config.pop('act_args', {}))
            self.conditional_mlp = MLP(**cond_config)
        else:
            self.conditional_mlp = None

        self.output_size = self.mlp_args['output_size']

        # Initialize loss function
        # For flow loss, auto-set context_features to match MLP output if not specified
        loss_config = dict(self.loss_args)
        if self.loss_type == 'flow' and 'context_features' not in loss_config:
            loss_config['context_features'] = self.mlp_args['output_size']

        self.loss_fn, self.flow = build_embedding_loss(self.loss_type, loss_config)

    def forward(self, batch_dict):
        """Forward pass through GNN -> MLP [+ CondMLP]."""
        # GNN featurizer
        embedding = self.gnn(
            batch_dict['x'],
            batch_dict['edge_index'],
            batch=batch_dict['batch'],
            edge_attr=batch_dict['edge_attr'],
            edge_weight=batch_dict['edge_weight'],
        )

        # MLP projection
        embedding = self.mlp(embedding)

        # Add conditional features if provided
        if self.conditional_mlp is not None:
            cond_embedding = self.conditional_mlp(batch_dict['cond'])
            embedding = embedding + cond_embedding

        return embedding

    def _prepare_batch(self, batch):
        """Prepare batch data for training/validation.

        Override this method to extract data from your specific batch format.

        Parameters
        ----------
        batch : Any
            Raw batch from dataloader

        Returns
        -------
        dict
            Dictionary containing:
            - 'x': Node features
            - 'edge_index': Edge connectivity
            - 'batch': Batch assignment
            - 'target': Target values for loss computation
            - 'edge_attr': Optional edge attributes
            - 'edge_weight': Optional edge weights
            - 'cond': Optional conditioning variables
            - 'batch_size': Batch size
        """
        batch = self.pre_transforms(batch) if self.pre_transforms else batch
        batch = batch.to(self.device)

        # Default implementation for PyG Data objects
        batch_dict = {
            'x': batch.x,
            'edge_index': batch.edge_index,
            'batch': batch.batch,
            'target': batch.theta,
            'edge_attr': batch.edge_attr if hasattr(batch, 'edge_attr') else None,
            'edge_weight': batch.edge_weight if hasattr(batch, 'edge_weight') else None,
            'cond': batch.cond if hasattr(batch, 'cond') else None,
            'batch_size': batch.num_graphs if hasattr(batch, 'num_graphs') else batch.batch.max().item() + 1,
        }
        return batch_dict

    def training_step(self, batch, batch_idx):
        """Training step for PyTorch Lightning.

        Parameters
        ----------
        batch : Any
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
        # Use hierarchical naming for better organization in loggers like WandB/TensorBoard
        self.log(
            'train/loss', loss, on_step=True, on_epoch=True, prog_bar=True,
            logger=True, batch_size=batch_dict['batch_size'], sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for PyTorch Lightning.

        Parameters
        ----------
        batch : Any
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
        # Validation metrics are typically only logged at epoch level
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
