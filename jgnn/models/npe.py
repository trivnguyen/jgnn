"""Neural Posterior Estimation (NPE) module for graph-based inference."""

from typing import Dict, Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
from ml_collections import ConfigDict
from tqdm import tqdm

from jgnn import models, transforms


class NPE(pl.LightningModule):
    """Neural Posterior Estimation model for graph-structured data.

    This model combines an embedding network and normalizing flows
    to perform posterior estimation on graph-structured data.
    The embedding network is initialized externally and passed to the model,
    allowing for flexible architecture choices (e.g., GNN, Transformer, etc.).
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        flows_args: ConfigDict,
        embedding_nn: nn.Module=None,
        pre_transform_args: ConfigDict=None,
        optimizer_args: ConfigDict=None,
        scheduler_args: ConfigDict=None,
        norm_dict: Dict[str, Any]=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.flows_args = flows_args
        self.embedding_nn = embedding_nn
        self.pre_transform_args = pre_transform_args
        self.optimizer_args = optimizer_args or {}
        self.scheduler_args = scheduler_args or {}
        self.norm_dict = norm_dict

        self.save_hyperparameters(ignore=['embedding_nn'])

        self._setup_model()

    def _freeze_module(self, module: nn.Module):
        """Freeze all parameters in a module.

        Args:
            module: The module to freeze
        """
        for param in module.parameters():
            param.requires_grad = False

    def _setup_model(self):
        """Set up all model components including flows and pre-transforms."""

        # if embedding_nn is not provided, assume identity mapping
        if self.embedding_nn is None:
            self.embedding_nn = nn.Identity()
            embedding_output_size = self.input_size
        else:
            # Get embedding output size from the embedding network
            # Assumes embedding network has an mlp with output_size or an output_size attribute
            if hasattr(self.embedding_nn, 'mlp_args'):
                embedding_output_size = self.embedding_nn.mlp_args['output_size']
            elif hasattr(self.embedding_nn, 'output_size'):
                embedding_output_size = self.embedding_nn.output_size
            else:
                raise ValueError(
                    "Embedding network must have either 'mlp_args' with 'output_size' "
                    "or an 'output_size' attribute"
                )

        # Create the flows
        activation_fn = models.get_activation(self.flows_args.activation)
        self.flows = models.build_flows(
            context_features=embedding_output_size,
            features=self.output_size,
            hidden_features=self.flows_args.hidden_sizes,
            num_transforms=self.flows_args.num_transforms,
            num_bins=self.flows_args.num_bins,
            activation=activation_fn,
            randperm=True
        )

        # Create pre-transforms if specified
        if self.pre_transform_args is not None:
            self.pre_transform = transforms.build_transformation(
                graph_name=self.pre_transform_args.graph_name,
                graph_params=self.pre_transform_args.graph_params,
                random_projection=self.pre_transform_args.get('random_projection', False),
                selection=self.pre_transform_args.get('selection', False),
                selection_args=self.pre_transform_args.get('selection_params', {}),
                uncertainty=self.pre_transform_args.get('uncertainty', False),
                uncertainty_args=self.pre_transform_args.get('uncertainty_params', {}),
                norm_dict=self.norm_dict
            )
        else:
            self.pre_transform = None

    def forward(self, x, edge_index, batch, edge_attr, edge_weight, cond=None):
        """Forward pass through the model.

        Args:
            x: Node features
            edge_index: Graph connectivity
            batch: Batch assignment vector
            edge_attr: Edge attributes
            edge_weight: Edge weights
            cond: Optional conditioning variables

        Returns:
            Flow context vector for the normalizing flow
        """
        flow_context = self.embedding_nn(
            x, edge_index, batch=batch,
            edge_attr=edge_attr, edge_weight=edge_weight, cond=cond
        )

        return flow_context

    def _prepare_batch(self, batch):
        """Prepare batch data for training/validation.

        Args:
            batch: Raw batch from dataloader (PyG Data object)

        Returns:
            Dictionary containing all necessary batch data:
            - 'x': Node features
            - 'edge_index': Edge connectivity
            - 'batch': Batch assignment
            - 'theta': Target parameters for posterior estimation
            - 'edge_attr': Optional edge attributes
            - 'edge_weight': Optional edge weights
            - 'cond': Optional conditioning variables
            - 'batch_size': Number of graphs in batch
        """
        # Apply pre-transforms if specified
        if self.pre_transform is not None:
            batch = self.pre_transform(batch)

        # Extract data from PyG Data object
        batch_dict = {
            'x': batch.x,
            'edge_index': batch.edge_index,
            'batch': batch.batch,
            'theta': batch.theta,
            'edge_attr': batch.edge_attr if hasattr(batch, 'edge_attr') else None,
            'edge_weight': batch.edge_weight if hasattr(batch, 'edge_weight') else None,
            'cond': batch.cond if hasattr(batch, 'cond') else None,
            'batch_size': batch.num_graphs if hasattr(batch, 'num_graphs') else batch.batch.max().item() + 1,
        }
        return batch_dict

    def training_step(self, batch, batch_idx):
        """Training step for PyTorch Lightning.

        Args:
            batch: Training batch
            batch_idx: Batch index

        Returns:
            Training loss
        """
        batch_dict = self._prepare_batch(batch)

        # forward pass
        flow_context = self.forward(
            batch_dict['x'],
            batch_dict['edge_index'],
            batch=batch_dict['batch'],
            edge_attr=batch_dict['edge_attr'],
            edge_weight=batch_dict['edge_weight'],
            cond=batch_dict['cond']
        )
        log_prob = self.flows(flow_context).log_prob(batch_dict['theta'])
        loss = -log_prob.mean()

        # log the loss
        self.log(
            'train/loss', loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_dict['batch_size'],
            sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for PyTorch Lightning.

        Args:
            batch: Validation batch
            batch_idx: Batch index

        Returns:
            Validation loss
        """
        batch_dict = self._prepare_batch(batch)

        # forward pass
        flow_context = self.forward(
            batch_dict['x'], batch_dict['edge_index'],
            batch=batch_dict['batch'],
            edge_attr=batch_dict['edge_attr'],
            edge_weight=batch_dict['edge_weight'],
            cond=batch_dict['cond']
        )
        log_prob = self.flows(flow_context).log_prob(batch_dict['theta'])
        loss = -log_prob.mean()

        # log the loss
        self.log(
            'val/loss', loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_dict['batch_size'],
            sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        """Initialize optimizer and LR scheduler."""
        return models.configure_optimizers(
            self.parameters(), self.optimizer_args, self.scheduler_args)

    @torch.no_grad()
    def sample_from_batch(self, batch, num_samples):
        raise NotImplementedError("Use the standalone sample function instead.")

    def sample_from_loader(self, loader, num_samples):
        raise NotImplementedError("Use the standalone sample function instead.")


@torch.no_grad()
def sample(model, data_loader, num_posteriors=100, transforms=None, norm_dict=None, to_numpy=True):
    """Sample from the posterior distribution using the trained model.

    Args:
        model: Trained NPE model
        data_loader: DataLoader containing the input data
        num_posteriors: Number of posterior samples to draw per input
        transforms: Optional data transformations to apply
        norm_dict: Optional normalization dictionary to denormalize outputs
        to_numpy: Whether to convert outputs to numpy arrays

    Returns:
        Tuple of (posteriors, truths) as numpy arrays or torch tensors
    """
    model.eval()
    device = model.device
    posteriors = []
    truths = []
    for batch in tqdm(data_loader):
        if transforms is not None:
            batch = transforms(batch)
        batch = batch.to(device)
        cond = batch.cond if hasattr(batch, 'cond') else None
        edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None
        edge_weight = batch.edge_weight if hasattr(batch, 'edge_weight') else None

        flow_context = model(
            batch.x, batch.edge_index, batch=batch.batch,
            edge_attr=edge_attr, edge_weight=edge_weight, cond=cond
        )
        posterior = model.flows(flow_context).sample(
            (num_posteriors, ))
        # Shape: (batch_size, num_posteriors, num_features)
        posterior = posterior.transpose(0, 1).cpu()
        posteriors.append(posterior)
        truths.append(batch.theta.cpu())
    posteriors = torch.cat(posteriors, dim=0)
    truths = torch.cat(truths, dim=0)

    if to_numpy:
        posteriors = posteriors.numpy()
        truths = truths.numpy()

    if norm_dict is not None:
        posteriors = posteriors * norm_dict['theta_scale'] + norm_dict['theta_loc']
        truths = truths * norm_dict['theta_scale'] + norm_dict['theta_loc']

    return posteriors, truths