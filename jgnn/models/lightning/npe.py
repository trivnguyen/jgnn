"""Neural Posterior Estimation (NPE) module for graph-based inference."""

from typing import Dict, Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
from ml_collections import ConfigDict
from tqdm import tqdm

from ..utils import get_activation, configure_optimizers
from ..flows import build_flows

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
        optimizer_args: ConfigDict=None,
        scheduler_args: ConfigDict=None,
        norm_dict: Dict[str, Any]=None,
        pre_transforms=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.flows_args = flows_args
        self.embedding_nn = embedding_nn
        self.pre_transforms = pre_transforms
        self.optimizer_args = optimizer_args or {}
        self.scheduler_args = scheduler_args or {}
        self.norm_dict = norm_dict

        self.save_hyperparameters(
            ignore=['embedding_nn', 'pre_transforms'])

        self._setup_model()

    def _setup_model(self):
        """Set up all model components including flows and pre-transforms."""

        # if embedding_nn is not provided, assume identity mapping
        if self.embedding_nn is None:
            self.embedding_nn = nn.Identity()
            embedding_output_size = self.input_size
        else:
            embedding_output_size = self.embedding_nn.output_size

        # create the flow
        features = self.output_size
        context_features = embedding_output_size
        num_transforms = self.flows_args.get('num_transforms', 4)
        hidden_features = self.flows_args.get('hidden_features', [32, 32])
        num_bins = self.flows_args.get('num_bins', 8)
        activation_name = self.flows_args.get('activation', 'tanh')
        activation_args = self.flows_args.get('activation_args', None)
        randperm = self.flows_args.get('randperm', True)

        # Get activation function
        activation_fn = get_activation(
            activation_name, activation_args, return_instance=False)

        # Build the flow
        self.flows = build_flows(
            features=features,
            context_features=context_features,
            num_transforms=num_transforms,
            hidden_features=hidden_features,
            num_bins=num_bins,
            activation=activation_fn,
            randperm=randperm
        )

    def forward(self, batch):
        return self.embedding_nn(self.embedding_nn._prepare_batch(batch))

    def _prepare_batch(self, batch):
        """Prepare batch data for training/validation."""
        if self.pre_transforms is not None:
            batch = self.pre_transforms(batch)
        return batch

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
        batch = self._prepare_batch(batch)
        batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else batch.batch_size
        embedding = self.forward(batch)

        # Compute loss
        log_prob = self.flows(embedding).log_prob(batch.theta)
        loss = -log_prob.mean()

        # Log metrics
        # Use hierarchical naming for better organization in loggers like WandB/TensorBoard
        self.log(
            'train/loss', loss, on_step=True, on_epoch=True, prog_bar=True,
            logger=True, batch_size=batch_size, sync_dist=True
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
        batch = self._prepare_batch(batch)
        batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else batch.batch_size
        embedding = self.forward(batch)

        # Compute loss
        log_prob = self.flows(embedding).log_prob(batch.theta)
        loss = -log_prob.mean()

        # Log metrics
        # Use hierarchical naming for better organization in loggers like WandB/TensorBoard
        self.log(
            'val/loss', loss, on_step=False, on_epoch=True, prog_bar=True,
            logger=True, batch_size=batch_size, sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        """Initialize optimizer and LR scheduler."""
        return configure_optimizers(
            self.parameters(), self.optimizer_args, self.scheduler_args)

    @torch.no_grad()
    def sample_from_batch(self, batch, num_samples, pre_transforms=None):
        """Sample from the posterior distribution for a given batch.

        Args:
            batch: Input batch data
            num_samples: Number of posterior samples to draw per input
            pre_transforms: Optional data transformations to apply. If given,
                            these will override the model's pre_transforms.
        Returns:
            torch.Tensor: Posterior samples of shape (batch_size, num_samples, output_size)
        """
        self.eval()
        if pre_transforms is not None:
            batch = pre_transforms(batch)
        batch = batch.to(self.device)
        embedding = self.forward(batch)
        posterior = self.flows(embedding).sample((num_samples, ))  # (num_samples, batch_size, output_size)
        posterior = posterior.transpose(0, 1) # (batch_size, num_samples, output_size)
        return posterior

    @torch.no_grad()
    def sample_from_loader(self, loader, num_samples, pre_transforms=None, verbose=True):
        """Sample from the posterior distribution for all data in a DataLoader.

        Args:
            loader: DataLoader containing the input data
            num_samples: Number of posterior samples to draw per input
            pre_transforms: Optional data transformations to apply. If given,
                            these will override the model's pre_transforms.
            verbose: Whether to display a progress bar
        Returns:
            torch.Tensor: Posterior samples of shape (num_data, num_samples, output_size)
        """
        self.eval()
        posteriors = []
        for batch in tqdm(loader, disable=not verbose):
            posterior = self.sample_from_batch(
                batch, num_samples, pre_transforms=pre_transforms)
            posteriors.append(posterior.cpu())
        posteriors = torch.cat(posteriors, dim=0)
        return posteriors
