"""Neural Posterior Estimation (NPE) module for graph-based inference."""

from typing import Dict, Any
import copy

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

    Parameters
    ----------
    input_size : int
        Size of input features
    output_size : int
        Size of output (target) features
    flows_args : ConfigDict
        Configuration for building the flow (num_transforms, hidden_features, etc.)
    embedding_nn : nn.Module, optional
        Pre-built embedding network. If None, uses Identity mapping.
    optimizer_args : ConfigDict, optional
        Optimizer configuration
    scheduler_args : ConfigDict, optional
        Scheduler configuration
    norm_dict : Dict[str, Any], optional
        Normalization dictionary for data preprocessing
    pre_transforms : optional
        Data transformations to apply before forward pass
    init_flows_from_embedding : bool, default=False
        If True and embedding_nn has a 'flow' attribute, initialize NPE.flows
        from embedding_nn.flow. The flows will be deep-copied and gradients
        will be enabled even if embedding_nn is frozen.
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
        init_flows_from_embedding: bool=False,
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
        self.init_flows_from_embedding = init_flows_from_embedding

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

        # Check if we should initialize flows from embedding_nn.flow
        if (
            self.init_flows_from_embedding
            and hasattr(self.embedding_nn, "flow")
            and self.embedding_nn.flow is not None
        ):
            print("[NPE] Initializing flows from embedding_nn.flow")
            # Create a copy of the embedding flow to avoid sharing parameters
            self.flows = copy.deepcopy(self.embedding_nn.flow)

            # Ensure all parameters in flows have gradients enabled
            # even if the embedding_nn was frozen
            for param in self.flows.parameters():
                param.requires_grad = True

            print(f"[NPE] Flows initialized from embedding with"
                  f" {sum(p.numel() for p in self.flows.parameters()):,} parameters")
        else:
            # create the flow from scratch
            # Get activation function
            activation_fn = get_activation(
                self.flows_args.get('activation', 'tanh'),
                self.flows_args.get('activation_args', None),
                return_instance=False
            )

            # Build the flow
            self.flows = build_flows(
                features=self.output_size,
                context_features=embedding_output_size,
                num_transforms=self.flows_args.get('num_transforms', 4),
                hidden_features=self.flows_args.get('hidden_features', [32, 32]),
                num_bins=self.flows_args.get('num_bins', 8),
                activation=activation_fn,
                flow_type=self.flows_args.get('flow_type', 'spline'),
                randperm=self.flows_args.get('randperm', True),
                dropout=self.flows_args.get('dropout', 0.0),
                residual=self.flows_args.get('residual', False)
            )
            print(f"[NPE] Flows built from scratch with"
                  f" {sum(p.numel() for p in self.flows.parameters()):,} parameters")

    def forward(self, batch):
        """ Forward pass through the embedding network. """
        return self.embedding_nn(self.embedding_nn._prepare_batch(batch))

    def log_prob(self, embedding, theta):
        """ Evaluate the flow log probability. """
        return self.flows(embedding).log_prob(theta)

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
        log_prob = self.log_prob(embedding, batch.theta)
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
        log_prob = self.log_prob(embedding, batch.theta)
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

        # Apply pre-transforms if provided, else fall back to model's pre_transforms
        if pre_transforms is not None:
            batch = pre_transforms(batch)
        elif self.pre_transforms is not None:
            batch = self.pre_transforms(batch)

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
