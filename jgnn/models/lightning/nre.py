"""Neural Ratio Estimation (NRE) module for graph-based inference."""

from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from ml_collections import ConfigDict
import emcee
import numpy as np

from ..utils import get_activation, configure_optimizers


class NRE(pl.LightningModule):
    """Neural Ratio Estimation model for graph-structured data.

    This model combines an embedding network and a classifier network
    to perform likelihood ratio estimation on graph-structured data.
    The embedding network is initialized externally and passed to the model,
    allowing for flexible architecture choices (e.g., GNN, Transformer, etc.).

    The model is trained using binary cross-entropy loss on positive (real) pairs
    of (embedding, theta) and negative (shuffled) pairs where theta is randomly
    permuted relative to the embedding.

    Parameters
    ----------
    input_size : int
        Size of input features
    output_size : int
        Size of output (target) features
    classifier_hidden : list of int, optional
        Hidden layer sizes for the classifier network. Default: [64, 64]
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
    activation : str, default='relu'
        Activation function for classifier network
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        classifier_hidden: list = None,
        embedding_nn: nn.Module = None,
        optimizer_args: ConfigDict = None,
        scheduler_args: ConfigDict = None,
        norm_dict: Dict[str, Any] = None,
        pre_transforms = None,
        activation: str = 'relu',
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.classifier_hidden = classifier_hidden or [64, 64]
        self.embedding_nn = embedding_nn
        self.pre_transforms = pre_transforms
        self.optimizer_args = optimizer_args or {}
        self.scheduler_args = scheduler_args or {}
        self.norm_dict = norm_dict
        self.activation_name = activation

        self.save_hyperparameters(
            ignore=['embedding_nn', 'pre_transforms'])

        self._setup_model()

    def _setup_model(self):
        """Set up all model components including classifier network."""

        # if embedding_nn is not provided, assume identity mapping
        if self.embedding_nn is None:
            self.embedding_nn = nn.Identity()
            embedding_output_size = self.input_size
        else:
            embedding_output_size = self.embedding_nn.output_size

        # Build classifier network: takes [embedding, theta] and outputs logit
        input_dim = embedding_output_size + self.output_size

        layers = []
        prev_dim = input_dim

        for hidden_dim in self.classifier_hidden:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(get_activation(self.activation_name, return_instance=True))
            prev_dim = hidden_dim

        # Final layer outputs a single logit for binary classification
        layers.append(nn.Linear(prev_dim, 1))

        self.classifier = nn.Sequential(*layers)

        print(f"[NRE] Classifier network built with"
              f" {sum(p.numel() for p in self.classifier.parameters()):,} parameters")

    def forward(self, batch):
        """Forward pass through the embedding network."""
        return self.embedding_nn(self.embedding_nn._prepare_batch(batch))

    def log_prob_ratio(self, embedding: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Compute log probability ratio log(p(x|theta) / p(x)).

        The classifier learns d = P(real pair | embedding, theta).
        The log probability ratio is: log(d / (1-d)) = log(P(x|theta) / P(x))

        For numerical stability, we use the fact that:
        log(sigmoid(z) / (1-sigmoid(z))) = z (the logit itself)

        Parameters
        ----------
        embedding : torch.Tensor
            Embedding of the observed data, shape (batch_size, embedding_dim)
        theta : torch.Tensor
            Parameters, shape (batch_size, output_size)

        Returns
        -------
        torch.Tensor
            Log probability ratio, shape (batch_size,)
        """
        # Concatenate embedding and theta
        combined = torch.cat([embedding, theta], dim=-1)
        # Get logit from classifier
        logit = self.classifier(combined).squeeze(-1)
        # log_ratio = log(d / (1-d)) where d = sigmoid(logit)
        # This equals the logit for numerical stability
        return logit

    def _prepare_batch(self, batch):
        """Prepare batch data for training/validation."""
        if self.pre_transforms is not None:
            batch = self.pre_transforms(batch)
        return batch

    def _compute_bce_loss(self, embedding: torch.Tensor, theta: torch.Tensor):
        """Compute BCE loss and accuracy for ratio estimation.

        Creates positive (real) and negative (shuffled) pairs and computes
        binary cross-entropy loss.

        Parameters
        ----------
        embedding : torch.Tensor
            Embeddings, shape (batch_size, embedding_dim)
        theta : torch.Tensor
            Parameters, shape (batch_size, output_size)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (loss, accuracy)
        """
        batch_size = embedding.shape[0]

        # Create positive samples (real pairs)
        pos_combined = torch.cat([embedding, theta], dim=-1)
        pos_logits = self.classifier(pos_combined).squeeze(-1)

        # Create negative samples (shuffled pairs)
        # the following makes sure no theta is paired with its own embedding
        probs = torch.ones((batch_size, batch_size)) * (1 - torch.eye(batch_size)) / (batch_size - 1)
        indices = torch.multinomial(probs, 1, replacement=False).squeeze(-1)
        neg_combined = torch.cat([embedding, theta[indices]], dim=-1)
        neg_logits = self.classifier(neg_combined).squeeze(-1)

        # Binary cross-entropy loss
        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)

        all_logits = torch.cat([pos_logits, neg_logits])
        all_labels = torch.cat([pos_labels, neg_labels])

        loss = F.binary_cross_entropy_with_logits(all_logits, all_labels)

        # Compute accuracy for monitoring
        with torch.no_grad():
            predictions = (all_logits > 0).float()
            accuracy = (predictions == all_labels).float().mean()

        return loss, accuracy

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
        theta = batch.theta

        # Compute loss and accuracy
        loss, accuracy = self._compute_bce_loss(embedding, theta)

        # Log metrics
        self.log(
            'train/loss', loss, on_step=True, on_epoch=True, prog_bar=True,
            logger=True, batch_size=batch_size, sync_dist=True
        )
        self.log(
            'train/accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True,
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
        theta = batch.theta

        # Compute loss and accuracy
        loss, accuracy = self._compute_bce_loss(embedding, theta)

        # Log metrics
        self.log(
            'val/loss', loss, on_step=False, on_epoch=True, prog_bar=True,
            logger=True, batch_size=batch_size, sync_dist=True
        )
        self.log(
            'val/accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True,
            logger=True, batch_size=batch_size, sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        """Initialize optimizer and LR scheduler."""
        return configure_optimizers(
            self.parameters(), self.optimizer_args, self.scheduler_args)

    @torch.no_grad()
    def sample(
        self,
        embedding: torch.Tensor,
        num_samples: int,
        num_walkers: int = 32,
        num_steps: int = 1000,
        burn_in: int = 100,
        theta_init: Optional[np.ndarray] = None,
    ):
        """Sample from the posterior distribution using MCMC (emcee).

        Args:
            embedding: Embedding of a single observation, shape (embedding_dim,)
            num_samples: Number of posterior samples to draw
            num_walkers: Number of MCMC walkers (default: 32)
            num_steps: Number of MCMC steps (default: 1000)
            burn_in: Number of burn-in steps to discard (default: 100)
            theta_init: Initial theta value for walkers, shape (output_size,)
                       If None, will initialize randomly from standard normal

        Returns:
            np.ndarray: Posterior samples of shape (num_samples, output_size)
        """
        self.eval()

        # Ensure embedding is on the correct device and has batch dimension
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)  # (1, embedding_dim)

        embedding = embedding.to(self.device)

        # Define log probability function for emcee
        def log_prob_fn(theta_array):
            """Log probability function for a single data point.

            Args:
                theta_array: numpy array of shape (n_walkers, output_size)

            Returns:
                numpy array of shape (n_walkers,)
            """
            theta_tensor = torch.from_numpy(theta_array).float().to(self.device)
            # Expand embedding to match number of walkers
            emb_expanded = embedding.expand(theta_tensor.shape[0], -1)
            log_ratio = self.log_prob_ratio(emb_expanded, theta_tensor)
            return log_ratio.cpu().numpy()

        # Initialize walkers
        if theta_init is not None:
            # Use provided initial theta
            # Add small noise to create different walker positions
            pos = theta_init + 1e-3 * np.random.randn(num_walkers, self.output_size)
        else:
            # Random initialization from standard normal
            pos = np.random.randn(num_walkers, self.output_size)

        # Create MCMC sampler
        sampler = emcee.EnsembleSampler(
            num_walkers, self.output_size, log_prob_fn
        )

        # Run MCMC
        sampler.run_mcmc(pos, num_steps + burn_in, progress=False)

        # Get samples after burn-in
        chain = sampler.get_chain(discard=burn_in, flat=True)  # (num_walkers * num_steps, output_size)

        # Randomly select num_samples from the chain
        indices = np.random.choice(chain.shape[0], size=num_samples, replace=False)
        samples = chain[indices]  # (num_samples, output_size)

        return samples
