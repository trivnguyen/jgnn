"""Sequential Neural Posterior Estimation (SNPE-C)

This module implements Sequential NPE using Automatic Posterior Transformation (APT, also known as SNPE-C)
with atomic proposals.
References
----------
Greenberg, D., Nonnenmacher, M., & Macke, J. (2019).
Automatic Posterior Transformation for Likelihood-Free Inference. ICML.
"""

from typing import List, Dict, Any, Tuple, Optional, Callable

import torch
import torch.nn as nn
import pytorch_lightning as pl
from ml_collections import ConfigDict
from tqdm import tqdm

from ..utils import get_activation, configure_optimizers
from ..flows import build_flows


class SequentialNPE(pl.LightningModule):
    """Sequential NPE using Automatic Posterior Transformation (APT, also known as SNPE-C)
    with atomic proposals.

    This extends the standard NPE to handle data from multiple rounds with
    different proposal distributions.

    References
    ----------
    Greenberg, D., Nonnenmacher, M., & Macke, J. (2019).
    Automatic Posterior Transformation for Likelihood-Free Inference. ICML.
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
        # Sequential NPE specific args
        prior: Optional[Callable]=None,
        proposal: Optional[List[str]]=None,
        current_round: int = 0,
        num_atoms: int = 2,
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

        # Sequential NPE specific
        self.prior = prior
        self.proposal = proposal
        self.current_round = current_round
        self.num_atoms = num_atoms

        self.save_hyperparameters(
            ignore=['embedding_nn', 'pre_transforms', 'prior', 'proposal'])

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

    def set_proposal(self, proposal):
        """ Set the proposal distribution for the current round. """
        self.proposal = proposal

    def set_round(self, current_round):
        """ Set the current round number. """
        self.current_round = current_round

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

    def _log_prob_proposal_posterior_atomic(self, embedding, batch):
        """ Implementation of log posterior computation using atomic proposals based on
        Greenberg et al. (2019) and the `sbi` library.
        """
        batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else batch.batch_size

        if self.current_round == 0:
            # First round: use prior as proposal
            return self.log_prob(embedding, batch.theta)

        # Subsequent rounds: use atomic proposals loss
        # number of atoms from 2 to batch_size
        num_atoms = min(max(self.num_atoms, 2), batch_size)

        # randomly sample the thetas for atomic proposals
        probs = torch.ones((batch_size, batch_size)) * (1 - torch.eye(batch_size)) / (batch_size - 1)
        indices = torch.multinomial(probs, num_atoms - 1, replacement=False)
        theta_proposals = batch.theta[indices]  # (batch_size, num_atoms, theta_dim)

        # prepare atomic thetas and embeddings
        atomic_theta = torch.cat([batch.theta.unsqueeze(1), theta_proposals], axis=1)
        atomic_theta = atomic_theta.reshape(batch_size * num_atoms, -1)
        embedding_rep = embedding.unsqueeze(1).repeat(1, num_atoms, 1)
        embedding_rep = embedding_rep.reshape(batch_size * num_atoms, -1)

        # TODO: Fix this: self.prior.log_prob does not handle device properly
        # Evaluate the log proposal posterior
        # compute log priors and log posteriors
        # if self.prior is None:
            # assume uniform prior if not provided
        log_priors = torch.zeros((batch_size, num_atoms), device=batch.theta.device)
        # else:
            # log_priors = self.prior.log_prob(atomic_theta)
            # log_priors = log_priors.reshape(batch_size, num_atoms)

        log_prob_posterior = self.log_prob(embedding_rep, atomic_theta)
        log_prob_posterior = log_prob_posterior.reshape(batch_size, num_atoms)

        # Compute unnormalized proposal posterior.
        unnormalized_log_prob = log_prob_posterior - log_priors
        log_prob_proposal_posterior = unnormalized_log_prob[:, 0] - torch.logsumexp(
            unnormalized_log_prob, dim=-1
        )

        return log_prob_proposal_posterior

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

        # Compute loss as the log posterior
        log_prob = self._log_prob_proposal_posterior_atomic(embedding, batch)
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
        log_prob = self._log_prob_proposal_posterior_atomic(embedding, batch)
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

        # Apply pre-transforms if provided
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

    def build_posterior(self):
        """Build a posterior object that can be used for sampling.

        Returns:
            DirectPosterior: A posterior object with a sample method
        """
        return DirectPosterior(self, prior=self.prior)


class DirectPosterior:
    """Wrapper class for the posterior distribution that provides a sample method.

    This class wraps a trained SequentialNPE model and provides an interface
    for sampling from the posterior distribution conditioned on a single observation.
    """

    def __init__(self, model: SequentialNPE, prior: Optional[Callable]=None):
        """Initialize the DirectPosterior.

        Args:
            model: Trained SequentialNPE model
        """
        self.model = model
        self.prior = prior

    @torch.no_grad()
    def sample(self, num_samples, x_obs, norm_dict=None, max_iterations=1000):
        """Sample from the posterior distribution given a single observation.

        Args:
            num_samples: Number of posterior samples to draw
            x_obs: Single observation input data
            norm_dict: Dictionary containing normalization parameters
            max_iterations: Maximum number of sampling iterations to prevent infinite loops
        Returns:
            torch.Tensor: Posterior samples of shape (num_samples, output_size)
        """
        self.model.eval()

        samples = []
        total_generated = 0
        current_samples = 0
        iteration = 0

        while current_samples < num_samples:
            if iteration >= max_iterations:
                raise RuntimeError(
                    f"Reached maximum iterations ({max_iterations}). "
                    f"Only generated {current_samples}/{num_samples} samples. "
                    f"Overall acceptance rate: {current_samples/total_generated:.3f}"
                )

            # Determine batch size for this iteration
            remaining = num_samples - current_samples
            current_batch = num_samples // 10

            # Sample from the model
            batch_samples = self.model.sample_from_batch(x_obs, current_batch)
            batch_samples = batch_samples.squeeze(0)  # remove batch dimension

            total_generated += current_batch

            # Denormalize
            if norm_dict is not None:
                theta_scale = torch.tensor(norm_dict['theta_scale'], device=batch_samples.device)
                theta_loc = torch.tensor(norm_dict['theta_loc'], device=batch_samples.device)
                batch_samples = batch_samples * theta_scale + theta_loc

            # Apply truncation if prior is given
            if self.prior is not None:
                within_bounds = self.prior.is_within_bounds(batch_samples)

                # Print periodically
                if iteration % print_every == 0:
                    acceptance_fraction = torch.sum(within_bounds) / len(batch_samples)
                    print(f"[Truncation] Iteration {iteration}: Batch acceptance fraction: {acceptance_fraction:.3f} "
                          f"({torch.sum(within_bounds)}/{len(batch_samples)} samples)")

                batch_samples = batch_samples[within_bounds]
                samples.append(batch_samples)
                current_samples += batch_samples.shape[0]
            else:
                samples.append(batch_samples)
                current_samples += batch_samples.shape[0]

            iteration += 1

        if self.prior is not None:
            acceptance_fraction = current_samples / total_generated
            print(f"[Truncation] Final acceptance fraction: {acceptance_fraction:.3f} "
                  f"({current_samples}/{total_generated} samples)")

        samples = torch.cat(samples, dim=0)[:num_samples]
        return samples