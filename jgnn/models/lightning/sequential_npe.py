"""Sequential Neural Posterior Estimation (SNPE) with importance weighting.

This module implements SNPE-C (Greenberg et al., 2019) which uses importance
weighting to correctly account for data generated from different proposal
distributions across rounds.
"""

from typing import List, Dict, Any, Tuple, Optional, Callable

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch_geometric.transforms as T
from ml_collections import ConfigDict

from jgnn import flows_utils, transforms_utils, models_utils, models


class SequentialNPE(pl.LightningModule):
    """Sequential NPE with importance weighting (SNPE-C).

    This extends the standard NPE to handle data from multiple rounds with
    different proposal distributions. The loss is weighted by importance
    weights to ensure the learned posterior is correct.

    For round r, the importance weight for a sample (x, θ) is:
        w(θ, x) = p(θ) / q_r(θ|x)

    where:
    - p(θ) is the prior
    - q_r(θ|x) is the proposal distribution from round r

    References
    ----------
    Greenberg, D., Nonnenmacher, M., & Macke, J. (2019).
    Automatic Posterior Transformation for Likelihood-Free Inference. ICML.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        featurizer_args: ConfigDict,
        mlp_args: ConfigDict,
        flows_args: ConfigDict,
        pre_transform_args: ConfigDict = None,
        optimizer_args: ConfigDict = None,
        scheduler_args: ConfigDict = None,
        conditional_mlp_args: ConfigDict = None,
        norm_dict: Dict[str, Any] = None,
        freeze_components: List[str] = None,
        # Sequential NPE specific args
        use_importance_weighting: bool = True,
        prior_log_prob_fn: Optional[Callable] = None,
        proposal_posteriors: Optional[List[str]] = None,
        current_round: int = 0,
        clip_weights: Optional[float] = None,
    ):
        """Initialize Sequential NPE.

        Parameters
        ----------
        use_importance_weighting : bool
            Whether to use importance weighting. If False, behaves like standard NPE.
        prior_log_prob_fn : Callable, optional
            Function that computes log p(θ) for the prior.
            If None, assumes uniform prior (no weighting).
        proposal_posteriors : List[str], optional
            List of checkpoint paths for proposal posteriors from previous rounds.
            Used to compute importance weights.
        current_round : int
            Current training round (0-indexed). Round 0 has no importance weighting.
        clip_weights : float, optional
            If provided, clip importance weights to [1/clip_weights, clip_weights].
            Helps prevent numerical instability.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.featurizer_args = featurizer_args
        self.mlp_args = mlp_args
        self.flows_args = flows_args
        self.pre_transform_args = pre_transform_args
        self.optimizer_args = optimizer_args or {}
        self.scheduler_args = scheduler_args or {}
        self.conditional_mlp_args = conditional_mlp_args or {}
        self.norm_dict = norm_dict
        self.freeze_components = freeze_components or []

        # Sequential NPE specific
        self.use_importance_weighting = use_importance_weighting
        self.prior_log_prob_fn = prior_log_prob_fn
        self.current_round = current_round
        self.clip_weights = clip_weights

        self.save_hyperparameters(ignore=['prior_log_prob_fn', 'proposal_posteriors'])

        self._setup_model()

        # Load proposal posteriors for importance weighting
        self.proposal_posteriors = []
        if proposal_posteriors is not None and len(proposal_posteriors) > 0:
            self._load_proposal_posteriors(proposal_posteriors)

    def _freeze_module(self, module: nn.Module):
        for param in module.parameters():
            param.requires_grad = False

    def _setup_model(self):
        """Set up the neural network components."""

        # create the featurizer (GNN)
        if self.featurizer_args.name == 'gnn':
            activation_fn = models_utils.get_activation(self.featurizer_args.activation)
            self.featurizer = models.GNN(
                input_size=self.input_size,
                hidden_sizes=self.featurizer_args.hidden_sizes,
                projection_size=self.featurizer_args.projection_size,
                graph_layer=self.featurizer_args.graph_layer,
                graph_layer_params=self.featurizer_args.graph_layer_params,
                activation_fn=activation_fn,
                pooling=self.featurizer_args.pooling,
                layer_norm=self.featurizer_args.layer_norm,
                norm_first=self.featurizer_args.norm_first,
            )
        else:
            raise ValueError(
                f'Featurizer {self.featurizer_args.name} not supported')

        # create the mlp layers
        activation_fn = models_utils.get_activation(self.mlp_args.activation)
        self.mlp = models.MLPBatchNorm(
            input_size=self.featurizer_args.hidden_sizes[-1],
            hidden_sizes=self.mlp_args.hidden_sizes,
            output_size=self.mlp_args.output_size,
            activation_fn=activation_fn,
            batch_norm=self.mlp_args.batch_norm,
            dropout=self.mlp_args.dropout,
        )

        # create the conditional mlp layers for extra conditioning
        if self.conditional_mlp_args is not None:
            activation_fn = models_utils.get_activation(self.conditional_mlp_args.activation)
            self.conditional_mlp = models.MLPBatchNorm(
                input_size=self.conditional_mlp_args.input_size,
                hidden_sizes=self.conditional_mlp_args.hidden_sizes,
                output_size=self.conditional_mlp_args.output_size,
                activation_fn=activation_fn,
                batch_norm=self.conditional_mlp_args.batch_norm,
                dropout=self.conditional_mlp_args.dropout,
            )
        else:
            self.conditional_mlp = None

        # create the flows
        activation_fn = models_utils.get_activation_zuko(self.flows_args.activation)
        self.flows = flows_utils.build_flows(
            context_features=self.mlp_args.output_size,
            features=self.output_size,
            hidden_features=self.flows_args.hidden_sizes,
            num_transforms=self.flows_args.num_transforms,
            num_bins=self.flows_args.num_bins,
            activation=activation_fn,
            randperm=True
        )

        # create pre-transforms
        self.pre_transform = transforms_utils.build_transformation(
            graph_name=self.pre_transform_args.graph_name,
            graph_params=self.pre_transform_args.graph_params,
            random_projection=self.pre_transform_args.get('random_projection', False),
            selection=self.pre_transform_args.get('selection', False),
            selection_args=self.pre_transform_args.get('selection_params', {}),
            uncertainty=self.pre_transform_args.get('uncertainty', False),
            uncertainty_args=self.pre_transform_args.get('uncertainty_params', {}),
            norm_dict=self.norm_dict
        )

        # freeze components if needed
        for component in self.freeze_components:
            if hasattr(self, component):
                print(f'Freezing component: {component}')
                self._freeze_module(getattr(self, component))
            else:
                raise ValueError(f'Component {component} not found in the model.')

    def _load_proposal_posteriors(self, proposal_checkpoints: List[str]):
        """Load proposal posteriors from previous rounds for importance weighting.

        Parameters
        ----------
        proposal_checkpoints : List[str]
            List of checkpoint paths for proposal posteriors.
        """
        print(f"Loading {len(proposal_checkpoints)} proposal posteriors for importance weighting...")

        for i, ckpt_path in enumerate(proposal_checkpoints):
            # Load checkpoint
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            hparams = checkpoint['hyper_parameters']

            # Create proposal model (same architecture)
            from jgnn.models import NPE as NPEModel
            proposal = NPEModel(
                input_size=hparams['input_size'],
                output_size=hparams['output_size'],
                featurizer_args=hparams['featurizer_args'],
                mlp_args=hparams['mlp_args'],
                flows_args=hparams['flows_args'],
                pre_transform_args=hparams['pre_transform_args'],
                norm_dict=hparams.get('norm_dict'),
            )

            # Load weights
            proposal.load_state_dict(checkpoint['state_dict'])
            proposal.eval()
            proposal.requires_grad_(False)  # Freeze all parameters

            self.proposal_posteriors.append(proposal)

        print(f"Loaded {len(self.proposal_posteriors)} proposal posteriors")

    def _prepare_batch(self, batch):
        """Prepare the batch for the model."""
        if self.pre_transform is not None:
            batch = self.pre_transform(batch)
        batch = batch.to(self.device)
        if not hasattr(batch, 'cond'):
            cond = None
        else:
            cond = batch.cond

        # Check if batch has round indices
        if hasattr(batch, 'round_idx'):
            round_idx = batch.round_idx
        else:
            round_idx = None

        batch_dict = {
            'x': batch.x,
            'theta': batch.theta,
            'edge_index': batch.edge_index,
            'edge_attr': batch.edge_attr,
            'edge_weight': batch.edge_weight,
            'batch': batch.batch,
            'batch_size': len(batch),
            'cond': cond,
            'round_idx': round_idx,
        }
        return batch_dict

    def forward(self, x, edge_index, batch, edge_attr, edge_weight, cond=None):
        """Forward pass through the network to get flow context."""
        flow_context = self.featurizer(
            x, edge_index, batch=batch, edge_attr=edge_attr, edge_weight=edge_weight)
        flow_context = self.mlp(flow_context)

        if self.conditional_mlp_args is not None:
            cond = self.conditional_mlp(cond)
            flow_context = flow_context + cond

        return flow_context

    def compute_importance_weights(
        self,
        theta: torch.Tensor,
        flow_context: torch.Tensor,
        round_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute importance weights for each sample using atomic proposals (SNPE-C).

        For samples from round r, the importance weight is:
            w = p(θ) / q_r(θ|x)

        where q_r is the SPECIFIC proposal that generated this sample (atomic proposal).
        This follows the SNPE-C algorithm (Greenberg et al., 2019).

        Parameters
        ----------
        theta : torch.Tensor
            Parameter values, shape (batch_size, output_size)
        flow_context : torch.Tensor
            Flow context from the neural network, shape (batch_size, context_size)
        round_idx : torch.Tensor, optional
            Round index for each sample, shape (batch_size,).
            If None, assumes all samples are from round 0 (prior).

        Returns
        -------
        weights : torch.Tensor
            Importance weights, shape (batch_size,)
        """
        batch_size = theta.shape[0]

        # If no importance weighting or round 0, return uniform weights
        if not self.use_importance_weighting or self.current_round == 0:
            return torch.ones(batch_size, device=theta.device)

        # If no proposal posteriors loaded, return uniform weights
        if len(self.proposal_posteriors) == 0:
            return torch.ones(batch_size, device=theta.device)

        # Compute log prior
        if self.prior_log_prob_fn is not None:
            log_prior = self.prior_log_prob_fn(theta)
        else:
            # Assume uniform prior in normalized space
            log_prior = torch.zeros(batch_size, device=theta.device)

        # Compute log proposal density using ATOMIC proposal approach
        # For each sample, use the specific proposal that generated it
        with torch.no_grad():
            log_proposal = torch.zeros(batch_size, device=theta.device)

            if round_idx is None:
                # If no round indices provided, assume all from prior (round 0)
                # Samples from prior have weight = 1.0
                return torch.ones(batch_size, device=theta.device)

            # For each sample, evaluate under its specific proposal
            for sample_idx in range(batch_size):
                r = int(round_idx[sample_idx].item())

                if r == 0:
                    # Sample from prior - no proposal density needed
                    # Weight will be p(θ) / p(θ) = 1
                    log_proposal[sample_idx] = log_prior[sample_idx]
                elif r > 0 and r <= len(self.proposal_posteriors):
                    # Sample from round r-1 proposal (0-indexed)
                    proposal = self.proposal_posteriors[r - 1].to(theta.device)

                    try:
                        # Evaluate log q_r(θ|x) for this specific sample
                        log_q = proposal.flows(flow_context[sample_idx:sample_idx+1]).log_prob(
                            theta[sample_idx:sample_idx+1]
                        )
                        log_proposal[sample_idx] = log_q.squeeze()
                    except Exception as e:
                        print(f"Warning: Could not evaluate proposal for round {r}: {e}")
                        # Fall back to uniform weight for this sample
                        log_proposal[sample_idx] = log_prior[sample_idx]
                else:
                    # Invalid round index - use uniform weight
                    print(f"Warning: Invalid round index {r}, using uniform weight")
                    log_proposal[sample_idx] = log_prior[sample_idx]

        # Importance weight: w = p(θ) / q_r(θ|x)
        log_weights = log_prior - log_proposal

        # Convert to linear weights
        weights = torch.exp(log_weights)

        # Clip weights if specified
        if self.clip_weights is not None:
            weights = torch.clamp(
                weights,
                min=1.0 / self.clip_weights,
                max=self.clip_weights
            )

        # Normalize weights (optional, but helps with stability)
        # weights = weights / weights.mean()

        return weights

    def training_step(self, batch, batch_idx):
        batch_dict = self._prepare_batch(batch)

        # Forward pass
        flow_context = self.forward(
            batch_dict['x'], batch_dict['edge_index'],
            batch=batch_dict['batch'],
            edge_attr=batch_dict['edge_attr'],
            edge_weight=batch_dict['edge_weight'],
            cond=batch_dict['cond']
        )

        # Compute log probability
        log_prob = self.flows(flow_context).log_prob(batch_dict['theta'])

        # Compute importance weights for sequential NPE
        weights = self.compute_importance_weights(
            batch_dict['theta'],
            flow_context,
            batch_dict['round_idx']
        )

        # Weighted loss
        weighted_log_prob = weights * log_prob
        loss = -weighted_log_prob.mean()

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True,
                prog_bar=True, batch_size=batch_dict['batch_size'])
        self.log('train_log_prob', log_prob.mean(), on_step=False, on_epoch=True)
        self.log('train_weight_mean', weights.mean(), on_step=False, on_epoch=True)
        self.log('train_weight_std', weights.std(), on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        batch_dict = self._prepare_batch(batch)

        # Forward pass
        flow_context = self.forward(
            batch_dict['x'], batch_dict['edge_index'],
            batch=batch_dict['batch'],
            edge_attr=batch_dict['edge_attr'],
            edge_weight=batch_dict['edge_weight'],
            cond=batch_dict['cond']
        )

        # Compute log probability
        log_prob = self.flows(flow_context).log_prob(batch_dict['theta'])

        # Compute importance weights
        weights = self.compute_importance_weights(
            batch_dict['theta'],
            flow_context,
            batch_dict['round_idx']
        )

        # Weighted loss
        weighted_log_prob = weights * log_prob
        loss = -weighted_log_prob.mean()

        # Log metrics
        self.log('val_loss', loss, on_step=True, on_epoch=True,
                prog_bar=True, batch_size=batch_dict['batch_size'])
        self.log('val_log_prob', log_prob.mean(), on_step=False, on_epoch=True)
        self.log('val_weight_mean', weights.mean(), on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Initialize optimizer and LR scheduler."""
        return models_utils.configure_optimizers(
            self.parameters(), self.optimizer_args, self.scheduler_args)
