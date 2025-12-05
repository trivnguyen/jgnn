"""Utility functions for models."""

from functools import partial
from typing import Optional, Dict, Callable, Any
import math

import torch
import torch.nn as nn

from .flows import build_flows

def get_activation(
    act: str, act_args: Optional[Dict] = None, return_instance: bool = True
) -> Callable[..., nn.Module]:
    """Get an activation callable (class or factory). If args is provided, returns a
    callable that will instantiate the activation with those kwargs (using functools.partial)."""

    key = act.lower().replace('-', '').replace('_', '')
    mapping = {
        'identity': nn.Identity,
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'gelu': nn.GELU,
        'silu': nn.SiLU,
        'leakyrelu': nn.LeakyReLU,
        'elu': nn.ELU,
        'prelu': nn.PReLU,
        'selu': nn.SELU,
    }

    act_cls = mapping.get(key)
    if act_cls is None:
        raise ValueError(f'Unknown activation function: {act}')

    # return instance or class/partial
    # zuko requires class, but torch nn modules usually want instances
    if return_instance:
        return act_cls(**act_args) if act_args else act_cls()
    return partial(act_cls, **act_args) if act_args else act_cls

class WarmUpCosineAnnealingLR(torch.optim.lr_scheduler.LambdaLR):
    """Cosine annealing learning rate scheduler with warmup.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimizer
    decay_steps : int
        Initial number of steps for the first cosine cycle (T_0)
    warmup_steps : int
        Number of warmup steps at the beginning
    eta_min : float, default=0
        Minimum learning rate multiplier
    last_epoch : int, default=-1
        The index of last epoch
    restart : bool, default=False
        Whether to restart the cosine annealing after decay_steps
    T_mult : float, default=1
        Multiplicative factor for increasing cycle length after each restart.
        After each restart, the next cycle length becomes: T_i = T_{i-1} * T_mult
        Only used when restart=True. T_mult=1 gives constant cycle length.
    """

    def __init__(self, optimizer, decay_steps, warmup_steps, eta_min=0, last_epoch=-1, restart=False, T_mult=1):
        self.T_0 = decay_steps  # Initial period
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        self.restart = restart
        self.T_mult = T_mult
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def _get_cycle_length(self, cycle_idx):
        """Calculate the length of a given cycle."""
        if self.T_mult == 1:
            return self.T_0
        return int(self.T_0 * (self.T_mult ** cycle_idx))

    def _find_cycle(self, step):
        """Find which cycle the current step belongs to and position within that cycle.

        Returns
        -------
        tuple
            (cycle_idx, step_in_cycle, cycle_length)
        """
        if not self.restart or self.T_mult == 1:
            # Simple case: fixed cycle length or no restart
            cycle_idx = step // self.T_0
            step_in_cycle = step % self.T_0
            return cycle_idx, step_in_cycle, self.T_0

        # Variable cycle length case
        cumulative_steps = 0
        cycle_idx = 0

        while True:
            cycle_length = self._get_cycle_length(cycle_idx)
            if cumulative_steps + cycle_length > step:
                # Found the cycle
                step_in_cycle = step - cumulative_steps
                return cycle_idx, step_in_cycle, cycle_length
            cumulative_steps += cycle_length
            cycle_idx += 1

    def lr_lambda(self, step):
        if not self.restart and step >= self.decay_steps:
            # No restart: clamp to final step
            step_in_cycle = self.decay_steps
            cycle_length = self.decay_steps
        else:
            # Find current cycle and position
            _, step_in_cycle, cycle_length = self._find_cycle(step)

        # Warmup phase
        if step_in_cycle < self.warmup_steps:
            return float(step_in_cycle) / float(max(1, self.warmup_steps))

        # Cosine annealing phase
        progress = (step_in_cycle - self.warmup_steps) / (cycle_length - self.warmup_steps)
        return self.eta_min + 0.5 * (1 + math.cos(math.pi * progress))


def configure_optimizers(parameters, optimizer_args, scheduler_args=None):
    """Configure optimizer and scheduler for PyTorch Lightning.

    Parameters
    ----------
    parameters : iterable
        Model parameters to optimize
    optimizer_args : ConfigDict
        Optimizer configuration with 'name', 'lr', 'weight_decay'
    scheduler_args : ConfigDict, optional
        Scheduler configuration with 'name' and scheduler-specific args

    Returns
    -------
    dict or Optimizer
        If scheduler is specified, returns dict with 'optimizer' and 'lr_scheduler'.
        Otherwise returns optimizer only.
    """
    scheduler_args = scheduler_args or {}

    # Setup optimizer
    if optimizer_args.name == "Adam":
        optimizer = torch.optim.Adam(
            parameters,
            lr=optimizer_args.lr,
            weight_decay=optimizer_args.weight_decay
        )
    elif optimizer_args.name == "AdamW":
        optimizer = torch.optim.AdamW(
            parameters,
            lr=optimizer_args.lr,
            weight_decay=optimizer_args.weight_decay
        )
    else:
        raise NotImplementedError(f"Optimizer {optimizer_args.name} not implemented")

    # Setup scheduler
    if scheduler_args.name is None:
        return optimizer

    if scheduler_args.name == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min',
            factor=scheduler_args.factor,
            patience=scheduler_args.patience
        )
        # ReduceLROnPlateau needs a metric to monitor (default: validation loss)
        monitor = scheduler_args.get('monitor', 'val/loss')
    elif scheduler_args.name == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_args.T_max,
            eta_min=scheduler_args.eta_min
        )
        # Step-based schedulers don't need a monitor metric
        monitor = None
    elif scheduler_args.name == 'WarmUpCosineAnnealingLR':
        scheduler = WarmUpCosineAnnealingLR(
            optimizer,
            decay_steps=scheduler_args.decay_steps,
            warmup_steps=scheduler_args.warmup_steps,
            eta_min=scheduler_args.eta_min,
            restart=scheduler_args.get('restart', False),
            T_mult=scheduler_args.get('T_mult', 1)
        )
        # Step-based schedulers don't need a monitor metric
        monitor = None
    else:
        raise NotImplementedError(f"Scheduler {scheduler_args.name} not implemented")

    lr_scheduler_config = {
        'scheduler': scheduler,
        'interval': scheduler_args.interval,
        'frequency': 1
    }

    # Only add monitor if scheduler needs it (e.g., ReduceLROnPlateau)
    if monitor is not None:
        lr_scheduler_config['monitor'] = monitor

    return {
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler_config
    }


def build_embedding_loss(loss_type: str, loss_args: Optional[Dict[str, Any]] = None):
    """Build a loss function for embedding models.

    Parameters
    ----------
    loss_type : str
        Type of loss function. Options: 'mse', 'flow'
    loss_args : dict, optional
        Additional arguments for the loss function
        For 'flow' loss, should contain flow configuration:
            - features: int, number of output features
            - context_features: int, number of context/embedding features
            - num_transforms: int, number of flow transformations
            - hidden_features: list of int, hidden layer sizes
            - num_bins: int, number of bins for spline transforms
            - activation: str, activation function name
            - activation_args: dict, optional args for activation
            - randperm: bool, whether to use random permutation (default: True)

    Returns
    -------
    tuple
        (loss_fn, flow_model) where:
        - loss_fn: Callable that takes (embedding, target) and returns loss
        - flow_model: Flow model if loss_type=='flow', else None

    Examples
    --------
    MSE Loss:
    >>> loss_fn, _ = build_embedding_loss('mse')
    >>> loss = loss_fn(embedding, target)

    Flow Loss (Variational Information):
    >>> flow_args = {
    ...     'features': 10,
    ...     'context_features': 128,
    ...     'num_transforms': 4,
    ...     'hidden_features': [64, 64],
    ...     'num_bins': 8,
    ...     'activation': 'relu'
    ... }
    >>> loss_fn, flow = build_embedding_loss('flow', flow_args)
    >>> loss = loss_fn(embedding, target)  # embedding is context, target is data
    """
    loss_args = loss_args or {}
    loss_type = loss_type.lower()

    if loss_type == 'mse':
        # Simple MSE loss
        mse_loss = nn.MSELoss()
        def mse_loss_fn(embedding, target):
            return mse_loss(embedding, target)
        return mse_loss_fn, None

    elif loss_type == 'vmim':
        # Flow-based Variational Mutual Information Maximization (VMIM)
        features = loss_args.get('features')
        context_features = loss_args.get('context_features')
        num_transforms = loss_args.get('num_transforms', 4)
        hidden_features = loss_args.get('hidden_features', [32, 32])
        num_bins = loss_args.get('num_bins', 8)
        activation_name = loss_args.get('activation', 'relu')
        activation_args = loss_args.get('activation_args', None)
        randperm = loss_args.get('randperm', True)

        if features is None or context_features is None:
            raise ValueError(
                "Flow loss requires 'features' and 'context_features' in loss_args"
            )

        # Get activation function
        activation_fn = get_activation(activation_name, activation_args, return_instance=False)

        # Build the flow
        flow = build_flows(
            features=features,
            context_features=context_features,
            num_transforms=num_transforms,
            hidden_features=hidden_features,
            num_bins=num_bins,
            activation=activation_fn,
            randperm=randperm
        )

        def flow_loss_fn(embedding, target):
            """Compute negative log-likelihood loss.

            Parameters
            ----------
            embedding : torch.Tensor
                Context/embedding from the model [batch_size, context_features]
            target : torch.Tensor
                Target values to model [batch_size, features]

            Returns
            -------
            torch.Tensor
                Negative log-likelihood loss (scalar)
            """
            # Condition flow on embedding and evaluate log_prob of target
            log_prob = flow(embedding).log_prob(target)
            return -log_prob.mean()

        return flow_loss_fn, flow

    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            "Supported types: 'mse', 'flow'"
        )
