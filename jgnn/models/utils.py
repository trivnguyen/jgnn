"""Utility functions for models."""

from functools import partial
import math

import torch
import torch.nn as nn


def get_activation(activation):
    """Get an activation function class or partial (returns class, not instance)."""

    def _leaky_relu():
        alpha = activation.get('leaky_relu_alpha', 0.01)
        return partial(nn.LeakyReLU, negative_slope=alpha)

    activations = {
        'identity': nn.Identity,
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'leaky_relu': _leaky_relu(),
        'gelu': nn.GELU,
        'silu': nn.SiLU,
    }

    name = activation.name.lower()
    if name not in activations:
        raise ValueError(f'Unknown activation function: {activation.name}')

    return activations[name]


class WarmUpCosineAnnealingLR(torch.optim.lr_scheduler.LambdaLR):
    """Cosine annealing learning rate scheduler with warmup."""

    def __init__(self, optimizer, decay_steps, warmup_steps, eta_min=0, last_epoch=-1, restart=False):
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        self.restart = restart
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step >= self.decay_steps:
            if self.restart:
                step = step % self.decay_steps
            else:
                step = self.decay_steps
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return self.eta_min + (
            0.5 * (1 + math.cos(math.pi * (step - self.warmup_steps) / (self.decay_steps - self.warmup_steps))))


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
    elif scheduler_args.name == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_args.T_max,
            eta_min=scheduler_args.eta_min
        )
    elif scheduler_args.name == 'WarmUpCosineAnnealingLR':
        scheduler = WarmUpCosineAnnealingLR(
            optimizer,
            decay_steps=scheduler_args.decay_steps,
            warmup_steps=scheduler_args.warmup_steps,
            eta_min=scheduler_args.eta_min,
            restart=scheduler_args.get('restart', False)
        )
    else:
        raise NotImplementedError(f"Scheduler {scheduler_args.name} not implemented")

    return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': scheduler,
            'monitor': 'train_loss',
            'interval': scheduler_args.interval,
            'frequency': 1
        }
    }
