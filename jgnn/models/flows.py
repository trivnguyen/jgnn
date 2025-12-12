
from typing import Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn

import zuko
from zuko.flows import (
    Flow,
    MaskedAutoregressiveTransform,
    UnconditionalTransform,
)
from zuko.distributions import DiagNormal


def build_flows(
    features: int,
    context_features: int,
    num_transforms: int,
    hidden_features: List[int],
    activation: Callable,
    flow_type: str = "spline",
    num_bins: Optional[int] = None,
    randperm: bool = True,
    dropout: float = 0.0,
    residual: bool = False
):
    """ Build normalizing flow (spline or MAF)

    Parameters
    ----------
    features : int
        Number of features
    context_features : int
        Number of context features
    num_transforms : int
        Number of flow transforms
    hidden_features : List[int]
        Number of hidden features of the MADE network
    activation : Callable
        Activation function of the MADE network
    flow_type : str
        Type of flow to use: 'spline' for Neural Spline Flow or 'maf' for
        Masked Autoregressive Flow (affine). Default is 'spline'.
    num_bins : Optional[int]
        Number of bins of the spline (required when flow_type='spline',
        ignored for flow_type='maf'). Default is None.
    randperm : bool
        Whether to apply random permutation to the features. Default is True.
    dropout : float
        Dropout probability in the MADE network. Default is 0.0 (no dropout).
    residual : bool
        Whether to use residual connections in the MADE network. Default is False.
    """
    if flow_type == "spline" and num_bins is None:
        raise ValueError("num_bins must be specified when flow_type='spline'")

    transforms = []
    for i in range(num_transforms):
        order = torch.arange(features)
        if randperm:
            order = order[torch.randperm(order.size(0))]

        if flow_type == "spline":
            shapes = ([num_bins], [num_bins], [num_bins - 1])
            transform = zuko.flows.MaskedAutoregressiveTransform(
                features=features, context=context_features,
                univariate=zuko.transforms.MonotonicRQSTransform,
                shapes=shapes, hidden_features=hidden_features, order=order,
                activation=activation,
                # dropout=dropout, residual=residual,
            )
        elif flow_type == "maf":
            # Standard MAF with affine transforms
            shapes = ([], [])  # scale and shift parameters
            transform = zuko.flows.MaskedAutoregressiveTransform(
                features=features, context=context_features,
                univariate=zuko.transforms.AffineTransform,
                shapes=shapes, hidden_features=hidden_features, order=order,
                activation=activation,
                #  dropout=dropout, residual=residual,
            )
        else:
            raise ValueError(f"Unknown flow_type: {flow_type}. Must be 'spline' or 'maf'.")

        transforms.append(transform)

    flow = zuko.flows.Flow(
        transform=transforms,
        base=UnconditionalTransform(
            DiagNormal, torch.zeros(features), torch.ones(features), buffer=True)
    )
    return flow
