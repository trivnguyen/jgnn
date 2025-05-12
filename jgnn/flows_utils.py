
from typing import Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn

import zuko
from zuko.flows import (
    Flow,
    MaskedAutoregressiveTransform,
    Unconditional,
)
from zuko.distributions import DiagNormal


def build_flows(
    features: int, context_features: int, num_transforms: int,
    hidden_features: List[int], num_bins: int, activation: Callable,
    randperm: bool = True
):
    """ Build neural spline flow

    Parameters
    ----------
    features : int
        Number of features
    context_features : int
        Number of context features
    num_transforms : int
        Number of flow transforms
    hidden_features : List[int]
        Number of hidden features of the MLP
    num_bins : int
        Number of bins of the spline
    activation : Callable
        Activation function of the MLP
    randperm : bool
        Whether to apply random permutation to the features
    """
    transforms = []
    for i in range(num_transforms):
        order = torch.arange(features)
        if randperm:
            order = order[torch.randperm(order.size(0))]
        shapes = ([num_bins], [num_bins], [num_bins - 1])
        transform = zuko.flows.MaskedAutoregressiveTransform(
            features=features, context=context_features,
            univariate=zuko.transforms.MonotonicRQSTransform,
            shapes=shapes, hidden_features=hidden_features, order=order,
            activation=activation,
        )
        transforms.append(transform)

    flow = zuko.flows.Flow(
        transform=transforms,
        base=Unconditional(
            DiagNormal, torch.zeros(features), torch.ones(features), buffer=True)
    )
    return flow
