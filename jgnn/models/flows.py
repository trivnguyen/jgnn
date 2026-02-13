
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
    flow_type: str = "spline",
    **kwargs
):
    """ Build normalizing flow (spline, MAF, or CNF)

    Parameters
    ----------
    features : int
        Number of features
    context_features : int
        Number of context features
    num_transforms : int
        Number of flow transforms (unused for CNF)
    flow_type : str
        Type of flow: 'spline' for Neural Spline Flow, 'maf' for Masked
        Autoregressive Flow (affine), or 'cnf' for Continuous Normalizing
        Flow (FFJORD). Default is 'spline'.
    **kwargs
        Additional keyword arguments for the flow transforms, such as:
        - num_bins: int (for spline flows)
        - hidden_features: sequence of int
        - activation: callable activation function
        - randperm: bool
        - freqs: int (for CNF, number of time embedding frequencies)
        - exact: bool (for CNF, exact vs stochastic log-det Jacobian)
        - atol: float (for CNF, absolute integration tolerance)
        - rtol: float (for CNF, relative integration tolerance)
    """

    # Common MLP kwargs shared across all flow types
    mlp_kwargs = {}
    for key in ("hidden_features", "activation"):
        if key in kwargs:
            mlp_kwargs[key] = kwargs[key]

    if flow_type in ["spline", "maf"]:
        randperm = kwargs.get("randperm", False)

        if flow_type == "spline":
            num_bins = kwargs.get("num_bins", 8)
            univariate = zuko.transforms.MonotonicRQSTransform
            shapes = ([num_bins], [num_bins], [num_bins - 1])
        else:  # maf
            univariate = zuko.transforms.AffineTransform
            shapes = ([], [])

        transforms = []
        for i in range(num_transforms):
            order = torch.arange(features)
            if randperm:
                order = order[torch.randperm(order.size(0))]

            transform = zuko.flows.MaskedAutoregressiveTransform(
                features=features, context=context_features,
                univariate=univariate, shapes=shapes, order=order,
                **mlp_kwargs,
            )
            transforms.append(transform)

        flow = zuko.flows.Flow(
            transform=transforms,
            base=UnconditionalTransform(
                DiagNormal, torch.zeros(features), torch.ones(features), buffer=True)
        )
    elif flow_type == "cnf":
        cnf_kwargs = {}
        for key in ("freqs", "exact", "atol", "rtol"):
            if key in kwargs:
                cnf_kwargs[key] = kwargs[key]
        flow = zuko.flows.continuous.CNF(
            features=features,
            context=context_features,
            **mlp_kwargs,
            **cnf_kwargs,
        )
    else:
        raise ValueError(
            f"Unknown flow_type: {flow_type}. Must be 'spline', 'maf', or 'cnf'."
        )

    return flow
