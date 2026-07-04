
import torch
from torch_geometric import transforms as T

from .basic import GetNodeFeatures, Normalize
from .graph import ALL_GRAPHS, AdaptiveKNNGraph
from .projection import RandomProjection
from .selection_function import (
    BaseSelectionFunction,
    RadialSelectionFunction,
    RandomSelectionStrategy,
    ExponentialSelectionFunction,
    LinearSelectionFunction,
)
from .uncertainty import UncertaintySampler

__all__ = [
    'GetNodeFeatures',
    'Normalize',
    'ALL_GRAPHS',
    'AdaptiveKNNGraph',
    'RandomProjection',
    'BaseSelectionFunction',
    'RadialSelectionFunction',
    'RandomSelectionStrategy',
    'ExponentialSelectionFunction',
    'LinearSelectionFunction',
    'UncertaintySampler',
    'build_transformation',
]

def build_transformation(
    apply_graph: bool = True,
    apply_projection: bool = False,
    apply_selection: bool = False,
    apply_uncertainty: bool = False,
    graph_name: str = 'KNN',
    graph_args: dict = None,
    projection_args: dict = None,
    selection_args: dict = None,
    uncertainty_args = None,
    norm_dict = None,
    use_log_features: bool = True
):
    """
    Build a transformation pipeline for graph data.

    `uncertainty_args` accepts either a single dict (one uncertainty
    transform applied to a single feature) or a list of dicts to chain
    multiple `UncertaintySampler` transforms, each with its own
    `distribution_type`, `feature_idx`, and parameters. This is useful when
    `apply_projection` is configured with `use_proper_motions=True`, so that
    the line-of-sight velocity and the two proper-motion components can
    each be assigned a different uncertainty distribution.
    """

    transforms = []
    transforms.append(T.ToDevice(device=torch.device("cpu")))  # not gpu-supported yet

    # Apply random projection and/or selection function
    if apply_projection:
        transforms.append(RandomProjection(**projection_args))
    if apply_selection:
        if selection_args is None:
            raise ValueError('`selection_args` must be provided when `apply_selection` is True.')
        # transforms.append(RadialSelectionFunction(**selection_args))
        transforms.append(RandomSelectionStrategy(**selection_args))
    if apply_projection or apply_selection:
        # only recompute node features if projection or selection is applied
        transforms.append(GetNodeFeatures(log=use_log_features))

    # Apply uncertainty sampling
    if apply_uncertainty:
        if uncertainty_args is None:
            raise ValueError('`uncertainty_args` must be provided when `apply_uncertainty` is True.')
        # allow a single dict or a list of dicts, one per feature (e.g. one
        # per velocity coordinate), each with its own uncertainty distribution
        if isinstance(uncertainty_args, dict):
            uncertainty_args = [uncertainty_args]
        for args in uncertainty_args:
            transforms.append(UncertaintySampler(**args))

    # Normalizing node features
    if norm_dict is not None:
        print(f"Applying normalization with provided norm_dict: {norm_dict}")
        transforms.append(Normalize(norm_dict['x_loc'], norm_dict['x_scale']))

    # Apply graph transformation, connect edges based on the specified graph type
    # set to False for no graph construction (e.g., for Transformer models)
    if apply_graph:
        if graph_name.lower() not in ALL_GRAPHS:
            raise ValueError(f"Unknown graph name: {graph_name}. Supported graphs: {list(ALL_GRAPHS.keys())}")
        transforms.append(ALL_GRAPHS[graph_name.lower()](**graph_args))

    transforms = T.Compose(transforms)
    return transforms
