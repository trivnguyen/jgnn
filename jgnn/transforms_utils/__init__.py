
import torch
from torch_geometric import transforms as T

from .basic import GetNodeFeatures
from .projection import RandomProjection
from .selection_function import RadialSelectionFunction, RandomSelectionStrategy
from .selection_function import ExponentialSelectionFunction, LinearSelectionFunction

def build_transformation(
    graph_name: str, graph_params: dict, random_projection: bool = False,
    selection: bool = False, selection_args: dict = None,
    norm_dict = None
):
    """ Build the transformation pipeline """
    transforms = []

    # move graph to CPU because graph construction is not supported on GPU
    transforms.append(T.ToDevice(device=torch.device("cpu")))

    # apply random projection if specified
    if random_projection:
        transforms.append(RandomProjection())

    # apply radial selection if specified
    if selection:
        # transforms.append(RadialSelectionFunction(**selection_args))
        transforms.append(RandomSelectionStrategy(**selection_args))

    # if random projection or selection is applied, we need to re-compute node features
    # otherwise, we assume that node features are already computed
    if random_projection or selection:
        if norm_dict is None:
            transforms.append(GetNodeFeatures())
        else:
            transforms.append(GetNodeFeatures(norm_dict['x_loc'], norm_dict['x_scale']))

    # graph transformation
    if graph_name.lower() == "knn":
        transforms.append(T.KNNGraph(**graph_params))
    elif graph_name.lower() == "radius":
        transforms.append(T.RadiusGraph(**graph_params))
    else:
        raise ValueError(f"Unknown graph name: {graph_name}")

    transforms = T.Compose(transforms)

    return transforms

