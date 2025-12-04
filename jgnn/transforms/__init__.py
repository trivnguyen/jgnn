
import torch
from torch_geometric import transforms as T

from .basic import GetNodeFeatures, Normalize
from .projection import RandomProjection
from .selection_function import RadialSelectionFunction, RandomSelectionStrategy
from .selection_function import ExponentialSelectionFunction, LinearSelectionFunction
from .uncertainty import UncertaintySampler

def build_transformation(
    graph_name: str, graph_params: dict, random_projection: bool = False,
    selection: bool = False, selection_args: dict = None,
    uncertainty: bool = False, uncertainty_args: dict = None,
    norm_dict = None, log: bool = True
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
        if selection_args is None:
            raise ValueError("Selection arguments must be provided when selection is enabled.")
        transforms.append(RandomSelectionStrategy(**selection_args))

    # if random projection or selection is applied, we need to re-compute node features
    # otherwise, we assume that node features are already computed
    if random_projection or selection:
        transforms.append(GetNodeFeatures(log=log))

    # add uncertainty
    if uncertainty:
        if uncertainty_args is None:
            raise ValueError("Uncertainty arguments must be provided when uncertainty is enabled.")
        transforms.append(UncertaintySampler(**uncertainty_args))

    # normalize
    if norm_dict is not None:
        transforms.append(Normalize(norm_dict['x_loc'], norm_dict['x_scale']))

    # connect edges based on the specified graph type
    if graph_name.lower() == "knn":
        transforms.append(T.KNNGraph(**graph_params))
    elif graph_name.lower() == "radius":
        transforms.append(T.RadiusGraph(**graph_params))
    else:
        raise ValueError(f"Unknown graph name: {graph_name}")

    transforms = T.Compose(transforms)

    return transforms

