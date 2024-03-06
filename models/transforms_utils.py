
import torch
from torch_geometric import transforms as T

def build_transformation(graph_name: str, graph_params: dict):
    """ Build the transformation pipeline
    """
    transforms = []

    # move graph to CPU because graph construction is not supported on GPU
    transforms.append(T.ToDevice(device=torch.device("cpu")))

    # graph transformation
    if graph_name.lower() == "knn":
        transforms.append(T.KNNGraph(**graph_params))
    elif graph_name.lower() == "radius":
        transforms.append(T.RadiusGraph(**graph_params))
    else:
        raise ValueError(f"Unknown graph name: {graph_name}")

    transforms = T.Compose(transforms)
    return transforms