
import torch
from torch_geometric import transforms as T


def build_transformation(
    graph_name: str, graph_params: dict, random_projection: bool = False):
    """ Build the transformation pipeline """
    transforms = []

    # move graph to CPU because graph construction is not supported on GPU
    transforms.append(T.ToDevice(device=torch.device("cpu")))

    # apply random projection and get the node features
    if random_projection:
        transforms.append(RandomProjection())
        transforms.append(GetNodeFeatures())

    # graph transformation
    if graph_name.lower() == "knn":
        transforms.append(T.KNNGraph(**graph_params))
    elif graph_name.lower() == "radius":
        transforms.append(T.RadiusGraph(**graph_params))
    else:
        raise ValueError(f"Unknown graph name: {graph_name}")

    transforms = T.Compose(transforms)
    return transforms

def random_rotation_matrix():
    # Generate a random quaternion
    q = torch.randn(4)
    q /= torch.norm(q)  # Normalize the quaternion

    # Convert quaternion to rotation matrix
    q0, q1, q2, q3 = q.unbind()
    R = torch.tensor([
        [1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q3*q0, 2*q1*q3 + 2*q2*q0],
        [2*q1*q2 + 2*q3*q0, 1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q1*q0],
        [2*q1*q3 - 2*q2*q0, 2*q2*q3 + 2*q1*q0, 1 - 2*q1**2 - 2*q2**2]
    ])
    return R

class RandomProjection:
    """ Apply a random projection to the input data """
    def __init__(self):
        pass

    def __call__(self, data):
        data = data.clone()

        # create the random projection matrix
        R = random_rotation_matrix()

        # apply rotation to position and velocity
        pos_proj = torch.matmul(data.pos, R)
        vel_proj = torch.matmul(data.vel, R)

        # apply the projection by removing the last dimension
        pos_proj = pos_proj[:, :2]
        vel_proj = vel_proj[:, 2].unsqueeze(1)

        # update the data
        data.pos = pos_proj
        data.vel = vel_proj

        return data

class GetNodeFeatures:
    """ Extract node features from the input data """
    def __init__(self, x_loc=0, x_scale=1):
        # Convert inputs to tensors if they aren't already
        if not isinstance(x_loc, torch.Tensor):
            self.x_loc = torch.tensor(x_loc, requires_grad=False)
            self.x_scale = torch.tensor(x_scale, requires_grad=False)
        else:
            self.x_loc = x_loc.detach()
            self.x_scale = x_scale.detach()

    def __call__(self, data):
        data = data.clone()
        rad = torch.norm(data.pos, dim=1).unsqueeze(1)
        vel = torch.norm(data.vel, dim=1).unsqueeze(1)
        log_rad = torch.log10(rad + 1e-6)
        log_vel = torch.log10(vel + 1e-6)
        x = torch.cat([log_rad, log_vel], dim=1)
        x = (x - self.x_loc.to(x.device)) / self.x_scale.to(x.device)
        data.x = x
        return data
