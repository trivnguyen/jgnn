
import torch
from torch_geometric import transforms as T


def build_transformation(
    graph_name: str, graph_params: dict, random_projection: bool = False,
    selection: bool = False, selection_args: dict = None,
    norm_dict = None
):
    """ Build the transformation pipeline """
    transforms = []

    # move graph to CPU because graph construction is not supported on GPU
    transforms.append(T.ToDevice(device=torch.device("cpu")))

    # apply random projection and get the node features
    if random_projection:
        transforms.append(RandomProjection())
    if selection:
        transforms.append(RadialSelectionFunction(**selection_args))

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

class RadialSelectionFunction:
    """ Rough selection function """
    def __init__(self, q_min, q_max, mode):
        self.q_min = q_min
        self.q_max = q_max
        self.mode = mode

        # check if q_min and q_max are valid
        if not (0 <= q_min <= 1):
            raise ValueError(f"q_min should be in [0, 1], but got {q_min}")
        if not (0 <= q_max <= 1):
            raise ValueError(f"q_max should be in [0, 1], but got {q_max}")
        if q_min >= q_max:
            raise ValueError(f"q_min should be smaller than q_max, but got {q_min} >= {q_max}")

    def __call__(self, batch):
        batch = batch.clone()
        n_per_batch = batch.ptr[1:] - batch.ptr[:-1]
        n_graph = batch.num_graphs
        q_vals = torch.rand(n_graph) * (self.q_max - self.q_min) + self.q_min

        radii = torch.norm(batch.pos, dim=1)
        radii_q = []
        for i in range(n_graph):
            rad = radii[batch.ptr[i]:batch.ptr[i + 1]]
            rad_q = torch.quantile(rad, q=q_vals[i])
            radii_q.append(rad_q)
        radii_q = torch.stack(radii_q, dim=0)

        diff = radii - torch.repeat_interleave(radii_q, n_per_batch)

        if self.mode == 'low':
            # No sign change - select nodes with negative differences (radii < quantile)
            mask = diff <= 0
        elif self.mode == 'high':
            # Reverse sign - select nodes with positive differences (radii > quantile)
            # But we'll negate the difference, so we're looking for diff >= 0
            mask = diff >= 0
        elif self.mode == 'random':
            # Randomly reverse sign
            sign = torch.repeat_interleave(torch.rand(n_graph) - 0.5, n_per_batch)
            mask = diff * sign <= 0
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # apply mask
        batch.x = batch.x[mask]
        batch.pos = batch.pos[mask]
        batch.vel = batch.vel[mask]
        batch.batch = batch.batch[mask]
        batch.ptr = torch.searchsorted(batch.batch, torch.arange(n_graph+1))
        return batch