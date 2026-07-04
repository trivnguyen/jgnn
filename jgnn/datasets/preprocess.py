
import torch
from torch_geometric.data import Data

def create_graph_from_posvel(pos, vel, vel_error=None, label=None, cond=None):
    """
    Build a bare PyG graph holding raw Cartesian phase-space data.

    No node features (`x`) are computed here: `GetNodeFeatures`, together
    with any projection/selection/uncertainty transforms, builds `x` from
    `pos`/`vel` later in the pre_transform pipeline (see
    `jgnn.transforms.build_transformation`).
    """
    # convert to tensor
    pos = torch.tensor(pos, dtype=torch.float32)
    vel = torch.tensor(vel, dtype=torch.float32)
    if vel_error is not None:
        vel_error = torch.tensor(vel_error, dtype=torch.float32)
    if label is not None:
        label = torch.tensor(label, dtype=torch.float32)
    if cond is not None:
        cond = torch.tensor(cond, dtype=torch.float32)

    # make sure the dimensions are correct
    if pos.dim() == 1:
        pos = pos.view(-1, 1)
    if vel.dim() == 1:
        vel = vel.view(-1, 1)
    if vel_error is not None and vel_error.dim() == 1:
        vel_error = vel_error.view(-1, 1)
    if label is not None and label.dim() == 1:
        label = label.view(1, -1)
    if cond is not None and cond.dim() == 1:
        cond = cond.view(1, -1)

    return Data(pos=pos, vel=vel, vel_error=vel_error, theta=label, cond=cond)
