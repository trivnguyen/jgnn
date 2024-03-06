
import torch
from torch_geometric.data import Data

def create_graph_from_posvel(
    pos, vel, vel_error=None, label=None, cond=None,
    use_vel_error=False):
    """
    Create a PyG graph from position and velocity data.
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

    # create a PyG graph
    rad = torch.norm(pos, dim=1).view(-1, 1)
    log_rad = torch.log10(rad + 1e-6)

    if vel_error is not None and use_vel_error:
        x = torch.cat([log_rad, vel, vel_error], dim=1)
    else:
        x = torch.cat([log_rad,  vel], dim=1)
    graph = Data(x=x, theta=label, pos=pos, cond=cond)

    return graph
