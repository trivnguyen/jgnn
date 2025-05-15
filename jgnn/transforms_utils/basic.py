
import torch

class GetNodeFeatures:
    """ Extract node features from the input batch """
    def __init__(self, x_loc=0, x_scale=1):
        # Convert inputs to tensors if they aren't already
        if not isinstance(x_loc, torch.Tensor):
            self.x_loc = torch.tensor(x_loc, requires_grad=False)
            self.x_scale = torch.tensor(x_scale, requires_grad=False)
        else:
            self.x_loc = x_loc.detach()
            self.x_scale = x_scale.detach()

    def __call__(self, batch):
        batch = batch.clone()
        rad = torch.norm(batch.pos, dim=1).unsqueeze(1)
        vel = torch.norm(batch.vel, dim=1).unsqueeze(1)
        log_rad = torch.log10(rad + 1e-6)
        log_vel = torch.log10(vel + 1e-6)
        x = torch.cat([log_rad, log_vel], dim=1)
        x = (x - self.x_loc.to(x.device)) / self.x_scale.to(x.device)
        batch.x = x
        return batch
