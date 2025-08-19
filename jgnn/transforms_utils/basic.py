
import torch

class GetNodeFeatures:
    """ Extract node features from the input batch """
    def __init__(self, log=True):
        self.log = log

    def __call__(self, batch):
        batch = batch.clone()
        rad = torch.norm(batch.pos, dim=1).unsqueeze(1)
        if self.log:
            rad = torch.log10(rad + 1e-6)
        x = torch.cat([rad, batch.vel], dim=1)
        batch.x = x
        return batch

class Normalize:
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
        x_loc = self.x_loc.to(batch.x.device)
        x_scale = self.x_scale.to(batch.x.device)

        # If UncertaintySampler is called, it will increase the number of features by 1
        # to account for this, we add a feature to x_loc and x_scale if mismatch in dimensions
        # x_loc and x_scale should be 0 and 1 respectively for the additional feature
        # TODO: This is a temporary fix, should be handled better in the future
        if x_loc.shape[0] != batch.x.shape[1]:
            x_loc = torch.cat([x_loc, torch.zeros(1, device=x_loc.device)])
            x_scale = torch.cat([x_scale, torch.ones(1, device=x_scale.device)])

        batch.x = (batch.x - x_loc) / x_scale
        return batch
