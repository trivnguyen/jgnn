
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
    """ Normalize node features using mean and standard deviation """
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

        # Each UncertaintySampler applied to the batch appends one extra
        # uncertainty column (e.g. one per velocity coordinate when a
        # different distribution is used for each). Pad x_loc/x_scale with
        # 0/1 for every appended column so the mismatch is handled
        # regardless of how many uncertainty transforms were chained.
        # TODO: This is a temporary fix, should be handled better in the future
        n_missing = batch.x.shape[1] - x_loc.shape[0]
        if n_missing > 0:
            x_loc = torch.cat([x_loc, torch.zeros(n_missing, device=x_loc.device)])
            x_scale = torch.cat([x_scale, torch.ones(n_missing, device=x_scale.device)])

        batch.x = (batch.x - x_loc) / x_scale
        return batch
