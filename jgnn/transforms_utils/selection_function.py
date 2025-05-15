
import torch
from torch_geometric import transforms as T

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
        if q_min > q_max:
            raise ValueError(f"q_min should be smaller than or equal to q_max, but got {q_min} > {q_max}")

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