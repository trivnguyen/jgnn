import torch

class RadialSelectionFunction:
    """ Selection function with various modes """
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
            raise ValueError(f"q_min should be smaller than q_max, but got {q_min} > {q_max}")

    def __call__(self, batch):
        batch = batch.clone()
        n_per_batch = batch.ptr[1:] - batch.ptr[:-1]
        n_graph = batch.num_graphs

        # Generate random q values between q_min and q_max for each graph
        q_vals = torch.rand(n_graph) * (self.q_max - self.q_min) + self.q_min

        if self.mode == 'dropout':
            # Use q_vals as dropout probabilities for each graph
            keep_probs = 1 - q_vals
            node_rand = torch.rand(batch.num_nodes, device=batch.pos.device)
            graph_keep_probs = torch.repeat_interleave(keep_probs, n_per_batch)
            mask = node_rand < graph_keep_probs
        elif self.mode == 'identity':
            # Keep all nodes (identity transform)
            return batch
        else:
            # Proceed with quantile-based selection
            radii = torch.norm(batch.pos, dim=1)
            radii_q = []
            for i in range(n_graph):
                rad = radii[batch.ptr[i]:batch.ptr[i + 1]]
                rad_q = torch.quantile(rad, q=q_vals[i])
                radii_q.append(rad_q)
            radii_q = torch.stack(radii_q, dim=0)

            # Calculate differences between radii and repeated quantile values
            diff = radii - torch.repeat_interleave(radii_q, n_per_batch)

            if self.mode == 'low':
                # For low mode, select negative differences
                mask = diff <= 0
            elif self.mode == 'high':
                # For high mode, select positive differences
                mask = diff >= 0
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

        # Apply mask to all relevant attributes
        batch.x = batch.x[mask]
        batch.pos = batch.pos[mask]
        batch.vel = batch.vel[mask]
        batch.batch = batch.batch[mask]

        # Recalculate ptr
        batch.ptr = torch.searchsorted(batch.batch, torch.arange(n_graph+1, device=batch.batch.device))

        return batch


class RandomSelectionStrategy:
    """
    Randomly applies different node selection strategies with specified probabilities.
    """
    def __init__(self, modes=['low', 'high', 'dropout', 'identity'], probs=None, q_min=0.1, q_max=0.5):
        """
        Args:
            modes: List of selection modes ('low', 'high', 'dropout', 'identity')
                  - 'low': Select nodes with radius <= quantile
                  - 'high': Select nodes with radius >= quantile
                  - 'dropout': Randomly drop nodes with probability = quantile
                  - 'identity': Keep all nodes (no filtering)
            probs: List of probabilities for each mode (must sum to 1.0)
            q_min: Minimum quantile value
            q_max: Maximum quantile value
        """
        self.modes = modes
        if probs is None:
            # Equal probability for each mode
            self.probs = torch.ones(len(modes)) / len(modes)
        else:
            assert len(probs) == len(modes), "Number of probabilities must match number of modes"
            assert abs(sum(probs) - 1.0) < 1e-6, "Probabilities must sum to 1.0"
            self.probs = torch.tensor(probs)

        self.q_min = q_min
        self.q_max = q_max

        # Create selection functions for each mode
        self.selection_functions = {}
        for mode in modes:
            self.selection_functions[mode] = RadialSelectionFunction(q_min, q_max, mode)

    def __call__(self, batch):
        # Randomly select a mode based on probabilities
        mode_idx = torch.multinomial(self.probs, 1).item()
        selected_mode = self.modes[mode_idx]

        # Apply the selected mode's selection function
        return self.selection_functions[selected_mode](batch)
