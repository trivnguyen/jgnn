
import torch

class ExponentialDecaySelectionFunction:
    """ Selection function with exponential decay probability based on radial distance """
    def __init__(self, alpha_range=(0.1, 2.0), norm_range=(0.5, 1.0)):
        """
        Args:
            alpha_range: Tuple (min, max) for random sampling of alpha
                        (decay parameter in norm * exp(-alpha * r_normalized))
            norm_range: Tuple (min, max) for random sampling of normalization
                       constant
        """
        self.alpha_range = alpha_range
        self.norm_range = norm_range

        # Validate alpha range
        if not (alpha_range[0] > 0 and alpha_range[0] <= alpha_range[1]):
            raise ValueError(
                f"alpha_range should be (min, max) with 0 < min <= max, "
                f"but got {alpha_range}"
            )

        # Validate normalization range
        if not (0 < norm_range[0] <= norm_range[1] <= 1):
            raise ValueError(
                f"norm_range should be (min, max) with 0 < min <= max <= 1, "
                f"but got {norm_range}"
            )

    def __call__(self, batch):
        batch = batch.clone()
        n_per_batch = batch.ptr[1:] - batch.ptr[:-1]
        n_graph = batch.num_graphs

        # Calculate radii for all nodes
        radii = torch.norm(batch.pos, dim=1)

        # Calculate r_min and r_max for each graph
        r_min_vals = []
        r_max_vals = []
        selection_probs = []

        # Sample random alpha and normalization values for each graph
        alpha_vals = (torch.rand(n_graph) *
                     (self.alpha_range[1] - self.alpha_range[0]) +
                     self.alpha_range[0])
        norm_vals = (torch.rand(n_graph) *
                    (self.norm_range[1] - self.norm_range[0]) +
                    self.norm_range[0])

        for i in range(n_graph):
            # Get radii for current graph
            graph_radii = radii[batch.ptr[i]:batch.ptr[i + 1]]

            # Calculate r_min and r_max for this graph
            r_min = torch.min(graph_radii)
            r_max = torch.max(graph_radii)

            r_min_vals.append(r_min)
            r_max_vals.append(r_max)

            # Use randomly sampled alpha and normalization for this graph
            alpha = alpha_vals[i]
            norm = norm_vals[i]

            # Calculate exponential decay probabilities for this graph
            if r_max > r_min:
                # Normalize radii to [0, 1] range: r_norm = (r - r_min) / (r_max - r_min)
                # Then apply: p(r) = norm * exp(-alpha * r_norm)
                normalized_radii = (graph_radii - r_min) / (r_max - r_min)
                graph_probs = norm * torch.exp(-alpha * normalized_radii)
            else:
                # All nodes at same radius, probability = norm * exp(-alpha * 0) = norm
                graph_probs = torch.full_like(graph_radii, norm)

            selection_probs.append(graph_probs)

        # Concatenate all probabilities
        all_probs = torch.cat(selection_probs, dim=0)

        # Generate random values and create mask
        random_vals = torch.rand(batch.num_nodes, device=batch.pos.device)
        mask = random_vals < all_probs

        # Apply mask to all relevant attributes
        batch.x = batch.x[mask]
        batch.pos = batch.pos[mask]
        batch.vel = batch.vel[mask]
        batch.batch = batch.batch[mask]

        # Recalculate ptr
        batch.ptr = torch.searchsorted(
            batch.batch,
            torch.arange(n_graph+1, device=batch.batch.device)
        )

        return batch


class LinearDecaySelectionFunction:
    """ Selection function with linear decay probability based on radial distance """
    def __init__(self, p_min_range=(0.0, 0.3), p_max_range=(0.7, 1.0)):
        """
        Args:
            p_min_range: Tuple (min, max) for random sampling of p_min
                        (probability at r_max)
            p_max_range: Tuple (min, max) for random sampling of p_max
                        (probability at r_min)
        """
        self.p_min_range = p_min_range
        self.p_max_range = p_max_range

        # Validate probability ranges
        if not (0 <= p_min_range[0] <= p_min_range[1] <= 1):
            raise ValueError(
                f"p_min_range should be (min, max) with 0 <= min <= max <= 1, "
                f"but got {p_min_range}"
            )
        if not (0 <= p_max_range[0] <= p_max_range[1] <= 1):
            raise ValueError(
                f"p_max_range should be (min, max) with 0 <= min <= max <= 1, "
                f"but got {p_max_range}"
            )
        if p_min_range[1] > p_max_range[0]:
            raise ValueError(
                f"p_min_range max ({p_min_range[1]}) should be <= "
                f"p_max_range min ({p_max_range[0]}) to ensure p_min <= p_max"
            )

    def __call__(self, batch):
        batch = batch.clone()
        n_per_batch = batch.ptr[1:] - batch.ptr[:-1]
        n_graph = batch.num_graphs

        # Calculate radii for all nodes
        radii = torch.norm(batch.pos, dim=1)

        # Calculate r_min and r_max for each graph
        r_min_vals = []
        r_max_vals = []
        selection_probs = []

        # Sample random p_min and p_max values for each graph
        p_min_vals = (torch.rand(n_graph) *
                     (self.p_min_range[1] - self.p_min_range[0]) +
                     self.p_min_range[0])
        p_max_vals = (torch.rand(n_graph) *
                     (self.p_max_range[1] - self.p_max_range[0]) +
                     self.p_max_range[0])

        for i in range(n_graph):
            # Get radii for current graph
            graph_radii = radii[batch.ptr[i]:batch.ptr[i + 1]]

            # Calculate r_min and r_max for this graph
            r_min = torch.min(graph_radii)
            r_max = torch.max(graph_radii)

            r_min_vals.append(r_min)
            r_max_vals.append(r_max)

            # Use randomly sampled probabilities for this graph
            p_min = p_min_vals[i]
            p_max = p_max_vals[i]

            # Calculate linear decay probabilities for this graph
            if r_max > r_min:
                # Linear interpolation:
                # p = p_max + (p_min - p_max) * (r - r_min) / (r_max - r_min)
                normalized_radii = (graph_radii - r_min) / (r_max - r_min)
                graph_probs = p_max + (p_min - p_max) * normalized_radii
            else:
                # All nodes at same radius, use maximum probability
                graph_probs = torch.full_like(graph_radii, p_max)

            selection_probs.append(graph_probs)

        # Concatenate all probabilities
        all_probs = torch.cat(selection_probs, dim=0)

        # Generate random values and create mask
        random_vals = torch.rand(batch.num_nodes, device=batch.pos.device)
        mask = random_vals < all_probs

        # Apply mask to all relevant attributes
        batch.x = batch.x[mask]
        batch.pos = batch.pos[mask]
        batch.vel = batch.vel[mask]
        batch.batch = batch.batch[mask]

        # Recalculate ptr
        batch.ptr = torch.searchsorted(
            batch.batch,
            torch.arange(n_graph+1, device=batch.batch.device)
        )

        return batch


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
                # For low mode, select negative differences, i.e. take nodes with radius <= quantile
                mask = diff <= 0
            elif self.mode == 'high':
                # For high mode, select positive differences, i.e. take nodes with radius >= quantile
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
