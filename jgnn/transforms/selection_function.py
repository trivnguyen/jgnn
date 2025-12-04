
import torch
from torch_geometric.data import Data, Batch

def apply_mask(batch, mask):
    """Apply a boolean mask to all relevant attributes of the batch."""
    data_list = []
    for i in range(batch.num_graphs):
        node_start, node_end = batch.ptr[i], batch.ptr[i + 1]
        graph_mask = mask[node_start:node_end]
        graph_data = batch[i]
        graph_data = Data(
            x=graph_data.x[graph_mask],
            pos=graph_data.pos[graph_mask],
            vel=graph_data.vel[graph_mask],
            theta=graph_data.theta,
            cond=graph_data.cond,
        )
        data_list.append(graph_data)
    batch = Batch.from_data_list(data_list)
    return batch

class ExponentialSelectionFunction:
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

    def get_functional_form(self, N=100, alpha=None, norm=None):
        """
        Return the functional form for normalized radius.

        Args:
            N: Number of points to sample

        Returns:
            x: Normalized radius values from 0 to 1
            y: Selection probability values corresponding to x
        """
        # Sample random alpha and normalization for demonstration
        if alpha is None:
            # Sample alpha from the defined range
            alpha = (torch.rand(1) *
                    (self.alpha_range[1] - self.alpha_range[0]) +
                    self.alpha_range[0]).item()
        if norm is None:
            # Sample norm from the defined range
            norm = (torch.rand(1) *
                (self.norm_range[1] - self.norm_range[0]) +
                self.norm_range[0]).item()

        # Create normalized radius values from 0 to 1
        x = torch.linspace(0, 1, N)

        # Apply exponential decay: p = norm * exp(-alpha * r_norm)
        y = norm * torch.exp(-alpha * x)

        return x, y


class LinearSelectionFunction:
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

        # # Apply mask to all relevant attributes
        # batch.x = batch.x[mask]
        # batch.pos = batch.pos[mask]
        # batch.vel = batch.vel[mask]
        # batch.batch = batch.batch[mask]
        # # Recalculate ptr
        # batch.ptr = torch.searchsorted(
        #     batch.batch,
        #     torch.arange(n_graph+1, device=batch.batch.device)
        # )

        # Apply mask to all relevant attributes
        return apply_mask(batch, mask)

    def get_functional_form(self, N=100, p_min=None, p_max=None):
        """
        Return the functional form for normalized radius.

        Args:
            N: Number of points to sample

        Returns:
            x: Normalized radius values from 0 to 1
            y: Selection probability values corresponding to x
        """
        # Sample random p_min and p_max for demonstration
        if p_min is None:
            # Sample p_min from the defined range
            p_min = (torch.rand(1) *
                    (self.p_min_range[1] - self.p_min_range[0]) +
                    self.p_min_range[0]).item()
        if p_max is None:
            # Sample p_max from the defined range
            p_max = (torch.rand(1) *
                    (self.p_max_range[1] - self.p_max_range[0]) +
                    self.p_max_range[0]).item()

        # Create normalized radius values from 0 to 1
        x = torch.linspace(0, 1, N)

        # Apply linear decay: p = p_max + (p_min - p_max) * r_norm
        y = p_max + (p_min - p_max) * x

        return x, y


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

            if self.mode in ('low', 'accept_low'):
                # For low mode, select negative differences, i.e. take nodes with radius <= quantile
                mask = diff <= 0
            elif self.mode in ('high', 'accept_high'):
                # For high mode, select positive differences, i.e. take nodes with radius >= quantile
                mask = diff >= 0
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

        # # Apply mask to all relevant attributes
        # batch.x = batch.x[mask]
        # batch.pos = batch.pos[mask]
        # batch.vel = batch.vel[mask]
        # batch.batch = batch.batch[mask]

        # # Recalculate ptr and num nodes
        # batch.ptr = torch.searchsorted(batch.batch, torch.arange(n_graph+1, device=batch.batch.device))
        # batch.num_nodes = mask.sum().item()

        # Apply mask to all relevant attributes
        return apply_mask(batch, mask)


class RandomSelectionStrategy:
    """
    Randomly applies different node selection strategies with specified probabilities.
    Supports RadialSelectionFunction, LinearSelectionFunction, and
    ExponentialSelectionFunction.
    """
    def __init__(self, selection_configs, probs=None):
        """
        Args:
            selection_configs: List of dictionaries, each containing:
                - 'type': 'radial', 'linear', or 'exponential'
                - 'params': dictionary of parameters for that selection function
            probs: List of probabilities for each selection config (must sum to 1.0)
                  If None, uses equal probability for each config

        Example:
            selection_configs = [
                {'type': 'radial', 'params': {'q_min': 0.1, 'q_max': 0.5, 'mode': 'low'}},
                {'type': 'radial', 'params': {'q_min': 0.3, 'q_max': 0.7, 'mode': 'high'}},
                {'type': 'linear', 'params': {'p_min_range': (0.0, 0.3),
                                              'p_max_range': (0.7, 1.0)}},
                {'type': 'exponential', 'params': {'alpha_range': (0.1, 2.0),
                                                   'norm_range': (0.5, 1.0)}}
            ]
        """
        self.selection_configs = selection_configs

        if probs is None:
            # Equal probability for each config
            self.probs = torch.ones(len(selection_configs)) / len(selection_configs)
        else:
            assert len(probs) == len(selection_configs), \
                "Number of probabilities must match number of selection configs"
            assert abs(sum(probs) - 1.0) < 1e-6, \
                "Probabilities must sum to 1.0"
            self.probs = torch.tensor(probs)

        # Create selection functions for each config
        self.selection_functions = []
        for config in selection_configs:
            selection_type = config['type']
            params = config['params']

            if selection_type == 'radial':
                func = RadialSelectionFunction(**params)
            elif selection_type == 'linear':
                func = LinearSelectionFunction(**params)
            elif selection_type == 'exponential':
                func = ExponentialSelectionFunction(**params)
            else:
                raise ValueError(f"Unknown selection type: {selection_type}")

            self.selection_functions.append(func)

    def __call__(self, batch):
        # Randomly select a configuration based on probabilities
        config_idx = torch.multinomial(self.probs, 1).item()

        # Apply the selected configuration's selection function
        return self.selection_functions[config_idx](batch)

    def add_selection_config(self, selection_config, prob=None):
        """
        Add a new selection configuration.

        Args:
            selection_config: Dictionary with 'type' and 'params'
            prob: Probability for this config. If None, redistributes
                 probabilities equally among all configs
        """
        self.selection_configs.append(selection_config)

        # Create the new selection function
        selection_type = selection_config['type']
        params = selection_config['params']

        if selection_type == 'radial':
            func = RadialSelectionFunction(**params)
        elif selection_type == 'linear':
            func = LinearSelectionFunction(**params)
        elif selection_type == 'exponential':
            func = ExponentialSelectionFunction(**params)
        else:
            raise ValueError(f"Unknown selection type: {selection_type}")

        self.selection_functions.append(func)

        # Update probabilities
        if prob is None:
            # Redistribute equally
            n_configs = len(self.selection_configs)
            self.probs = torch.ones(n_configs) / n_configs
        else:
            # Normalize existing probabilities and add new one
            current_sum = torch.sum(self.probs)
            remaining_prob = 1.0 - prob
            self.probs = self.probs * (remaining_prob / current_sum)
            self.probs = torch.cat([self.probs, torch.tensor([prob])])

    def get_config_info(self):
        """Return information about current selection configurations."""
        info = []
        for i, (config, prob) in enumerate(zip(self.selection_configs, self.probs)):
            info.append({
                'index': i,
                'type': config['type'],
                'params': config['params'],
                'probability': prob.item()
            })
        return info
