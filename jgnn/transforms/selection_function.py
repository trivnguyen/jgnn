
import torch
from abc import ABC, abstractmethod
from torch_geometric.data import Data, Batch

_NODE_ATTRS = ('x', 'pos', 'vel', 'vel_error')

class BaseSelectionFunction(ABC):
    """Base class for selection functions that filter nodes based on radial distance."""

    def __call__(self, batch):
        batch = batch.clone()
        n_graph = batch.num_graphs
        radii = torch.norm(batch.pos, dim=1)

        # Compute per-graph selection probabilities
        selection_probs = []
        for i in range(n_graph):
            graph_radii = radii[batch.ptr[i]:batch.ptr[i + 1]]
            graph_probs = self._compute_selection_probs(graph_radii, i, n_graph)
            selection_probs.append(graph_probs)

        all_probs = torch.cat(selection_probs, dim=0)

        # Generate random values and create mask
        random_vals = torch.rand(batch.num_nodes, device=batch.pos.device)
        mask = random_vals < all_probs

        return apply_mask(batch, mask)

    @abstractmethod
    def _compute_selection_probs(self, graph_radii, graph_idx, n_graph):
        """Compute selection probabilities for nodes in a single graph.

        Args:
            graph_radii: Tensor of radii for nodes in this graph
            graph_idx: Index of this graph in the batch
            n_graph: Total number of graphs in the batch

        Returns:
            Tensor of selection probabilities for each node
        """
        pass


def apply_mask(batch, mask, min_nodes=1):
    """Apply a boolean mask to the node-level attributes of the batch.

    Only masks whichever of `x`, `pos`, `vel`, `vel_error` are actually
    present on the batch — e.g. `x` doesn't exist yet if selection runs
    before `GetNodeFeatures` in the pre_transform pipeline. Graph-level
    attributes (`theta`, `cond`) are carried over unchanged.

    Args:
        batch: PyG Batch object
        mask: Boolean mask for nodes to keep
        min_nodes: Minimum number of nodes to keep per graph (default: 1).
                   If fewer nodes would remain, randomly keeps min_nodes nodes.
    """
    data_list = []
    for i in range(batch.num_graphs):
        node_start, node_end = batch.ptr[i], batch.ptr[i + 1]
        graph_mask = mask[node_start:node_end]

        # Safeguard: ensure at least min_nodes are kept
        num_kept = graph_mask.sum().item()
        num_nodes = graph_mask.shape[0]
        if num_kept < min_nodes and num_nodes >= min_nodes:
            # Randomly select min_nodes indices to keep
            keep_indices = torch.randperm(num_nodes)[:min_nodes]
            graph_mask = torch.zeros(num_nodes, dtype=torch.bool)
            graph_mask[keep_indices] = True

        graph_data = batch[i]
        kwargs = {
            attr: getattr(graph_data, attr)[graph_mask]
            for attr in _NODE_ATTRS
            if graph_data.get(attr) is not None
        }
        kwargs['theta'] = graph_data.theta
        kwargs['cond'] = graph_data.get('cond', None)

        data_list.append(Data(**kwargs))
    batch = Batch.from_data_list(data_list)
    return batch

class ExponentialSelectionFunction(BaseSelectionFunction):
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
        self._alpha_vals = None
        self._norm_vals = None

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
        # Sample random values for this batch before calling parent
        n_graph = batch.num_graphs
        self._alpha_vals = (torch.rand(n_graph) *
                          (self.alpha_range[1] - self.alpha_range[0]) +
                          self.alpha_range[0])
        self._norm_vals = (torch.rand(n_graph) *
                         (self.norm_range[1] - self.norm_range[0]) +
                         self.norm_range[0])
        return super().__call__(batch)

    def _compute_selection_probs(self, graph_radii, graph_idx, n_graph):
        alpha = self._alpha_vals[graph_idx]
        norm = self._norm_vals[graph_idx]

        r_min = torch.min(graph_radii)
        r_max = torch.max(graph_radii)

        if r_max > r_min:
            normalized_radii = (graph_radii - r_min) / (r_max - r_min)
            return norm * torch.exp(-alpha * normalized_radii)
        else:
            return torch.full_like(graph_radii, norm)

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


class LinearSelectionFunction(BaseSelectionFunction):
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
        self._p_min_vals = None
        self._p_max_vals = None

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
        # Sample random values for this batch before calling parent
        n_graph = batch.num_graphs
        self._p_min_vals = (torch.rand(n_graph) *
                          (self.p_min_range[1] - self.p_min_range[0]) +
                          self.p_min_range[0])
        self._p_max_vals = (torch.rand(n_graph) *
                          (self.p_max_range[1] - self.p_max_range[0]) +
                          self.p_max_range[0])
        return super().__call__(batch)

    def _compute_selection_probs(self, graph_radii, graph_idx, n_graph):
        p_min = self._p_min_vals[graph_idx]
        p_max = self._p_max_vals[graph_idx]

        r_min = torch.min(graph_radii)
        r_max = torch.max(graph_radii)

        if r_max > r_min:
            normalized_radii = (graph_radii - r_min) / (r_max - r_min)
            return p_max + (p_min - p_max) * normalized_radii
        else:
            return torch.full_like(graph_radii, p_max)

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
    """Selection function that drops nodes based on radial distance or randomly.

    The dropout_rate parameter is consistent across all modes: it always represents
    the fraction of nodes to drop (like standard dropout).

    Modes:
        - 'drop_outer': Drop nodes with large radii (outer nodes). Keeps inner nodes.
        - 'drop_inner': Drop nodes with small radii (inner nodes). Keeps outer nodes.
        - 'random': Drop nodes randomly (standard dropout behavior).
        - 'identity': Keep all nodes (no dropout).

    Examples:
        - dropout_rate=0.2, mode='drop_outer' → drops 20% of nodes (those with largest radii)
        - dropout_rate=0.2, mode='drop_inner' → drops 20% of nodes (those with smallest radii)
        - dropout_rate=0.2, mode='random' → drops 20% of nodes randomly
    """

    def __init__(self, dropout_min, dropout_max, mode):
        """
        Args:
            dropout_min: Minimum dropout rate (fraction of nodes to drop)
            dropout_max: Maximum dropout rate (fraction of nodes to drop)
            mode: One of 'drop_outer', 'drop_inner', 'random', or 'identity'
        """
        self.dropout_min = dropout_min
        self.dropout_max = dropout_max
        self.mode = mode

        if not (0 <= dropout_min <= 1):
            raise ValueError(f"dropout_min should be in [0, 1], but got {dropout_min}")
        if not (0 <= dropout_max <= 1):
            raise ValueError(f"dropout_max should be in [0, 1], but got {dropout_max}")
        if dropout_min > dropout_max:
            raise ValueError(
                f"dropout_min should be <= dropout_max, but got {dropout_min} > {dropout_max}"
            )
        if mode not in ('drop_outer', 'drop_inner', 'random', 'identity'):
            raise ValueError(
                f"mode should be one of 'drop_outer', 'drop_inner', 'random', 'identity', "
                f"but got '{mode}'"
            )

    def __call__(self, batch):
        batch = batch.clone()
        n_per_batch = batch.ptr[1:] - batch.ptr[:-1]
        n_graph = batch.num_graphs

        # Sample dropout rates for each graph
        dropout_rates = (
            torch.rand(n_graph) * (self.dropout_max - self.dropout_min) + self.dropout_min
        )

        if self.mode == 'identity':
            return batch

        if self.mode == 'random':
            # Standard dropout: randomly drop nodes
            keep_probs = 1 - dropout_rates
            node_rand = torch.rand(batch.num_nodes, device=batch.pos.device)
            graph_keep_probs = torch.repeat_interleave(keep_probs, n_per_batch)
            mask = node_rand < graph_keep_probs
        else:
            # Radial-based dropout
            radii = torch.norm(batch.pos, dim=1)
            radii_quantiles = []

            for i in range(n_graph):
                rad = radii[batch.ptr[i]:batch.ptr[i + 1]]
                dropout_rate = dropout_rates[i]

                if self.mode == 'drop_outer':
                    # Drop outer nodes → keep (1 - dropout_rate) fraction with smallest radii
                    # Quantile at (1 - dropout_rate) gives the cutoff
                    quantile = torch.quantile(rad, q=1 - dropout_rate)
                elif self.mode == 'drop_inner':
                    # Drop inner nodes → keep (1 - dropout_rate) fraction with largest radii
                    # Quantile at dropout_rate gives the cutoff
                    quantile = torch.quantile(rad, q=dropout_rate)

                radii_quantiles.append(quantile)

            radii_quantiles = torch.stack(radii_quantiles, dim=0)
            thresholds = torch.repeat_interleave(radii_quantiles, n_per_batch)

            if self.mode == 'drop_outer':
                # Keep nodes with radius <= threshold
                mask = radii <= thresholds
            else:  # drop_inner
                # Keep nodes with radius >= threshold
                mask = radii >= thresholds

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
                {'type': 'radial', 'params': {'dropout_min': 0.1, 'dropout_max': 0.3,
                                              'mode': 'drop_outer'}},
                {'type': 'radial', 'params': {'dropout_min': 0.1, 'dropout_max': 0.3,
                                              'mode': 'drop_inner'}},
                {'type': 'radial', 'params': {'dropout_min': 0.1, 'dropout_max': 0.3,
                                              'mode': 'random'}},
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
