
import torch
from torch_geometric import transforms as T
try:
    from torch_cluster import knn_graph as _knn_graph_cluster
except ImportError:  # Reason: torch_cluster has no wheels for macOS arm64
    _knn_graph_cluster = None


def _knn_graph_torch(pos, k, loop=False):
    """Pure-torch k-NN edges, same layout as torch_cluster.knn_graph
    (flow source_to_target: row 0 = neighbour, row 1 = query node).
    Used when torch_cluster is absent or `pos` lives on a device it has
    no kernel for (MPS). O(N^2) memory per graph: fine for <= few 1e3
    stars."""
    n = pos.shape[0]
    k = min(k, n if loop else n - 1)
    if k <= 0:
        return torch.empty(2, 0, dtype=torch.long, device=pos.device)
    d = torch.cdist(pos, pos)
    if not loop:
        d.fill_diagonal_(float('inf'))
    nbr = d.topk(k, dim=1, largest=False).indices  # (n, k)
    row = torch.arange(n, device=pos.device).repeat_interleave(k)
    return torch.stack([nbr.reshape(-1), row])


def knn_graph(pos, k, loop=False):
    if _knn_graph_cluster is None or pos.device.type == 'mps':
        return _knn_graph_torch(pos, k, loop)
    return _knn_graph_cluster(pos, k=k, loop=loop)


class AdaptiveKNNGraph:
    """
    k-NN graph where k is based on the number of nodes in each graph.
    Supports both single graphs and batched graphs with per-graph adaptive k.
    """
    def __init__(self, ratio: float = 0.1, loop: bool = False):
        """
        Args:
            ratio (float): Ratio of k to the number of nodes in each graph.
            loop (bool): Whether to include self-loops.
        """
        self.ratio = ratio
        self.loop = loop

    def __call__(self, data):
        data = data.clone()

        # Check if this is a batch (has ptr attribute with multiple graphs)
        is_batch = hasattr(data, 'ptr') and len(data.ptr) > 2

        if not is_batch:
            # Single graph: use simple approach
            num_nodes = data.num_nodes
            k = max(1, int(self.ratio * num_nodes))
            data.edge_index = knn_graph(data.pos, k=k, loop=self.loop)
            return data

        # Batch: process each graph with its own k value
        edge_indices = []
        n_graphs = len(data.ptr) - 1

        for i in range(n_graphs):
            start, end = data.ptr[i].item(), data.ptr[i + 1].item()
            graph_pos = data.pos[start:end]
            num_nodes = end - start
            k = max(1, int(self.ratio * num_nodes))

            # Build k-NN edges for this graph
            edge_index = knn_graph(graph_pos, k=k, loop=self.loop)

            # Offset edge indices to global node indices
            edge_index = edge_index + start
            edge_indices.append(edge_index)

        # Concatenate all edge indices
        data.edge_index = torch.cat(edge_indices, dim=1)
        return data


ALL_GRAPHS = {
    "knn": T.KNNGraph,
    "radius": T.RadiusGraph,
    "adaptive_knn": AdaptiveKNNGraph,
}
