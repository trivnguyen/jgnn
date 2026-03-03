
import torch
from torch_geometric import transforms as T
from torch_cluster import knn_graph

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
            transform = T.KNNGraph(k=k, loop=self.loop)
            return transform(data)

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
