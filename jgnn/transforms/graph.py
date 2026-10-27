
import torch
from torch_geometric import transforms as T
from torch_geometric.utils import to_dense_batch
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


# torch_cluster's CUDA kernel asserts k <= 100 (csrc/cuda/knn_cuda.cu),
# which a 0.2 ratio reaches at 500 stars; fall back to the torch
# implementation there instead of crashing mid-run.
_CLUSTER_MAX_K = 100


def knn_graph(pos, k, loop=False):
    if (_knn_graph_cluster is None or pos.device.type == 'mps'
            or (pos.device.type == 'cuda' and k > _CLUSTER_MAX_K)):
        return _knn_graph_torch(pos, k, loop)
    return _knn_graph_cluster(pos, k=k, loop=loop)


def _knn_batched(pos, batch_vec, ptr, ratio, loop=True, chunk=64):
    """Adaptive k-NN edges for a whole batch without a per-graph loop.

    Pads each chunk of graphs to (g, Nmax, 2), takes one `cdist` and one
    `topk` with the largest k in the chunk, then keeps each graph's own
    k = ratio * n_i by a rank mask. Same convention as
    `torch_cluster.knn_graph` (row 0 = neighbour, row 1 = query node) in
    global node indices, and the same edge set as the per-graph loop.
    `chunk` bounds the (g, Nmax, Nmax) distance matrix: 64 graphs of
    1000 stars is 0.25 GB in float32.
    """
    counts = ptr[1:] - ptr[:-1]
    ks = torch.clamp((ratio * counts).long(), min=1)
    out = []
    for lo in range(0, counts.numel(), chunk):
        hi = min(lo + chunk, counts.numel())
        sel = (batch_vec >= lo) & (batch_vec < hi)
        dense, mask = to_dense_batch(pos[sel], batch_vec[sel] - lo)
        d = torch.cdist(dense, dense)
        big = torch.finfo(d.dtype).max
        d = d.masked_fill(~mask[:, None, :], big)
        if not loop:
            eye = torch.eye(d.shape[1], dtype=torch.bool, device=d.device)
            d = d.masked_fill(eye[None], big)
        kmax = min(int(ks[lo:hi].max()), d.shape[1])
        nbr = d.topk(kmax, dim=2, largest=False).indices
        rank = torch.arange(kmax, device=d.device)
        keep = (rank[None, None, :] < ks[lo:hi, None, None]) & mask[:, :, None]
        g_idx = torch.arange(hi - lo, device=d.device)[:, None, None].expand_as(nbr)
        q_idx = torch.arange(d.shape[1], device=d.device)[None, :, None].expand_as(nbr)
        offset = ptr[lo:hi][g_idx[keep]]
        out.append(torch.stack([offset + nbr[keep], offset + q_idx[keep]]))
    return torch.cat(out, dim=1)


class AdaptiveKNNGraph:
    """
    k-NN graph where k is based on the number of nodes in each graph.
    Supports both single graphs and batched graphs with per-graph adaptive k.
    """
    def __init__(self, ratio: float = 0.1, loop: bool = False,
                 device: str = None, batched: bool = True, chunk: int = 64):
        """
        Args:
            ratio (float): Ratio of k to the number of nodes in each graph.
            loop (bool): Whether to include self-loops.
            device (str): Build the edges on this device ('auto' for
                the fastest one present: cuda, else mps, else the CPU)
                instead of the one `pos` sits on, and
                leave `edge_index` there. The rest of the augmentation
                pipeline stays on the CPU, so only `pos` (a few hundred
                KB) is copied over, while the edge list (tens of MB at
                ratio 0.2) is produced where the model needs it.
                Ignored when the device is unavailable, so a config
                carrying 'cuda' still loads on a CPU-only host.
            batched (bool): Build a batch's edges with `_knn_batched`
                (one cdist/topk per chunk of graphs) instead of looping
                over the graphs in Python. Same edge set either way.
            chunk (int): Graphs per chunk in the batched path.
        """
        self.ratio = ratio
        self.loop = loop
        self.device = device
        self.batched = batched
        self.chunk = chunk

    def _target_device(self, pos):
        """Device to build on: `self.device` if usable, else pos's."""
        if self.device is None:
            return pos.device
        if self.device == 'auto':
            # Reason: one config runs on the GPU cluster and on the Mac.
            if torch.cuda.is_available():
                return torch.device('cuda')
            if torch.backends.mps.is_available():
                return torch.device('mps')
            return pos.device
        dev = torch.device(self.device)
        if dev.type == 'cuda' and not torch.cuda.is_available():
            return pos.device
        if dev.type == 'mps' and not torch.backends.mps.is_available():
            return pos.device
        return dev

    def __call__(self, data):
        data = data.clone()

        # Check if this is a batch (has ptr attribute with multiple graphs)
        is_batch = hasattr(data, 'ptr') and len(data.ptr) > 2

        dev = self._target_device(data.pos)
        pos_all = data.pos if dev == data.pos.device else data.pos.to(dev)

        if not is_batch:
            # Single graph: use simple approach
            num_nodes = data.num_nodes
            k = max(1, int(self.ratio * num_nodes))
            data.edge_index = knn_graph(pos_all, k=k, loop=self.loop)
            return data

        if self.batched:
            ptr = data.ptr if dev == data.ptr.device else data.ptr.to(dev)
            bvec = data.batch if dev == data.batch.device else data.batch.to(dev)
            data.edge_index = _knn_batched(
                pos_all, bvec, ptr, self.ratio, self.loop, self.chunk)
            return data

        # Batch: process each graph with its own k value
        edge_indices = []
        # Reason: read the graph bounds once on the host. Indexing ptr
        # element by element costs a device sync per graph when the batch
        # already lives on the GPU.
        bounds = data.ptr.tolist()
        n_graphs = len(bounds) - 1

        for i in range(n_graphs):
            start, end = bounds[i], bounds[i + 1]
            graph_pos = pos_all[start:end]
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
