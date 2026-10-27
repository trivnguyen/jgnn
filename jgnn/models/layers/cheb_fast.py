"""Chebyshev spectral convolution from plain tensor operations.

`ChebConvFast` computes exactly what `torch_geometric.nn.ChebConv` does
with `normalization='sym'` and `lambda_max=2` (the production setting):

    out = sum_k W_k T_k(L_hat) x + b,   L_hat = L - I,
    T_0 = x, T_1 = L_hat x, T_k = 2 L_hat T_{k-1} - T_{k-2},

with L the symmetric normalised Laplacian from `get_laplacian`. The one
graph operation, "multiply by L_hat", is an `index_select` of the source
rows, a scale by the Laplacian weights and an `index_add_` onto the
targets. PyG's version routes the same product through
`MessagePassing.propagate` and rewrites its weights through boolean
masks, which on Apple's MPS backend forces a host sync (`nonzero`) per
call; on an H100 the two are equivalent. Parameter names (`lins.k.weight`,
`bias`) match PyG's so checkpoints load into either.

Reason for a separate layer rather than a patch of PyG: the registry key
`graph_layer = 'ChebConvFast'` makes the choice explicit in every config
and leaves existing configs and checkpoints untouched.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import get_laplacian


class ChebConvFast(nn.Module):
    """Drop-in for `ChebConv(in, out, K)` at sym normalisation, lambda_max 2.

    Args:
        in_channels: Input feature size.
        out_channels: Output feature size.
        K: Chebyshev filter size (number of terms).
        bias: Whether to add a learnable bias.
        normalization: Only 'sym' is supported (what the production
            models use); given for signature compatibility.
        lambda_max: Only 2.0 is supported, likewise.
    """

    def __init__(self, in_channels: int, out_channels: int, K: int,
                 bias: bool = True, normalization: str = 'sym',
                 lambda_max: float = 2.0):
        super().__init__()
        if normalization != 'sym' or lambda_max != 2.0:
            raise ValueError(
                'ChebConvFast implements normalization="sym", lambda_max=2 '
                f'only; got {normalization!r}, {lambda_max}')
        self.K = K
        self.lins = nn.ModuleList(
            [nn.Linear(in_channels, out_channels, bias=False)
             for _ in range(K)])
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        for lin in self.lins:
            # Reason: same initialisation as PyG's ChebConv (glorot).
            nn.init.xavier_uniform_(lin.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None) -> torch.Tensor:
        n = x.shape[0]
        lap_index, lap_weight = get_laplacian(
            edge_index, edge_weight, normalization='sym', num_nodes=n)
        src, dst = lap_index[0], lap_index[1]
        w = lap_weight.to(x.dtype).view(-1, 1)

        def lhat(v):
            # (L - I) v
            return torch.zeros_like(v).index_add_(
                0, dst, v.index_select(0, src) * w) - v

        tx0 = x
        out = self.lins[0](tx0)
        if self.K > 1:
            tx1 = lhat(tx0)
            out = out + self.lins[1](tx1)
            for k in range(2, self.K):
                tx2 = 2 * lhat(tx1) - tx0
                out = out + self.lins[k](tx2)
                tx0, tx1 = tx1, tx2
        return out if self.bias is None else out + self.bias

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.lins[0].in_features}, '
                f'{self.lins[0].out_features}, K={self.K})')
