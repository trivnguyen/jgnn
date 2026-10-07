"""Set Transformer encoder (Lee et al. 2019) as the NPE embedding.

Self-attention blocks over the stars, then pooling by multihead
attention: `n_seeds` learned seed vectors attend over the stars and
their outputs are concatenated, so the context passed to the flow is
`n_seeds x d_model` numbers chosen by attention rather than one mean
over the nodes. No graph is built (`pre_transforms.apply_graph =
False`). A non-default embedding type: existing GNN models are
untouched.
"""

from typing import Any, Dict

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from ..layers import MLP
from ..layers.transformer import MultiHeadAttentionBlock
from ..utils import get_activation, build_embedding_loss
from .gnn_embedding import GNNEmbedding


class SetTransformerEmbedding(GNNEmbedding):
    """Linear embed -> n_layers SAB -> PMA(n_seeds) -> MLP [+ CondMLP].

    Parameters
    ----------
    set_transformer_args : dict
        `d_model`, `n_layers`, `n_heads`, `n_seeds` and optional `d_mlp`
        (feed-forward width inside each attention block, default
        2 * d_model).
    **kwargs
        Passed to `GNNEmbedding`; `gnn_args` is accepted and ignored so
        configs and loaders keep one shape.
    """

    def __init__(self, set_transformer_args: Dict[str, Any], gnn_args=None,
                 **kwargs):
        self.set_args = dict(set_transformer_args)
        super().__init__(gnn_args=gnn_args or {}, **kwargs)

    def _setup_model(self):
        a = self.set_args
        d, d_mlp = a['d_model'], a.get('d_mlp', 2 * a['d_model'])
        self.embed = nn.Linear(self.input_size, d)
        self.sabs = nn.ModuleList([
            MultiHeadAttentionBlock(d, d, d_mlp, a['n_heads'])
            for _ in range(a['n_layers'])])
        self.seeds = nn.Parameter(torch.randn(a['n_seeds'], d) / d ** 0.5)
        self.pma = MultiHeadAttentionBlock(d, d, d_mlp, a['n_heads'])
        self.final_ln = nn.LayerNorm(d)
        self.gnn = None
        mlp_config = dict(self.mlp_args)
        mlp_config['input_size'] = a['n_seeds'] * d
        mlp_config['act'] = get_activation(
            mlp_config.pop('act_name'), mlp_config.pop('act_args', {}))
        self.mlp = MLP(**mlp_config)
        if self.conditional_mlp_args is not None:
            cond_config = dict(self.conditional_mlp_args)
            cond_config['act'] = get_activation(
                cond_config.pop('act_name'), cond_config.pop('act_args', {}))
            self.conditional_mlp = MLP(**cond_config)
        else:
            self.conditional_mlp = None
        self.output_size = self.mlp_args['output_size']
        loss_config = dict(self.loss_args)
        if self.loss_type == 'flow' and 'context_features' not in loss_config:
            loss_config['context_features'] = self.mlp_args['output_size']
        self.loss_fn, self.flow = build_embedding_loss(
            self.loss_type, loss_config)

    def encode(self, x_dense: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Stars `x_dense` [B, N, F] with `valid` [B, N] -> [B, n_seeds*d]."""
        pad = ~valid  # Reason: nn.MultiheadAttention masks keys where True.
        h = self.embed(x_dense)
        for sab in self.sabs:
            h = sab(h, h, pad)
        seeds = self.seeds.unsqueeze(0).expand(h.size(0), -1, -1)
        pooled = self.pma(seeds, h, pad)
        return self.final_ln(pooled).flatten(1)

    def embed_data(self, batch_dict):
        """SAB stack -> PMA -> MLP, no conditional term."""
        return self.mlp(
            self.encode(batch_dict['x_dense'], batch_dict['valid']))

    def forward(self, batch_dict):
        embedding = self.embed_data(batch_dict)
        if self.conditional_mlp is not None:
            embedding = embedding + self.conditional_mlp(batch_dict['cond'])
        return embedding

    def _prepare_batch(self, batch):
        batch = self.pre_transforms(batch) if self.pre_transforms else batch
        batch = batch.to(self.device)
        graph = batch.get('batch')
        if graph is None:
            graph = torch.zeros(batch.x.shape[0], dtype=torch.long,
                                device=batch.x.device)
        x_dense, valid = to_dense_batch(batch.x, graph)
        return {
            'x': batch.x, 'x_dense': x_dense, 'valid': valid,
            'edge_index': None, 'batch': graph, 'target': batch.get('theta'),
            'edge_attr': None, 'edge_weight': None, 'cond': batch.get('cond'),
            'batch_size': x_dense.size(0),
        }
