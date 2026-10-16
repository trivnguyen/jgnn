"""DeepSets encoder (Zaheer et al. 2017) as the NPE embedding.

A per-star MLP phi(x_i), summed over the stars of each galaxy and
divided by a reference count n_ref, with log(N / n_ref) appended, then
an MLP over the pooled vector [+ CondMLP]. For independent stars the
log-likelihood is a sum of per-star terms, so this is the function
class of a sufficient statistic; the oracle table of the gap project
is the special case of a fixed phi. No graph is built and stars are
never mixed before pooling. A non-default embedding type: existing
models are untouched.
"""

from typing import Any, Dict

import torch
from torch_geometric.nn import global_add_pool

from ..layers import MLP
from ..utils import get_activation, build_embedding_loss
from .gnn_embedding import GNNEmbedding


class DeepSetsEmbedding(GNNEmbedding):
    """phi per star -> sum / n_ref [, log(N / n_ref)] -> MLP [+ CondMLP].

    Parameters
    ----------
    deepsets_args : dict
        `hidden_sizes` and `output_size` of the per-star MLP phi,
        `act_name` (default 'gelu'), `n_ref` (star count the sum is
        divided by, default 100) and `use_log_n` (append log(N / n_ref)
        to the pooled vector, default True).
    **kwargs
        Passed to `GNNEmbedding`; `gnn_args` is accepted and ignored so
        configs and loaders keep one shape.
    """

    def __init__(self, deepsets_args: Dict[str, Any], gnn_args=None,
                 **kwargs):
        self.ds_args = dict(deepsets_args)
        super().__init__(gnn_args=gnn_args or {}, **kwargs)

    def _setup_model(self):
        a = self.ds_args
        self.n_ref = float(a.get('n_ref', 100))
        self.use_log_n = bool(a.get('use_log_n', True))
        # Reason: no final activation, so phi can represent signed basis
        # functions of (log R, v, sigma_v) like a log-likelihood does.
        self.phi = MLP(
            input_size=self.input_size, output_size=a['output_size'],
            hidden_sizes=list(a['hidden_sizes']),
            act=get_activation(a.get('act_name', 'gelu'),
                               a.get('act_args', {})),
            final_act=False)
        self.gnn = None
        mlp_config = dict(self.mlp_args)
        mlp_config['input_size'] = a['output_size'] + int(self.use_log_n)
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

    def pool(self, x: torch.Tensor, graph: torch.Tensor,
             n_graphs: int) -> torch.Tensor:
        """Sum of phi over each galaxy's stars / n_ref [, log(N / n_ref)]."""
        pooled = global_add_pool(self.phi(x), graph, size=n_graphs)
        pooled = pooled / self.n_ref
        if self.use_log_n:
            n = torch.bincount(graph, minlength=n_graphs).to(pooled.dtype)
            pooled = torch.cat(
                [pooled, torch.log(n / self.n_ref).unsqueeze(1)], 1)
        return pooled

    def embed_data(self, batch_dict):
        """phi -> sum pooling -> MLP, no conditional term."""
        return self.mlp(self.pool(
            batch_dict['x'], batch_dict['batch'], batch_dict['batch_size']))

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
        n_graphs = int(batch.get('num_graphs', int(graph.max()) + 1))
        return {
            'x': batch.x, 'edge_index': None, 'batch': graph,
            'target': batch.get('theta'), 'edge_attr': None,
            'edge_weight': None, 'cond': batch.get('cond'),
            'batch_size': n_graphs,
        }
