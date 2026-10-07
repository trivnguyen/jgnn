"""Flow context from a graph-level summary alone: no graph, no GNN."""

from typing import Any, Dict

from ..layers import MLP
from ..utils import get_activation, build_embedding_loss
from .gnn_embedding import GNNEmbedding


class SummaryEmbedding(GNNEmbedding):
    """summary MLP -> projection MLP [+ CondMLP] over `batch.summary`.

    Built for the oracle sufficient-statistic test
    (`jgnn.transforms.summary.OracleGaussianLogLike`), where the flow is
    conditioned on a summary that carries the whole likelihood, so the
    comparison with the exact posterior isolates the flow and training
    from the embedding. Works with any summary transform. `gnn_args` is
    accepted and ignored so configs and loaders keep one shape; the
    projection MLP and the conditional MLP are the parent's.

    Parameters
    ----------
    summary_args : dict
        `input_size`, `hidden_sizes`, `output_size`, `act_name` (and
        optional `act_args`, `final_act`) of the summary MLP.
    **kwargs
        Passed to `GNNEmbedding`.
    """

    def __init__(self, summary_args: Dict[str, Any], gnn_args=None,
                 **kwargs):
        self.summary_args = dict(summary_args)
        super().__init__(gnn_args=gnn_args or {}, **kwargs)

    def _setup_model(self):
        args = dict(self.summary_args)
        args['act'] = get_activation(
            args.pop('act_name'), args.pop('act_args', {}))
        self.summary_mlp = MLP(**args)
        self.gnn = None
        mlp_config = dict(self.mlp_args)
        mlp_config['input_size'] = self.summary_mlp.output_size
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

    def embed_data(self, batch_dict):
        """summary MLP -> MLP, no conditional term."""
        if batch_dict.get('summary') is None:
            raise ValueError(
                'batch has no `summary`; enable apply_summary in '
                'pre_transforms for SummaryEmbedding')
        return self.mlp(self.summary_mlp(batch_dict['summary']))

    def forward(self, batch_dict):
        """summary MLP -> MLP [+ CondMLP]."""
        embedding = self.embed_data(batch_dict)
        if self.conditional_mlp is not None:
            embedding = embedding + self.conditional_mlp(batch_dict['cond'])
        return embedding

    def _prepare_batch(self, batch):
        batch = self.pre_transforms(batch) if self.pre_transforms else batch
        batch = batch.to(self.device)
        summary = batch.get('summary')
        return {
            'x': batch.get('x'),
            'edge_index': batch.get('edge_index'),
            'batch': batch.get('batch'),
            'summary': summary,
            'target': batch.get('theta'),
            'edge_attr': batch.get('edge_attr'),
            'edge_weight': batch.get('edge_weight'),
            'cond': batch.get('cond'),
            'batch_size': summary.shape[0] if summary is not None else
            batch.get('num_graphs'),
        }
