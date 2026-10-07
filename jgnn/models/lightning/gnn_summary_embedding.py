"""GNN embedding augmented with radius-binned moment summaries."""

from typing import Any, Dict

import torch
import torch.nn as nn

from ..layers import MLP
from ..utils import get_activation
from .gnn_embedding import GNNEmbedding


class GNNSummaryEmbedding(GNNEmbedding):
    """`GNNEmbedding` plus an MLP over `batch.summary`.

    The pooled GNN features and the embedded summary vector are
    concatenated before the projection MLP, so the flow context sees
    both the learned graph representation and the explicit binned
    dispersion / kurtosis profile (see
    `jgnn.transforms.summary.BinnedMoments`). Everything else, including
    the conditional MLP, is inherited unchanged.

    Parameters
    ----------
    summary_args : dict
        `input_size`, `hidden_sizes`, `output_size`, `act_name`
        (and optional `act_args`, `final_act`, `zero_init`) of the
        summary MLP; `zero_init` starts the branch's last layer at zero.
    **kwargs
        Passed to `GNNEmbedding`.
    """

    def __init__(self, summary_args: Dict[str, Any], **kwargs):
        self.summary_args = dict(summary_args)
        super().__init__(**kwargs)

    def _setup_model(self):
        super()._setup_model()
        args = dict(self.summary_args)
        zero_init = args.pop('zero_init', False)
        args['act'] = get_activation(
            args.pop('act_name'), args.pop('act_args', {}))
        self.summary_mlp = MLP(**args)
        if zero_init:
            # Reason: with a free summary branch the model learned to read
            # the 20 summary numbers alone and never used the GNN (gap
            # test T3, run id057udh: scrambling the graphs left the val
            # NLL unchanged). Starting the branch at zero makes the model
            # the plain GNN model at step 0, so the summary can only be
            # taken up where it lowers the loss.
            last = [m for m in self.summary_mlp.modules()
                    if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        # Reason: the parent's MLP was sized for the GNN output alone;
        # rebuild it for the concatenated input.
        mlp_config = dict(self.mlp_args)
        mlp_config['input_size'] = (
            self.gnn_args.hidden_sizes[-1] + self.summary_mlp.output_size)
        mlp_config['act'] = get_activation(
            mlp_config.pop('act_name'), mlp_config.pop('act_args', {}))
        self.mlp = MLP(**mlp_config)

    def embed_data(self, batch_dict):
        """GNN -> [pool, summary MLP] -> MLP, no conditional term."""
        if batch_dict.get('summary') is None:
            raise ValueError(
                'batch has no `summary`; enable apply_summary in '
                'pre_transforms for GNNSummaryEmbedding')
        pooled = self.gnn(
            batch_dict['x'], batch_dict['edge_index'],
            batch=batch_dict['batch'], edge_attr=batch_dict['edge_attr'],
            edge_weight=batch_dict['edge_weight'])
        summary = self.summary_mlp(batch_dict['summary'])
        return self.mlp(torch.cat([pooled, summary], dim=1))

    def forward(self, batch_dict):
        """GNN -> [pool, summary MLP] -> MLP [+ CondMLP]."""
        embedding = self.embed_data(batch_dict)
        if self.conditional_mlp is not None:
            embedding = embedding + self.conditional_mlp(batch_dict['cond'])
        return embedding

    def _prepare_batch(self, batch):
        batch_dict = super()._prepare_batch(batch)
        batch_dict['summary'] = batch.get('summary')
        return batch_dict
