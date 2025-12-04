
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch_geometric.transforms as T
from ml_collections import ConfigDict

from jgnn import flows_utils
from jgnn import flows_utils, transforms_utils, models_utils, models

class NPE(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        featurizer_args: ConfigDict,
        mlp_args: ConfigDict,
        flows_args: ConfigDict,
        pre_transform_args: ConfigDict=None,
        optimizer_args: ConfigDict=None,
        scheduler_args: ConfigDict=None,
        conditional_mlp_args: ConfigDict=None,
        norm_dict: Dict[str, Any]=None,
        freeze_components: List[str]=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.featurizer_args = featurizer_args
        self.mlp_args = mlp_args
        self.flows_args = flows_args
        self.pre_transform_args = pre_transform_args
        self.optimizer_args = optimizer_args or {}
        self.scheduler_args = scheduler_args or {}
        self.conditional_mlp_args = conditional_mlp_args or {}
        self.norm_dict = norm_dict
        self.freeze_components = freeze_components or []
        self.save_hyperparameters()

        self._setup_model()

    def _freeze_module(self, module: nn.Module):
        for param in module.parameters():
            param.requires_grad = False

    def _setup_model(self):

        # create the feauturizer
        if self.featurizer_args.name == 'gnn':
            activation_fn = models_utils.get_activation(self.featurizer_args.activation)
            self.featurizer = models.GNN(
                input_size=self.input_size,
                hidden_sizes=self.featurizer_args.hidden_sizes,
                projection_size=self.featurizer_args.projection_size,
                graph_layer=self.featurizer_args.graph_layer,
                graph_layer_params=self.featurizer_args.graph_layer_params,
                activation_fn=activation_fn,
                pooling=self.featurizer_args.pooling,
                layer_norm=self.featurizer_args.layer_norm,
                norm_first=self.featurizer_args.norm_first,
            )
        else:
            raise ValueError(
                f'Featurizer {featurizer_name} not supported')

        # create the mlp layers
        activation_fn = models_utils.get_activation(self.mlp_args.activation)
        self.mlp = models.MLPBatchNorm(
            input_size=self.featurizer_args.hidden_sizes[-1],
            hidden_sizes=self.mlp_args.hidden_sizes,
            output_size=self.mlp_args.output_size,
            activation_fn=activation_fn,
            batch_norm=self.mlp_args.batch_norm,
            dropout=self.mlp_args.dropout,
        )

        # create the conditional mlp layers for extra conditioning
        if self.conditional_mlp_args is not None:
            activation_fn = models_utils.get_activation(self.conditional_mlp_args.activation)
            self.conditional_mlp = models.MLPBatchNorm(
                input_size=self.conditional_mlp_args.input_size,
                hidden_sizes=self.conditional_mlp_args.hidden_sizes,
                output_size=self.conditional_mlp_args.output_size,
                activation_fn=activation_fn,
                batch_norm=self.conditional_mlp_args.batch_norm,
                dropout=self.conditional_mlp_args.dropout,
            )
        else:
            self.conditional_mlp = None

        # create the flows
        activation_fn = models_utils.get_activation_zuko(self.flows_args.activation)
        self.flows = flows_utils.build_flows(
            context_features=self.mlp_args.output_size,
            features=self.output_size,
            hidden_features=self.flows_args.hidden_sizes,
            num_transforms=self.flows_args.num_transforms,
            num_bins=self.flows_args.num_bins,
            activation=activation_fn,
            randperm=True
        )

        # create pre-transforms
        self.pre_transform = transforms_utils.build_transformation(
            graph_name=self.pre_transform_args.graph_name,
            graph_params=self.pre_transform_args.graph_params,
            random_projection=self.pre_transform_args.get('random_projection', False),
            selection=self.pre_transform_args.get('selection', False),
            selection_args=self.pre_transform_args.get('selection_params', {}),
            uncertainty=self.pre_transform_args.get('uncertainty', False),
            uncertainty_args=self.pre_transform_args.get('uncertainty_params', {}),
            norm_dict=self.norm_dict
        )

        # freeze components if needed
        for component in self.freeze_components:
            if hasattr(self, component):
                print(f'Freezing component: {component}')
                self._freeze_module(getattr(self, component))
            else:
                raise ValueError(f'Component {component} not found in the model.')

    def _prepare_batch(self, batch):
        """ Prepare the batch for the model """
        if self.pre_transform is not None:
            batch = self.pre_transform(batch)
        batch = batch.to(self.device)
        if not hasattr(batch, 'cond'):
            cond = None
        else:
            cond = batch.cond
        batch_dict = {
            'x': batch.x,
            'theta': batch.theta,
            'edge_index': batch.edge_index,
            'edge_attr': batch.edge_attr,
            'edge_weight': batch.edge_weight,
            'batch': batch.batch,
            'batch_size': len(batch),
            'cond': cond
        }
        return batch_dict

    def forward(self, x, edge_index, batch, edge_attr, edge_weight, cond=None):
        flow_context = self.featurizer(
            x, edge_index, batch=batch, edge_attr=edge_attr, edge_weight=edge_weight)
        flow_context = self.mlp(flow_context)

        if self.conditional_mlp_args is not None:
            cond = self.conditional_mlp(cond)
            flow_context = flow_context + cond

        return flow_context

    def training_step(self, batch, batch_idx):
        batch_dict = self._prepare_batch(batch)

        # forward pass
        flow_context = self.forward(
            batch_dict['x'], batch_dict['edge_index'],
            batch=batch_dict['batch'],
            edge_attr=batch_dict['edge_attr'],
            edge_weight=batch_dict['edge_weight'],
            cond=batch_dict['cond']
        )
        log_prob = self.flows(flow_context).log_prob(batch_dict['theta'])
        loss = -log_prob.mean()

        # log the loss
        self.log(
            'train/loss', loss, on_step=True, on_epoch=True,
            prog_bar=True, batch_size=batch_dict['batch_size'])
        self.log(
            'train_loss', loss, on_step=True, on_epoch=True,
            prog_bar=True, batch_size=batch_dict['batch_size'])
        return loss

    def validation_step(self, batch, batch_idx):
        batch_dict = self._prepare_batch(batch)

        # forward pass
        flow_context = self.forward(
            batch_dict['x'], batch_dict['edge_index'],
            batch=batch_dict['batch'],
            edge_attr=batch_dict['edge_attr'],
            edge_weight=batch_dict['edge_weight'],
            cond=batch_dict['cond']
        )
        log_prob = self.flows(flow_context).log_prob(batch_dict['theta'])
        loss = -log_prob.mean()

        # log the loss
        self.log(
            'val/loss', loss, on_step=False, on_epoch=True,
            prog_bar=True, batch_size=batch_dict['batch_size'])
        self.log(
            'val_loss', loss, on_step=False, on_epoch=True,
            prog_bar=True, batch_size=batch_dict['batch_size'])
        return loss

    def configure_optimizers(self):
        """ Initialize optimizer and LR scheduler """
        return models_utils.configure_optimizers(
            self.parameters(), self.optimizer_args, self.scheduler_args)
