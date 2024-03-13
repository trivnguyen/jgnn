
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch_geometric.transforms as T
from ml_collections import ConfigDict

from . import models, models_utils, flows_utils, transforms_utils

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
        norm_dict: Dict[str, Any]=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.featurizer_args = featurizer_args
        self.mlp_args = mlp_args
        self.flows_args = flows_args
        self.pre_transform_args = pre_transform_args
        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args
        self.norm_dict = norm_dict
        self.pre_tranform = None
        self.save_hyperparameters()

        self._setup_model()

    def _setup_model(self):

        # create the feauturizer
        if self.featurizer_args.name == 'gnn':
            activation_fn = models_utils.get_activation(
                self.featurizer_args.activation)
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
        activation_fn = models_utils.get_activation(
            self.mlp_args.activation)
        self.mlp = models.MLP(
            input_size=self.featurizer_args.hidden_sizes[-1],
            hidden_sizes=self.mlp_args.hidden_sizes,
            output_size=self.mlp_args.output_size,
            activation_fn=activation_fn,
        )

        # create the flows
        activation_fn = models_utils.get_activation(
            self.flows_args.activation)
        self.flows = flows_utils.build_maf(
            context_features=self.mlp_args.output_size,
            hidden_features=self.flows_args.hidden_size,
            features=self.output_size,
            num_layers=self.flows_args.num_layers,
            num_blocks=self.flows_args.num_blocks,
            activation_fn=activation_fn,
            batch_norm=self.flows_args.batch_norm,
        )

        # create pre-transforms
        self.pre_transform = transforms_utils.build_transformation(
            graph_name=self.pre_transform_args.graph_name,
            graph_params=self.pre_transform_args.graph_params
        )

    def _prepare_batch(self, batch):
        """ Prepare the batch for the model """
        if self.pre_transform is not None:
            batch = self.pre_transform(batch)
        batch = batch.to(self.device)

        batch_dict = {
            'x': batch.x,
            'theta': batch.theta,
            'edge_index': batch.edge_index,
            'edge_attr': batch.edge_attr,
            'edge_weight': batch.edge_weight,
            'batch': batch.batch,
            'batch_size': len(batch)
        }
        return batch_dict

    def forward(self, x, edge_index, batch, edge_attr, edge_weight):
        flow_context = self.featurizer(
            x, edge_index, batch=batch, edge_attr=edge_attr, edge_weight=edge_weight)
        flow_context = self.mlp(flow_context)
        return flow_context


    def training_step(self, batch, batch_idx):
        batch_dict = self._prepare_batch(batch)

        # forward pass
        flow_context = self.forward(
            batch_dict['x'], batch_dict['edge_index'],
            batch=batch_dict['batch'],
            edge_attr=batch_dict['edge_attr'],
            edge_weight=batch_dict['edge_weight']
        )
        log_prob = self.flows.log_prob(
            batch_dict['theta'], context=flow_context)
        loss = -log_prob.mean()

        # log the loss
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
            edge_weight=batch_dict['edge_weight']
        )
        log_prob = self.flows.log_prob(
            batch_dict['theta'], context=flow_context)
        loss = -log_prob.mean()

        # log the loss
        self.log(
            'val_loss', loss, on_step=True, on_epoch=True,
            prog_bar=True, batch_size=batch_dict['batch_size'])
        return loss

    def configure_optimizers(self):
        """ Initialize optimizer and LR scheduler """
        return models_utils.configure_optimizers(
            self.parameters(), self.optimizer_args, self.scheduler_args)
