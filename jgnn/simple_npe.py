
from typing import List, Dict, Any, Tuple

import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from ml_collections import ConfigDict
from jgnn import flows_utils
from jgnn import flows_utils, transforms_utils, models_utils, models

class SimpleNPE(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        mlp_args: ConfigDict,
        flows_args: ConfigDict,
        optimizer_args: ConfigDict=None,
        scheduler_args: ConfigDict=None,
        norm_dict: Dict[str, Any]=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.mlp_args = mlp_args
        self.flows_args = flows_args
        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args
        self.norm_dict = norm_dict
        self.save_hyperparameters()
        self._setup_model()

    def _setup_model(self):

        # create the mlp layers
        activation_fn = models_utils.get_activation(self.mlp_args.activation)
        self.mlp = models.MLPBatchNorm(
            input_size=self.input_size,
            hidden_sizes=self.mlp_args.hidden_sizes,
            output_size=self.mlp_args.output_size,
            activation_fn=activation_fn,
            batch_norm=self.mlp_args.batch_norm,
            dropout=self.mlp_args.dropout,
        )

        # create the normalizing flows
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

    def _prepare_batch(self, batch):
        """ Prepare the batch for the model """
        pos, vel_true, theta = batch
        vel_obs, std_values, vel_true = uncertainty_model(
            vel_true, torch.tensor(vel_true_scale, dtype=torch.float32))

        # create the input vector with pos, vel_obs, std_values, and theta
        x = torch.cat([pos, vel_obs, std_values, theta], dim=1)
        y = vel_true
        batch_dict = {
            'x': x,
            'theta': y,
            'batch_size': len(x),
        }
        return batch_dict

    def forward(self, x):
        """ Forward pass through the model """
        flow_context = self.mlp(x)
        return flow_context

    def training_step(self, batch):
        batch_dict = self._prepare_batch(batch)

        # forward pass
        flow_context = self.forward(batch_dict['x'])
        log_prob = self.flows(flow_context).log_prob(batch_dict['theta'])
        loss = -log_prob.mean()

        # log the loss
        self.log(
            'train_loss', loss, on_step=True, on_epoch=True,
            prog_bar=True, batch_size=batch_dict['batch_size'])
        return loss

    def validation_step(self, batch):
        batch_dict = self._prepare_batch(batch)

        # forward pass
        flow_context = self.forward(batch_dict['x'])
        log_prob = self.flows(flow_context).log_prob(batch_dict['theta'])
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
