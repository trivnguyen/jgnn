
import os
import pickle
import sys
import shutil

import ml_collections
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
from torch.utils.data import DataLoader, TensorDataset
from absl import flags, logging
from ml_collections import config_flags

import datasets
from models.zuko import npe as zuko_npe
from models import models, npe, utils

logging.set_verbosity(logging.INFO)

def train(
    config: ml_collections.ConfigDict, workdir: str = "./logging/"
):
    # set up work directory
    if not hasattr(config, "name"):
        name = utils.get_random_name()
    else:
        name = config["name"]
    logging.info("Starting training run {} at {}".format(name, workdir))

    # set up random seed
    pl.seed_everything(config.seed)

    workdir = os.path.join(workdir, name)
    checkpoint_path = None
    if os.path.exists(workdir):
        if config.overwrite:
            shutil.rmtree(workdir)
        elif config.get('checkpoint', None) is not None:
            checkpoint_path = os.path.join(
                workdir, 'lightning_logs/checkpoints', config.checkpoint)
        else:
            raise ValueError(
                f"Workdir {workdir} already exists. Please set overwrite=True "
                "to overwrite the existing directory.")

    # read in the dataset and prepare the data loader for training
    node_feats, graph_feats = datasets.read_datasets(
        config.data_root, config.data_name, config.num_datasets,
        config.is_directory, concat=True)
    train_loader, val_loader, norm_dict = datasets.prepare_dataloaders(
        node_feats, graph_feats, config.labels, train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size, train_frac=config.train_frac,
        num_workers=config.num_workers, seed=config.seed,
    )

    # create model
    if config.model.zuko:
        model = zuko_npe.NPE(
            input_size=config.model.input_size,
            output_size=config.model.output_size,
            featurizer_args=config.model.featurizer,
            mlp_args=config.model.mlp,
            flows_args=config.model.flows,
            pre_transform_args=config.model.pre_transform,
            optimizer_args=config.optimizer,
            scheduler_args=config.scheduler,
            norm_dict=norm_dict,
        )
    else:
        model = npe.NPE(
            input_size=config.model.input_size,
            output_size=config.model.output_size,
            featurizer_args=config.model.featurizer,
            mlp_args=config.model.mlp,
            flows_args=config.model.flows,
            pre_transform_args=config.model.pre_transform,
            optimizer_args=config.optimizer,
            scheduler_args=config.scheduler,
            norm_dict=norm_dict,
        )

    # create the trainer object
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor=config.monitor, patience=config.patience, mode=config.mode,
            verbose=True),
        pl.callbacks.ModelCheckpoint(
            filename="{epoch}-{val_loss:.4f}", monitor=config.monitor,
            save_top_k=config.save_top_k, mode=config.mode,
            save_weights_only=False),
        pl.callbacks.LearningRateMonitor("step"),
    ]
    train_logger = pl_loggers.TensorBoardLogger(workdir, version='')
    trainer = pl.Trainer(
        default_root_dir=workdir,
        max_epochs=config.num_epochs,
        accelerator=config.accelerator,
        callbacks=callbacks,
        logger=train_logger,
        enable_progress_bar=config.get("enable_progress_bar", True),
    )

    # train the model
    logging.info("Training model...")
    trainer.fit(
        model, train_loader, val_loader,
        ckpt_path=checkpoint_path
    )

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config",
        None,
        "File path to the training or sampling hyperparameter configuration.",
        lock_config=True,
    )
    # Parse flags
    FLAGS(sys.argv)

    # Start training run
    train(config=FLAGS.config, workdir=FLAGS.config.workdir)