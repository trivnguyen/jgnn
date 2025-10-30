
import os
import pickle
import sys
import shutil

import yaml
import ml_collections
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
from torch.utils.data import DataLoader, TensorDataset
from absl import flags, logging
from ml_collections import config_flags

import datasets
from jgnn import models, npe, utils

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

    workdir = os.path.join(workdir, name)
    checkpoint_path = None

    if config.get('checkpoint', None) is not None:
        if os.path.isabs(config.checkpoint):
            checkpoint_path = config.checkpoint
        else:
            checkpoint_path = os.path.join(
                workdir, 'lightning_logs/checkpoints', config.checkpoint)

    if os.path.exists(workdir):
        if config.overwrite:
            shutil.rmtree(workdir)
            os.makedirs(workdir, exist_ok=True)
        elif checkpoint_path is None:
            raise ValueError(
                f"Workdir {workdir} already exists. Please set overwrite=True "
                "to overwrite the existing directory, or specify a checkpoint to resume.")
    else:
        os.makedirs(workdir, exist_ok=True)


    # copy yaml file
    os.makedirs(workdir, exist_ok=True)
    config_dict = ml_collections.ConfigDict.to_dict(config)
    with open(os.path.join(workdir, 'config.yaml'), 'w') as f:
        yaml.dump(config_dict, f)

    # read in the dataset and prepare the data loader for training
    node_feats, graph_feats = datasets.read_datasets(
        config.data_root, config.data_name, config.num_datasets,
        config.is_directory, concat=True)
    train_loader, val_loader, norm_dict = datasets.prepare_dataloaders(
        node_feats, graph_feats, config.labels, train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size, train_frac=config.train_frac,
        num_workers=config.num_workers, seed=config.seed_data,
        norm_version=config.get('norm_version', 'v2'),
    )

    # create model
    model = npe.NPE(
        input_size=config.model.input_size,
        output_size=config.model.output_size,
        featurizer_args=config.model.featurizer,
        mlp_args=config.model.mlp,
        flows_args=config.model.flows,
        pre_transform_args=config.model.pre_transform,
        optimizer_args=config.optimizer,
        scheduler_args=config.scheduler,
        conditional_mlp_args=config.model.get('conditional_mlp', None),
        norm_dict=norm_dict,
    )

    # create the trainer object
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor=config.monitor, patience=config.patience, mode=config.mode,
            verbose=True),
        pl.callbacks.ModelCheckpoint(
            filename="{epoch}-{step}-{val_loss:.4f}", monitor=config.monitor,
            save_top_k=config.save_top_k, mode=config.mode,
            save_weights_only=False),
        pl.callbacks.ModelCheckpoint(
            filename="last", save_top_k=0, save_weights_only=False),
        pl.callbacks.LearningRateMonitor("step"),
    ]
    train_logger = pl_loggers.TensorBoardLogger(workdir, version='')
    trainer = pl.Trainer(
        default_root_dir=workdir,
        max_epochs=config.num_epochs,
        max_steps=config.num_steps,
        accelerator=config.accelerator,
        callbacks=callbacks,
        logger=train_logger,
        enable_progress_bar=config.get("enable_progress_bar", True),
    )

    # train the model
    logging.info("Training model...")
    pl.seed_everything(config.seed_training)

    # Handle transfer learning: load checkpoint but reset optimizer if requested
    if checkpoint_path is not None and config.get('reset_optimizer', False):
        logging.info(f"Loading checkpoint from {checkpoint_path} with fresh optimizer")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        trainer.fit(model, train_loader, val_loader)
    else:
        logging.info(f"Loading checkpoint from {checkpoint_path} with full state")
        trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)


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