"""
Training script for a normalizing flow-based neural density estimator (NDE),
given a trained compressor model.
"""


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
from jgnn import models, npe

logging.set_verbosity(logging.INFO)

def train(
    config: ml_collections.ConfigDict, workdir: str = "./logging/"
):
    # set up work directory
    name = config.get("name", "default_nde")
    logging.info("Starting training run {} at {}".format(name, workdir))

    workdir = os.path.join(workdir, name)

    # read in the checkpoint of the Embedding network
    if config.get('checkpoint', None) is not None:
        if os.path.isabs(config.checkpoint):
            checkpoint_path = config.checkpoint
        else:
            checkpoint_path = os.path.join(
                workdir, 'lightning_logs/checkpoints', config.checkpoint)
    else:
        raise ValueError("Checkpoint must be specified.")
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

    # overwrite if specified
    if os.path.exists(workdir) and config.overwrite:
        shutil.rmtree(workdir)
    os.makedirs(workdir, exist_ok=True)

    # copy yaml file
    os.makedirs(workdir, exist_ok=True)
    config_dict = ml_collections.ConfigDict.to_dict(config)
    with open(os.path.join(workdir, 'config.yaml'), 'w') as f:
        yaml.dump(config_dict, f)

    # read in the dataset and prepare the data loader for training
    # NOTE: always keep norm_dict the same as the Embedding network, since we
    # only train the NDE on the latent space
    node_feats, graph_feats = datasets.read_datasets(
        config.data_root, config.data_name, config.num_datasets,
        init=config.init, is_directory=True, concat=True)
    train_loader, val_loader, norm_dict = datasets.prepare_dataloaders(
        node_feats, graph_feats, config.labels, train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size, train_frac=config.train_frac,
        num_workers=config.num_workers, seed=config.seed_data,
        norm_version=config.get('norm_version', 'v2'),
        norm_dict=checkpoint_dict['hyper_parameters']['norm_dict']
    )

    # check if norm_dict is the same as the checkpoint
    assert norm_dict == checkpoint_dict['hyper_parameters']['norm_dict'], \
        "Something's wrong, norm_dict does not match the checkpoint's norm_dict."

    # create model
    model = npe.NPE(
        input_size=config.model.input_size,
        output_size=config.model.output_size,
        featurizer_args=config.model.featurizer,
        mlp_args=config.model.mlp,
        flows_args=config.model.flows,
        pre_transform_args=config.model.pre_transform,
        optimizer_args=config.get('optimizer'),
        scheduler_args=config.get('scheduler'),
        conditional_mlp_args=config.model.get('conditional_mlp'),
        freeze_components=['featurizer', 'mlp', 'conditional_mlp'],
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
        gradient_clip_val=config.get("gradient_clip_val", None)
    )

    # train the model
    logging.info("Training model...")
    pl.seed_everything(config.seed_training)
    # load in the previous checkpoint
    if config.get('reset_optimizer', False):
        state_to_load = {
            k: v for k, v in checkpoint_dict['state_dict'].items() if k in config.load_parts}
        model.load_state_dict(state_to_load, strict=False)
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
