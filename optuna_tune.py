
import os
import pickle
import sys
import shutil
import functools

import yaml
import ml_collections
import numpy as np
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.trial import FixedTrial
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
from torch.utils.data import DataLoader, TensorDataset
from absl import flags, logging
from ml_collections import config_flags, ConfigDict

import datasets
from jgnn import models, npe, utils


logging.set_verbosity(logging.INFO)


def objective(trial, config):

    activation = trial.suggest_categorical("activation", ["gelu", "silu", "relu"])
    flows_context_size = trial.suggest_categorical("flows_context_size", [32, 64, 128])

   # Define hyperparameters to optimize
    featurizer_args = ConfigDict(dict(
        name='gnn',
        graph_layer='GATConv',
        projection_size=trial.suggest_categorical("featurizer_projection_size", [32, 64, 128]),
        activation=ConfigDict(dict(name=activation)),
        hidden_sizes=[trial.suggest_categorical("featurizer_hidden_size", [32, 64, 128])] * trial.suggest_int("featurizer_width", 2, 6),
        graph_layer_params=ConfigDict(dict(
            nheads=trial.suggest_categorical("featurizer_nheads", [1, 2, 4, 8]),
            concat=False
        )),
        pooling='mean',
        layer_norm=trial.suggest_categorical("featurizer_layer_norm", [True, False]),
        norm_first=trial.suggest_categorical("featurizer_norm_first", [True, False]),
    ))
    mlp_args = ConfigDict(dict(
        activation=ConfigDict(dict(name=activation)),
        hidden_sizes=[trial.suggest_categorical("mlp_hidden_size", [32, 64, 128])] * trial.suggest_int("mlp_width", 2, 6),
        output_size=flows_context_size,
        dropout=trial.suggest_float("mlp_dropout", 0.2, 0.8, step=0.1),
        batch_norm=trial.suggest_categorical("mlp_batch_norm", [True, False]),
    ))
    conditional_mlp_args = ConfigDict(dict(
        activation=ConfigDict(dict(name=activation)),
        input_size=1,
        hidden_sizes=[trial.suggest_categorical("conditional_mlp_hidden_size", [16, 32, 64])] * trial.suggest_int("conditional_mlp_width", 1, 4),
        output_size=flows_context_size,
        dropout=trial.suggest_float("conditional_mlp_dropout", 0.0, 0.5, step=0.1),
        batch_norm=False,
    ))
    flows_args = ConfigDict(dict(
        hidden_sizes=[trial.suggest_categorical("flows_hidden_size", [32, 64, 128])] * trial.suggest_int("flows_width", 2, 6),
        num_transforms=trial.suggest_int("flows_num_transforms", 2, 8),
        num_bins=trial.suggest_int("flows_num_bins", 4, 16),
        activation=ConfigDict(dict(name=activation)),
    ))

    # Optimizer hyperparameters
    optimizer_args = ConfigDict(dict(
        name='AdamW',
        lr=trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        eps=1e-9,
        betas=(0.9, 0.98),
    ))

    # Scheduler hyperparameters
    decay_steps = trial.suggest_int('decay_steps', 100_000, 250_000)  # need for later
    scheduler_args = ConfigDict(dict(
        name='WarmUpCosineAnnealingLR',
        warmup_steps=trial.suggest_int('warmup_steps', 5000, 25_000),
        decay_steps=decay_steps,
        eta_min=trial.suggest_float('eta_min_factor', 0.001, 0.1, log=True),
        interval='step'
    ))

    # read in the dataset and prepare the data loader for training
    node_feats, graph_feats = datasets.read_datasets(
        config.data_root,
        config.data_name,
        config.num_datasets,
        config.is_directory,
        concat=True
    )
    train_loader, val_loader, norm_dict = datasets.prepare_dataloaders(
        node_feats,
        graph_feats,
        config.labels,
        train_batch_size=trial.suggest_categorical('batch_size', [64, 128, 256]),
        eval_batch_size=config.eval_batch_size,
        train_frac=config.train_frac,
        num_workers=config.num_workers,
        seed=config.seed_data,
        norm_version=config.get('norm_version', 'v2'),
    )

    # create model
    model = npe.NPE(
        # fixed parameters
        input_size=config.model.input_size,
        output_size=config.model.output_size,
        pre_transform_args=config.model.pre_transform,
        norm_dict=norm_dict,
        # tuneable hyperparameters
        featurizer_args=featurizer_args,
        mlp_args=mlp_args,
        conditional_mlp_args=conditional_mlp_args,
        flows_args=flows_args,
        optimizer_args=optimizer_args,
        scheduler_args=scheduler_args,
    )
    # Create callbacks with Optuna pruning but without checkpointing during trials
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
            verbose=True
        ),
        pl.callbacks.LearningRateMonitor("step"),
    ]
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")

    # Minimal logger to reduce overhead
    train_logger = pl_loggers.TensorBoardLogger(
        os.path.join(config.workdir, 'optuna_logs'), name=f"trial_{trial.number}")

    # Create trainer with properly separated callbacks
    trainer = pl.Trainer(
        default_root_dir=os.path.join(config.workdir, 'optuna_logs'),
        max_epochs=100,
        max_steps=decay_steps,
        accelerator='auto',
        callbacks=callbacks,
        logger=train_logger,
        gradient_clip_val=0.5,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.callbacks.append(pruning_callback)

    # train the model
    logging.info("Training model...")
    trainer.fit(model, train_loader, val_loader)

    return trainer.callback_metrics['val_loss'].item()


def main():
    # Parse command line arguments
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config",
        None,
        "File path to the training or sampling hyperparameter configuration.",
        lock_config=True,
    )
    FLAGS(sys.argv)
    config = FLAGS.config

    if config.overwrite:
        if os.path.exists(config.workdir):
            shutil.rmtree(config.workdir)
    os.makedirs(config.workdir, exist_ok=True)

    # Pruner
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=config.optuna.n_startup_trials,
        n_warmup_steps=config.optuna.n_warmup_steps,
        interval_steps=config.optuna.interval_steps
    )

    # Create storage path with workdir
    storage_path = config.optuna.get("storage", "sqlite:///optuna_studies.db")

    # If it's a relative path (not starting with sqlite://, mysql://, postgresql://)
    if not storage_path.startswith(("sqlite://", "mysql://", "postgresql://")):
        # Check if the path is absolute or relative
        if not os.path.isabs(storage_path):
            # If it's a relative path, prepend the workdir
            storage_file = os.path.join(config.workdir, storage_path)
            # Make sure the directory exists
            os.makedirs(os.path.dirname(storage_file), exist_ok=True)
            # Create the SQLite URI
            storage_path = f"sqlite:///{storage_file}"
        else:
            # It's an absolute path, just convert to SQLite URI
            storage_path = f"sqlite:///{storage_path}"

    # Create the study
    study = optuna.create_study(
        direction='minimize',
        pruner=pruner,
        study_name=config.optuna.get("study_name", "npe_hyperparameter_optimization"),
        storage=storage_path,
        load_if_exists=config.optuna.get("load_if_exists", True)
    )

    # Run the optimization
    objective_with_config = functools.partial(objective, config=config)
    study.optimize(
        objective_with_config,
        n_trials=config.optuna.get("n_trials", 100),
        timeout=config.optuna.get("timeout", None)
    )

    # Print results
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Save best parameters to file for later use
    output_dir = os.path.join(config.workdir, 'optuna_results')
    os.makedirs(output_dir, exist_ok=True)

    best_params_path = os.path.join(output_dir, 'best_params.json')
    with open(best_params_path, 'w') as f:
        json.dump(best_trial.params, f, indent=4)
    print(f"Best parameters saved to '{best_params_path}'")


if __name__ == "__main__":
    main()
