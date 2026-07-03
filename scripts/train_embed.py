"""Training script for the Neural Posterior Estimation embedding model."""

import sys

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import wandb
import ml_collections
import pytorch_lightning as pl
from absl import flags
from ml_collections import config_flags

from jgnn import datasets, training
from jgnn.models import GNNEmbedding, TransformerEmbedding
from jgnn.transforms import build_transformation


def prepare_data(config: ml_collections.ConfigDict):
    """Load and prepare datasets with transformations.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (train_loader, val_loader, norm_dict)
    """
    node_feats, graph_feats = datasets.read_datasets(
        config.data_root,
        config.data_name,
        config.num_datasets,
        init=config.get('init', 0),
        concat=True,
    )

    train_loader, val_loader, norm_dict = datasets.prepare_dataloaders(
        node_feats,
        graph_feats,
        config.labels,
        cond_labels=config.get('cond_labels', None),
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        train_frac=config.train_frac,
        num_workers=config.num_workers,
        seed=config.seed_data,
    )

    return train_loader, val_loader, norm_dict


def create_model(
    config: ml_collections.ConfigDict,
    pre_transforms,
    norm_dict
):
    """Create the GNN or Transformer embedding model.

    Args:
        config: Configuration dictionary
        pre_transforms: Pre-transformation pipeline
        norm_dict: Normalization dictionary to pass to the model

    Returns:
        Embedding model instance
    """
    if config.model.type == 'gnn':
        print("[Model] Creating GNN Embedding model...")
        return GNNEmbedding(
            input_size=config.model.input_size,
            gnn_args=config.model.gnn,
            mlp_args=config.model.mlp,
            loss_type=config.model.loss_type,
            loss_args=config.model.get('loss_args', None),
            conditional_mlp_args=config.model.get('conditional_mlp', None),
            optimizer_args=config.optimizer,
            scheduler_args=config.scheduler,
            pre_transforms=pre_transforms,
            norm_dict=norm_dict,
        )
    elif config.model.type == 'transformer':
        print("[Model] Creating Transformer Embedding model...")
        return TransformerEmbedding(
            input_size=config.model.input_size,
            transformer_args=config.model.transformer,
            loss_type=config.model.loss_type,
            loss_args=config.model.get('loss_args', None),
            mlp_args=config.model.get('mlp', None),
            optimizer_args=config.optimizer,
            scheduler_args=config.scheduler,
            pre_transforms=pre_transforms,
            norm_dict=norm_dict,
        )
    else:
        raise ValueError(f"Unknown model type: {config.model.type}")


def main(config: ml_collections.ConfigDict, workdir: str = "./logging/"):
    """Train the GNN embedding model with wandb logging.

    Args:
        config: Configuration dictionary containing model and training parameters
        workdir: Working directory for logging and checkpoints
    """
    resume_training = config.get('checkpoint') is not None
    print(f"[Setup] Resume training: {resume_training}")
    print(f"[Setup] Working directory: {workdir}")

    run_dir = training.setup_workdir(workdir)
    print(f"[Setup] Run directory: {run_dir}")

    wandb_logger = training.create_wandb_logger(config, run_dir, tag='embedding')

    print("[Data] Loading datasets...")
    train_loader, val_loader, norm_dict = prepare_data(config)
    print(f"[Data] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    print("[Transforms] Building pre-transforms...")
    pre_transforms = build_transformation(
        norm_dict=norm_dict, **config.pre_transforms)

    model = create_model(config, pre_transforms, norm_dict)
    training.report_param_counts(model)

    # this watches all parameters and gradients
    wandb_logger.watch(model, log="all", log_freq=500, log_graph=True)

    checkpoint_path = None
    if resume_training:
        checkpoint_path = training.get_checkpoint_path(config, run_dir)
        print(f"[Checkpoint] Resuming from: {checkpoint_path}")
        print(f"[Checkpoint] Reset optimizer: {config.get('reset_optimizer', False)}")

    callbacks = training.create_base_callbacks(config)
    print(f"[Callbacks] Created {len(callbacks)} callbacks")

    trainer = training.build_trainer(config, run_dir, callbacks, wandb_logger)

    pl.seed_everything(config.seed_training, workers=True)
    print(f"[Seed] Training seed set to: {config.seed_training}")

    training.fit(
        trainer, model, train_loader, val_loader,
        checkpoint_path=checkpoint_path,
        reset_optimizer=config.get('reset_optimizer', False),
    )

    wandb.finish()
    print("[WandB] Finished")


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config",
        None,
        "File path to the training hyperparameter configuration.",
        lock_config=True,
    )
    FLAGS(sys.argv)
    main(config=FLAGS.config, workdir=FLAGS.config.workdir)
