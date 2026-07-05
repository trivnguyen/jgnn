"""Training script for Neural Posterior Estimation (NPE)."""

import json
import os
import shutil
import sys
from pathlib import Path

os.environ['WANDB_DATA_DIR'] = '/scratch/tvnguyen/wandb_data'

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import wandb
import ml_collections
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import summarize
import torch
from absl import flags
from ml_collections import config_flags

from jgnn import datasets, training
from jgnn.models import NPE, GNNEmbedding
from jgnn.transforms import build_transformation
from jgnn.callbacks.visualization import NPEVisualizationCallback


def save_config_snapshot(
    config: ml_collections.ConfigDict, snapshot_dir: Path, config_path: str = None,
) -> None:
    """Write a config snapshot into snapshot_dir, next to where the checkpoint lands.

    Downstream consumers that only have the checkpoint file (e.g.
    tsnpe/register_run.py's local_checkpoint_dir path) can read
    snapshot_dir/config_snapshot.json to reconstruct model/pre_transforms
    without needing wandb.

    Args:
        config: Full training config.
        snapshot_dir: Directory to write into - the same one
            ModelCheckpoint writes checkpoints/ under (see main()), not
            run_dir's root, since a workdir can accumulate multiple runs.
        config_path: Path to the source config.py file, if known - also
            copied verbatim (e.g. snapshot_dir/config_snapshot.py).
    """
    snapshot_path = snapshot_dir / 'config_snapshot.json'
    with open(snapshot_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2, default=str)
    print(f"[Setup] Wrote config snapshot -> {snapshot_path}")

    if config_path and os.path.exists(config_path):
        shutil.copy2(config_path, snapshot_dir / 'config_snapshot.py')


def prepare_data(config: ml_collections.ConfigDict, norm_dict=None):
    """Load and prepare datasets with transformations.

    Args:
        config: Configuration dictionary
        norm_dict: Fixed normalization dict to reuse (e.g. from a resumed
            checkpoint's own hyper_parameters) instead of computing a fresh
            one from this call's training data.

    Returns:
        Tuple of (train_loader, val_loader, norm_dict)

    Config fields
    -------------
    dataset_type : str, default 'cartesian'
        'cartesian' — 3-D Cartesian phase-space (sample_galaxies.py output)
                      node features: pos, vel, vel_error
        'icrs'      — sky-plane ICRS observables (sample_galaxies_target.py output)
                      node features: ra, dec, vlos, R_proj, vlos_err
    """
    node_feats, graph_feats = datasets.read_datasets(
        config.data_root,
        config.data_name,
        config.num_datasets,
        init=config.get('init', 0),
        is_directory=True,
        concat=True
    )

    train_loader, val_loader, norm_dict = datasets.cartesian.prepare_dataloaders(
        node_feats,
        graph_feats,
        config.labels,
        cond_labels=config.get('cond_labels', None),
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        train_frac=config.train_frac,
        num_workers=config.num_workers,
        seed=config.seed_data,
        norm_dict=norm_dict,
        pre_transform_kwargs=dict(config.pre_transforms),
    )

    return train_loader, val_loader, norm_dict


def load_embedding_network(
    config: ml_collections.ConfigDict, checkpoint_path: str, freeze: bool = False):
    """Load a pre-trained embedding network from checkpoint.

    Args:
        config: Configuration dictionary
        checkpoint_path: Path to checkpoint
        freeze: If True, freeze all parameters of the embedding network

    Returns:
        The loaded GNNEmbedding model.
    """
    print(f"[Embedding] Loading pre-trained embedding network from: {checkpoint_path}")

    embedding_nn = GNNEmbedding.load_from_checkpoint(checkpoint_path)

    if freeze:
        for param in embedding_nn.parameters():
            param.requires_grad = False
        embedding_nn.eval()
        print(f"[Embedding] Froze all parameters in embedding network")

    print(f"[Embedding] Embedding network loaded successfully")
    print(f"[Embedding] Output size: {embedding_nn.output_size}")

    return embedding_nn


def create_embedding_network(config: ml_collections.ConfigDict):
    """Create a new embedding network.

    Args:
        config: Configuration dictionary

    Returns:
        Embedding network instance
    """
    print("[Embedding] Creating GNN Embedding model...")

    model_type = config.model.embedding.get('type', 'gnn')
    return GNNEmbedding(
        input_size=config.model.input_size,
        gnn_args=config.model.embedding.gnn,
        mlp_args=config.model.embedding.mlp,
        loss_type=config.model.embedding.get('loss_type', 'mse'),
        loss_args=config.model.embedding.get('loss_args', None),
        conditional_mlp_args=config.model.embedding.get('conditional_mlp', None),
        # NPE handles optimizer, scheduler, and pre_transforms
        optimizer_args=None,
        scheduler_args=None,
        pre_transforms=None,
    )


def create_model(
    config: ml_collections.ConfigDict,
    pre_transforms,
    norm_dict
) -> NPE:
    """Create the NPE model with optional pre-trained embedding network.

    Args:
        config: Configuration dictionary
        pre_transforms: Pre-transformation pipeline (passed to NPE)
        norm_dict: Normalization dictionary to pass to NPE

    Returns:
        NPE model instance
    """
    embedding_checkpoint = config.model.embedding.get('checkpoint', None)
    freeze_embedding = config.model.embedding.get('freeze', False)

    if embedding_checkpoint is not None:
        embedding_nn = load_embedding_network(
            config, embedding_checkpoint, freeze=freeze_embedding)
    else:
        print("[Model] Creating new embedding network...")
        embedding_nn = create_embedding_network(config)

    print("[Model] Creating NPE model...")
    init_flows_from_embedding = config.model.get('init_flows_from_embedding', False)

    return NPE(
        input_size=config.model.input_size,
        output_size=config.model.output_size,
        flows_args=config.model.flows,
        embedding_nn=embedding_nn,
        optimizer_args=config.optimizer,
        scheduler_args=config.scheduler,
        norm_dict=norm_dict,
        pre_transforms=pre_transforms,
        init_flows_from_embedding=init_flows_from_embedding,
    )


def create_callbacks(config: ml_collections.ConfigDict) -> list:
    """Create PyTorch Lightning callbacks, including NPE-specific visualization.

    Args:
        config: Configuration dictionary

    Returns:
        List of callback instances
    """
    callbacks = training.create_base_callbacks(config)

    if config.get('enable_visualization_callback', False):
        print("[Callbacks] Adding NPE Visualization Callback")
        callbacks.append(
            NPEVisualizationCallback(
                plot_every_n_epochs=config.visualization.get('plot_every_n_epochs', 1),
                n_posterior_samples=config.visualization.get('n_posterior_samples', 1000),
                n_val_samples=config.visualization.get('n_val_samples', 100),
                plot_median_v_true=config.visualization.get('plot_median_v_true', True),
                plot_tarp=config.visualization.get('plot_tarp', True),
                plot_rank=config.visualization.get('plot_rank', True),
                use_default_mplstyle=config.visualization.get('use_default_mplstyle', True),
            )
        )
    return callbacks


def main(config: ml_collections.ConfigDict, config_path: str = None):
    """Train the NPE model with wandb logging.

    Args:
        config: Configuration dictionary containing model and training parameters
        config_path: Path to the source config.py file, if known - see
            save_config_snapshot.
    """
    resume_training = config.get('checkpoint') is not None
    wandb_logger, project_dir = training.create_wandb_logger(config, tag='npe')

    print(f"[Setup] Resume training: {resume_training}")
    print(f"[Setup] Project directory: {project_dir}")

    # snapshot config
    print(f"[Setup] Saving config snapshot to: {project_dir}")
    save_config_snapshot(config, project_dir, config_path)

    checkpoint_path = None
    norm_dict = None
    if resume_training:
        checkpoint_path = training.get_checkpoint_path(config, project_dir)
        print(f"[Checkpoint] Resuming from: {checkpoint_path}")
        print(f"[Checkpoint] Reset optimizer: {config.get('reset_optimizer', False)}")

        # Make sure that the norm_dict is reused from the resumed checkpoint's hyper_parameters
        resume_checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False)
        norm_dict = resume_checkpoint['hyper_parameters']['norm_dict']
        print("[Checkpoint] Reusing norm_dict from resumed checkpoint")

    print("[Data] Loading datasets...")
    train_loader, val_loader, norm_dict = prepare_data(config, norm_dict=norm_dict)
    print(f"[Data] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    print("[Transforms] Building pre-transforms...")
    pre_transforms = build_transformation(
        norm_dict=norm_dict, **config.pre_transforms)

    print("[Model] Creating NPE model...")
    model = create_model(config, pre_transforms, norm_dict)
    summarize(model, max_depth=3)
    training.report_param_counts(model)

    # this watches all parameters and gradients
    wandb_logger.watch(model, log="all", log_freq=1000, log_graph=False)

    callbacks = create_callbacks(config)
    print(f"[Callbacks] Created {len(callbacks)} callbacks")

    trainer = training.build_trainer(
        config, project_dir, callbacks, wandb_logger, num_sanity_val_steps=0)

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
    main(
        config=FLAGS.config,
        config_path=config_flags.get_config_filename(FLAGS['config']),
    )
