"""Training script for the Neural Posterior Estimation embedding model."""

import os
import sys
import shutil
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import yaml
import wandb
import ml_collections
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
import torch
from absl import flags
from ml_collections import config_flags

import datasets
from jgnn.models.gnn_embedding import GNNEmbedding
from jgnn.transforms import build_transformation


def setup_workdir(workdir: str, name: str, reset: bool, resume: bool) -> Path:
    """Set up the working directory for training.

    Args:
        workdir: Base working directory
        name: Name of the training run
        reset:  If True, reset training run and overwrite local directory
        resume: Whether resuming from checkpoint

    Returns:
        Path object for the working directory
    """
    run_dir = Path(workdir) / name

    if reset and resume:
        raise ValueError("Cannot reset and resume training at the same time.")

    if run_dir.exists():
        if reset:
            shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True)
        else:
            raise ValueError(
                f"Directory {run_dir} already exists. Set reset=True to overwrite "
                "or provide a checkpoint to resume training."
            )
    else:
        run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def get_checkpoint_path(config: ml_collections.ConfigDict, workdir: Path) -> str | None:
    """Resolve checkpoint path from config.

    Args:
        config: Configuration dictionary
        workdir: Working directory path

    Returns:
        Resolved checkpoint path or None
    """
    if config.get('checkpoint') is None:
        return None

    ckpt = config.checkpoint
    if os.path.isabs(ckpt):
        return ckpt

    return str(workdir / 'lightning_logs' / 'checkpoints' / ckpt)


def prepare_data(config: ml_collections.ConfigDict):
    """Load and prepare datasets with transformations.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (train_loader, val_loader, pre_transforms)
    """
    # Load datasets
    node_feats, graph_feats = datasets.read_datasets(
        config.data_root,
        config.data_name,
        config.num_datasets,
        concat=True
    )

    # Create dataloaders
    train_loader, val_loader, norm_dict = datasets.prepare_dataloaders(
        node_feats,
        graph_feats,
        config.labels,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        train_frac=config.train_frac,
        num_workers=config.num_workers,
        seed=config.seed_data,
    )

    # Build pre-transforms if specified
    pre_transforms = build_transformation(**config.pre_transforms)

    return train_loader, val_loader, pre_transforms


def create_model(config: ml_collections.ConfigDict, pre_transforms) -> GNNEmbedding:
    """Create the GNN embedding model.

    Args:
        config: Configuration dictionary
        pre_transforms: Pre-transformation pipeline

    Returns:
        GNNEmbedding model instance
    """
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
    )


def create_callbacks(config: ml_collections.ConfigDict) -> list:
    """Create PyTorch Lightning callbacks.

    Args:
        config: Configuration dictionary

    Returns:
        List of callback instances
    """
    return [
        EarlyStopping(
            monitor='val/loss',
            mode='min',
            patience=config.patience,
            verbose=True
        ),
        ModelCheckpoint(
            filename="{epoch}-{step}",
            monitor='val/loss',
            mode='min',
            save_top_k=config.save_top_k,
            save_weights_only=False
        ),
        ModelCheckpoint(
            filename="last",
            save_top_k=1,
            save_weights_only=False,
            save_last=True
        ),
        LearningRateMonitor(logging_interval="step"),
    ]


def main(config: ml_collections.ConfigDict, workdir: str = "./logging/"):
    """Train the GNN embedding model with wandb logging.

    Args:
        config: Configuration dictionary containing model and training parameters
        workdir: Working directory for logging and checkpoints
    """
    # Setup
    name = config.get("name", "embedding_training")
    checkpoint_path = None
    resume_training = config.get('checkpoint') is not None

    print(f"[Setup] Training run: {name}")
    print(f"[Setup] Resume training: {resume_training}")
    print(f"[Setup] Working directory: {workdir}")

    # Setup working directory
    run_dir = setup_workdir(
        workdir,
        name,
        config.get('reset', False),
        resume_training
    )
    print(f"[Setup] Run directory: {run_dir}")

    # Save config
    config_dict = config.to_dict()
    config_path = run_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f)
    print(f"[Setup] Config saved to: {config_path}")

    # Initialize wandb logger
    wandb_mode = 'disabled' if config.get('debug', False) else 'online'
    print(f"[WandB] Mode: {wandb_mode}")

    wandb_logger = WandbLogger(
        project=config.get("wandb_project", "jgnn"),
        name=name,
        save_dir=str(run_dir),
        log_model=config.get("log_model", "all"),
        config=config_dict,
        mode=wandb_mode,
    )

    # Prepare data
    print("[Data] Loading datasets...")
    train_loader, val_loader, pre_transforms = prepare_data(config)
    print(f"[Data] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    print("[Model] Creating GNN embedding model...")
    model = create_model(config, pre_transforms)
    print(f"[Model] Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Get checkpoint path if resuming
    if resume_training:
        checkpoint_path = get_checkpoint_path(config, run_dir)
        print(f"[Checkpoint] Resuming from: {checkpoint_path}")
        print(f"[Checkpoint] Reset optimizer: {config.get('reset_optimizer', False)}")

    # Create callbacks
    callbacks = create_callbacks(config)
    print(f"[Callbacks] Created {len(callbacks)} callbacks")

    # Create trainer
    print(f"[Trainer] Max epochs: {config.num_epochs}, Max steps: {config.num_steps}")
    print(f"[Trainer] Accelerator: {config.accelerator}")

    trainer = pl.Trainer(
        default_root_dir=str(run_dir),
        max_epochs=config.num_epochs,
        max_steps=config.num_steps,
        accelerator=config.accelerator,
        callbacks=callbacks,
        logger=wandb_logger,
        enable_progress_bar=config.get("enable_progress_bar", True),
        gradient_clip_val=config.get('gradient_clip_val', None),
    )

    # Set random seed for training
    pl.seed_everything(config.seed_training, workers=True)
    print(f"[Seed] Training seed set to: {config.seed_training}")

    # Train model
    if checkpoint_path and config.get('reset_optimizer', False):
        # Load weights only, reset optimizer state
        print("[Training] Loading model weights with fresh optimizer state")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        trainer.fit(model, train_loader, val_loader)
    elif checkpoint_path:
        # Full checkpoint resume
        print("[Training] Resuming training from full checkpoint")
        trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)
    else:
        # Fresh training
        print(f"[Training] Starting fresh training: {name}")
        trainer.fit(model, train_loader, val_loader)

    print("[Training] Training complete!")

    # Finalize wandb
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
