"""Training script for the Neural Posterior Estimation embedding model."""

import os
import sys
import shutil
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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

from jgnn import datasets
from jgnn.models import GNNEmbedding, TransformerEmbedding
from jgnn.transforms import build_transformation


def setup_workdir(workdir: str) -> Path:
    """Set up the working directory for training.

    Args:
        workdir: Base working directory

    Returns:
        Path object for the working directory
    """
    run_dir = Path(workdir)
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
        init=config.get('init', 0),
        concat=True,
    )

    # Create dataloaders
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
    """Create the GNN embedding model.

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


def create_callbacks(config: ml_collections.ConfigDict, wandb_logger: WandbLogger) -> list:
    """Create PyTorch Lightning callbacks.

    Args:
        config: Configuration dictionary
        wandb_logger: WandB logger instance (used to get checkpoint directory)

    Returns:
        List of callback instances
    """
    callbacks = [
        EarlyStopping(
            monitor='val/loss',
            mode='min',
            patience=config.patience,
            verbose=True
        ),
        ModelCheckpoint(
            filename="epoch={epoch}-step={step}-loss={val/loss:.4f}",
            monitor='val/loss',
            mode='min',
            save_top_k=3,  # saves last 3 best checkpoints
            save_weights_only=False,
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(
            filename="last",
            save_weights_only=False,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    return callbacks

def main(config: ml_collections.ConfigDict, workdir: str = "./logging/"):
    """Train the GNN embedding model with wandb logging.

    Args:
        config: Configuration dictionary containing model and training parameters
        workdir: Working directory for logging and checkpoints
    """
    # Setup
    checkpoint_path = None
    resume_training = config.get('checkpoint') is not None

    print(f"[Setup] Resume training: {resume_training}")
    print(f"[Setup] Working directory: {workdir}")

    # Setup working directory
    run_dir = setup_workdir(workdir)
    print(f"[Setup] Run directory: {run_dir}")

    # Initialize wandb logger
    wandb_mode = 'disabled' if config.get('debug', False) else 'online'
    print(f"[WandB] Mode: {wandb_mode}")

    tags = config.get('tags', [])
    tags.append('embedding')
    wandb_logger = WandbLogger(
        project=config.get("wandb_project", "jgnn-npe"),
        name=config.get("name"),
        id=config.get("id", None),
        save_dir=str(run_dir),
        log_model="all",
        config=config.to_dict(),
        mode=wandb_mode,
        resume="allow",
        tags=list(set(tags))
    )

    # Prepare data
    print("[Data] Loading datasets...")
    train_loader, val_loader, norm_dict = prepare_data(config)
    print(f"[Data] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Build pre-transforms
    print("[Transforms] Building pre-transforms...")
    pre_transforms = build_transformation(
        norm_dict=norm_dict, **config.pre_transforms)

    # Create model
    print("[Model] Creating GNN embedding model...")
    model = create_model(config, pre_transforms, norm_dict)

    # Print trainable vs frozen parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[Model] Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[Model] Trainable parameters: {trainable_params:,}")
    print(f"[Model] Frozen parameters: {frozen_params:,}")

    # this watches all parameters and gradients
    wandb_logger.watch(model, log="all", log_freq=500, log_graph=True)

    # Get checkpoint path if resuming
    if resume_training:
        checkpoint_path = get_checkpoint_path(config, run_dir)
        print(f"[Checkpoint] Resuming from: {checkpoint_path}")
        print(f"[Checkpoint] Reset optimizer: {config.get('reset_optimizer', False)}")

    # Create callbacks
    callbacks = create_callbacks(config, wandb_logger)
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
        print(f"[Training] Starting fresh training")
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
