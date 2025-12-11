"""Training script for Neural Posterior Estimation (NPE)."""

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
from jgnn.models import NRE, GNNEmbedding, TransformerEmbedding
from jgnn.transforms import build_transformation
from jgnn.callbacks.visualization import NPEVisualizationCallback


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


def load_embedding_network(
    config: ml_collections.ConfigDict, checkpoint_path: str, freeze: bool = False):
    """Load a pre-trained embedding network from checkpoint.

    Args:
        config: Configuration dictionary
        checkpoint_path: Path to checkpoint
        freeze: If True, freeze all parameters of the embedding network

    Returns:
        Tuple of (embedding_network, norm_dict)
        - embedding_network: The loaded GNNEmbedding model
        - norm_dict: The normalization dictionary from the checkpoint (if available)
    """
    print(f"[Embedding] Loading pre-trained embedding network from: {checkpoint_path}")

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load the embedding model
    if config.model.embedding.type == 'transformer':
        print(f"[Embedding] Detected TransformerEmbedding model type")
        embedding_nn = TransformerEmbedding.load_from_checkpoint(checkpoint_path)
    elif config.model.embedding.type == 'gnn':
        print(f"[Embedding] Detected GNNEmbedding model type")
        embedding_nn = GNNEmbedding.load_from_checkpoint(checkpoint_path)
    else:
        raise ValueError(
            f"Unsupported embedding model type: {config.model.embedding.type}")

    # Extract norm_dict if available in the hyperparameters
    norm_dict = None
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        if 'norm_dict' in hparams:
            norm_dict = hparams['norm_dict']
            print(f"[Embedding] Loaded norm_dict from embedding checkpoint")

    # Freeze parameters if requested
    if freeze:
        for param in embedding_nn.parameters():
            param.requires_grad = False
        embedding_nn.eval()
        print(f"[Embedding] Froze all parameters in embedding network")

    print(f"[Embedding] Embedding network loaded successfully")
    print(f"[Embedding] Output size: {embedding_nn.output_size}")

    return embedding_nn, norm_dict


def prepare_data(config: ml_collections.ConfigDict, embedding_norm_dict=None):
    """Load and prepare datasets with transformations.

    Args:
        config: Configuration dictionary
        embedding_norm_dict: Optional norm_dict from embedding checkpoint.
                           If provided, this will be used instead of computing from data.

    Returns:
        Tuple of (train_loader, val_loader, norm_dict)
    """
    # Load datasets
    node_feats, graph_feats = datasets.read_datasets(
        config.data_root,
        config.data_name,
        config.num_datasets,
        init=config.get('init', 0),
        concat=True
    )

    # Create dataloaders
    # If we have a norm_dict from embedding, we can reuse it or compute new one
    if embedding_norm_dict is not None and config.get('reuse_embedding_norm_dict', True):
        print("[Data] Reusing normalization dict from embedding checkpoint")
        norm_dict = embedding_norm_dict
    else:
        norm_dict = None

    # Create dataloaders with existing norm_dict
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
        norm_dict=norm_dict
    )

    return train_loader, val_loader, norm_dict


def create_embedding_network(config: ml_collections.ConfigDict):
    """Create a new embedding network.

    Args:
        config: Configuration dictionary

    Returns:
        Embedding network instance
    """
    model_type = config.model.embedding.get('type', 'gnn')
    if model_type == 'gnn':
        print("[Model] Creating GNN Embedding model...")
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
    elif model_type == 'transformer':
        print("[Model] Creating Transformer Embedding model...")
        return TransformerEmbedding(
            input_size=config.model.input_size,
            transformer_args=config.model.embedding.transformer,
            loss_type=config.model.embedding.get('loss_type', 'mse'),
            loss_args=config.model.embedding.get('loss_args', None),
            mlp_args=config.model.embedding.get('mlp', None),
            optimizer_args=None,
            scheduler_args=None,
            pre_transforms=None,
        )
    else:
        raise ValueError(f"Unsupported embedding model type: {config.model.type}")

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
    # Check if we should load a pre-trained embedding network
    embedding_checkpoint = config.model.embedding.get('checkpoint', None)
    freeze_embedding = config.model.embedding.get('freeze', False)

    if embedding_checkpoint is not None:
        # Load pre-trained embedding network
        embedding_nn, _ = load_embedding_network(
            config,
            embedding_checkpoint,
            freeze=freeze_embedding
        )
    else:
        # Create new embedding network
        print("[Model] Creating new embedding network...")
        embedding_nn = create_embedding_network(config)

    # Create NPE model
    # Note: pre_transforms goes to NPE, not embedding_nn
    print("[Model] Creating NPE model...")

    # Check if we should initialize flows from embedding
    init_flows_from_embedding = config.model.get('init_flows_from_embedding', False)

    model = NRE(
        input_size=config.model.input_size,
        output_size=config.model.output_size,
        classifier_hidden=config.mode.classifier_hidden,
        embedding_nn=embedding_nn,
        optimizer_args=config.optimizer,
        scheduler_args=config.scheduler,
        norm_dict=norm_dict,
        pre_transforms=pre_transforms,
    )

    return model


def create_callbacks(config: ml_collections.ConfigDict, wandb_logger: WandbLogger) -> list:
    """Create PyTorch Lightning callbacks.

    Args:
        config: Configuration dictionary
        wandb_logger: WandB logger instance (used to get checkpoint directory)

    Returns:
        List of callback instances
    """
    # default callbacks
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

    if config.get('enable_visualization_callback', True):
        print("[Callbacks] Visualization callback enabled, but not implemented for NRE yet.")

    return callbacks


def main(config: ml_collections.ConfigDict, workdir: str = "./logging/"):
    """Train the NRE model with wandb logging.

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
    tags.append('nre')
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

    # Load embedding network if specified (to potentially extract norm_dict)
    embedding_norm_dict = None
    embedding_checkpoint = config.model.get('embedding_checkpoint', None)
    if embedding_checkpoint is not None:
        _, embedding_norm_dict = load_embedding_network(
            embedding_checkpoint,
            freeze=False  # Don't freeze yet, just extracting norm_dict
        )

    # Prepare data
    print("[Data] Loading datasets...")
    train_loader, val_loader, norm_dict = prepare_data(config, embedding_norm_dict)
    print(f"[Data] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Build pre-transforms (will be passed to NPE, not embedding_nn)
    print("[Transforms] Building pre-transforms...")
    pre_transforms = build_transformation(
        norm_dict=norm_dict, **config.pre_transforms)

    # Create model
    print("[Model] Creating NPE model...")
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

    # Create callbacks (after wandb_logger is initialized)
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
