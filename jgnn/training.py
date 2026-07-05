"""Shared helpers for the Lightning training scripts (setup, callbacks, fit loop).

Used by scripts/train_embed.py and scripts/train_npe.py to avoid duplicating
the wandb/checkpoint/trainer boilerplate that is identical across both.
"""

import os
import socket
from pathlib import Path
from typing import Optional

import torch
import ml_collections
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)


def _wandb_server_reachable(timeout: float = 3.0) -> bool:
    """Check whether the WandB API is reachable, with a short timeout.

    Used to pick 'online' vs 'offline' mode up front, so that training
    doesn't stall behind wandb's own (much longer) connection retries on
    machines without internet access, e.g. HPC compute nodes.
    """
    try:
        socket.create_connection(("api.wandb.ai", 443), timeout=timeout).close()
        return True
    except OSError:
        return False


def create_wandb_logger(
    config: ml_collections.ConfigDict, tag: str
) -> WandbLogger:
    """Create a WandbLogger for the run, tagged with `tag`.

    Args:
        config: Configuration dictionary.
        tag: Tag identifying the training stage, e.g. 'npe' or 'embedding'.

    Config fields
    -------------
    wandb_mode : str, optional
        Force 'online', 'offline', or 'disabled'. If unset, the mode is
        chosen automatically: 'online' if the WandB API is reachable,
        otherwise 'offline' (metrics are logged locally and can be
        uploaded later with `wandb sync`).

    Returns:
        Configured WandbLogger instance and project directory path
        (where checkpoints and config snapshots are saved).
    """
    workdir = Path(config.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    if config.get('debug', False):
        wandb_mode = 'disabled'
    else:
        wandb_mode = config.get('wandb_mode', None)
        if wandb_mode is None:
            wandb_mode = 'online' if _wandb_server_reachable() else 'offline'
    print(f"[WandB] Mode: {wandb_mode}")

    tags = set(config.get('tags', [])) | {tag}
    logger = WandbLogger(
        project=config.get("wandb_project", "jgnn-npe"),
        name=config.get("name"),
        entity=config.get("entity", None),
        id=config.get("id", None),
        save_dir=str(workdir),
        log_model="all",
        config=config.to_dict(),
        mode=wandb_mode,
        resume="allow",
        tags=list(tags),
    )
    project_dir = workdir / logger.experiment.project / logger.experiment.id
    project_dir.mkdir(parents=True, exist_ok=True)

    return logger, project_dir


def get_checkpoint_path(
    config: ml_collections.ConfigDict, project_dir: Optional[Path] = None,
) -> Optional[str]:
    """Resolve the checkpoint path to resume training from, if any.

    Args:
        config: Configuration dictionary.
        project_dir: Project directory path.

    Returns:
        Resolved checkpoint path, or None if `config.checkpoint` is unset.
    """
    if config.get('checkpoint') is None:
        return None

    ckpt = config.checkpoint
    if os.path.isabs(ckpt):
        return ckpt

    if project_dir is None:
        raise ValueError(
            "If `config.checkpoint` is a relative path, `project_dir` must be"
            " provided to resolve the full path."
        )
    return str(project_dir / 'checkpoints' / ckpt)


def create_base_callbacks(config: ml_collections.ConfigDict) -> list:
    """Create the standard early-stopping / checkpointing / LR-monitor callbacks.

    Args:
        config: Configuration dictionary.

    Returns:
        List of callback instances shared by all training scripts.
    """
    return [
        EarlyStopping(
            monitor='val/loss',
            mode='min',
            patience=config.patience,
            verbose=True,
            # False: check right after validation logs val/loss (on
            # on_validation_end), not right after the training epoch ends.
            # With the default (True), resuming from a full checkpoint can
            # hit an epoch-end boundary before this session's validation
            # loop has run yet, and the check crashes with "metric not
            # available" even though nothing is actually wrong.
            check_on_train_epoch_end=False,
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


def report_param_counts(model) -> None:
    """Print total, trainable, and frozen parameter counts for a model.

    Args:
        model: PyTorch module to summarize.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total parameters: {total:,}")
    print(f"[Model] Trainable parameters: {trainable:,}")
    print(f"[Model] Frozen parameters: {total - trainable:,}")


def build_trainer(
    config: ml_collections.ConfigDict,
    project_dir: Path,
    callbacks: list,
    wandb_logger: WandbLogger,
    **trainer_kwargs,
) -> pl.Trainer:
    """Construct the PyTorch Lightning Trainer.

    Args:
        config: Configuration dictionary.
        project_dir: Project directory for the run.
        callbacks: List of Lightning callbacks.
        wandb_logger: WandB logger instance.
        **trainer_kwargs: Extra keyword arguments forwarded to `pl.Trainer`.

    Returns:
        Configured Trainer instance.
    """
    print(f"[Trainer] Max epochs: {config.num_epochs}, Max steps: {config.num_steps}")
    print(f"[Trainer] Accelerator: {config.accelerator}")

    return pl.Trainer(
        default_root_dir=str(project_dir),
        max_epochs=config.num_epochs,
        max_steps=config.num_steps,
        accelerator=config.accelerator,
        callbacks=callbacks,
        logger=wandb_logger,
        enable_progress_bar=config.get("enable_progress_bar", True),
        gradient_clip_val=config.get('gradient_clip_val', None),
        **trainer_kwargs,
    )


def fit(
    trainer: pl.Trainer,
    model,
    train_loader,
    val_loader,
    checkpoint_path: Optional[str],
    reset_optimizer: bool,
) -> None:
    """Run `trainer.fit`, handling checkpoint resume and weights-only reset.

    Args:
        trainer: PyTorch Lightning Trainer.
        model: Lightning module to train.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        checkpoint_path: Checkpoint to resume from, or None for fresh training.
        reset_optimizer: If True and `checkpoint_path` is set, load weights only
            and start the optimizer/scheduler state from scratch.
    """
    if checkpoint_path and reset_optimizer:
        print("[Training] Loading model weights with fresh optimizer state")
        checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False
        )
        model.load_state_dict(checkpoint['state_dict'])
        trainer.fit(model, train_loader, val_loader)
    elif checkpoint_path:
        print("[Training] Resuming training from full checkpoint")
        trainer.fit(
            model,
            train_loader,
            val_loader,
            ckpt_path=checkpoint_path,
            weights_only=False,
        )
    else:
        print("[Training] Starting fresh training")
        trainer.fit(model, train_loader, val_loader)

    print("[Training] Training complete!")
