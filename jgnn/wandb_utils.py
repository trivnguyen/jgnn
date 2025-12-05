"""Utilities for working with WandB checkpoints and runs."""

import os
from pathlib import Path
from typing import Optional, Union, Literal
import wandb


def find_local_checkpoint(
    run_dir: Union[str, Path],
    checkpoint_type: Literal['best', 'last'] = 'best',
    monitor_metric: str = 'val/loss'
) -> Optional[str]:
    """Find a checkpoint in the local WandB directory structure.

    This function searches for checkpoints in the standard WandB/PyTorch Lightning
    directory structure: {run_dir}/{run_name}/{version}/checkpoints/

    Args:
        run_dir: Root directory where WandB logs are saved
        checkpoint_type: Type of checkpoint to find ('best' or 'last')
        monitor_metric: Metric name that was monitored (used to identify best checkpoint)

    Returns:
        Path to the checkpoint file, or None if not found

    Examples:
        >>> # Find best checkpoint by validation loss
        >>> ckpt = find_local_checkpoint('./logging/my_run', 'best')

        >>> # Find last checkpoint
        >>> ckpt = find_local_checkpoint('./logging/my_run', 'last')
    """
    run_dir = Path(run_dir)

    if not run_dir.exists():
        print(f"[WandB Utils] Run directory does not exist: {run_dir}")
        return None

    # Search for checkpoint directories
    checkpoint_dirs = list(run_dir.rglob('checkpoints'))

    if not checkpoint_dirs:
        print(f"[WandB Utils] No checkpoint directories found in {run_dir}")
        return None

    # Use the most recent checkpoint directory
    checkpoint_dir = max(checkpoint_dirs, key=lambda p: p.stat().st_mtime)
    print(f"[WandB Utils] Found checkpoint directory: {checkpoint_dir}")

    if checkpoint_type == 'last':
        # Look for last.ckpt
        last_ckpt = checkpoint_dir / 'last.ckpt'
        if last_ckpt.exists():
            print(f"[WandB Utils] Found last checkpoint: {last_ckpt}")
            return str(last_ckpt)
        else:
            print(f"[WandB Utils] Last checkpoint not found in {checkpoint_dir}")
            return None

    elif checkpoint_type == 'best':
        # Find the best checkpoint (lowest loss or highest metric)
        # PyTorch Lightning saves checkpoints as: epoch=X-step=Y.ckpt
        ckpt_files = list(checkpoint_dir.glob('epoch=*.ckpt'))

        if not ckpt_files:
            print(f"[WandB Utils] No epoch checkpoints found in {checkpoint_dir}")
            return None

        # Return the most recent checkpoint (highest epoch number)
        # PyTorch Lightning's ModelCheckpoint with save_top_k keeps only the best ones
        best_ckpt = max(ckpt_files, key=lambda p: p.stat().st_mtime)
        print(f"[WandB Utils] Found best checkpoint: {best_ckpt}")
        return str(best_ckpt)

    else:
        raise ValueError(f"Invalid checkpoint_type: {checkpoint_type}. Must be 'best' or 'last'")


def find_checkpoint_from_wandb_run(
    project: str,
    run_name: str,
    checkpoint_type: Literal['best', 'last'] = 'best',
    entity: Optional[str] = None,
    download_dir: str = './wandb_downloads'
) -> Optional[str]:
    """Download and return path to checkpoint from a WandB run.

    This function uses the WandB API to download checkpoints from WandB cloud.
    Requires wandb to be logged in (run `wandb login` first).

    Args:
        project: WandB project name
        run_name: WandB run name or run ID
        checkpoint_type: Type of checkpoint to download ('best' or 'last')
        entity: WandB entity (username or team). If None, uses default entity.
        download_dir: Directory to download checkpoint to

    Returns:
        Path to the downloaded checkpoint file, or None if not found

    Examples:
        >>> # Download best checkpoint from a specific run
        >>> ckpt = find_checkpoint_from_wandb_run(
        ...     project='jgnn-npe',
        ...     run_name='test_run',
        ...     checkpoint_type='best'
        ... )

        >>> # Download last checkpoint with specific entity
        >>> ckpt = find_checkpoint_from_wandb_run(
        ...     project='jgnn-npe',
        ...     run_name='my_run_id',
        ...     checkpoint_type='last',
        ...     entity='my_team'
        ... )
    """
    api = wandb.Api()

    # Construct the run path
    if entity is None:
        entity = api.default_entity
    run_path = f"{entity}/{project}/{run_name}"

    try:
        run = api.run(run_path)
    except Exception as e:
        print(f"[WandB Utils] Failed to fetch run {run_path}: {e}")
        return None

    # Get checkpoint files
    checkpoint_files = [f for f in run.files() if f.name.endswith('.ckpt')]

    if not checkpoint_files:
        print(f"[WandB Utils] No checkpoint files found in run {run_path}")
        return None

    # Filter by checkpoint type
    if checkpoint_type == 'last':
        target_file = next((f for f in checkpoint_files if 'last' in f.name), None)
    elif checkpoint_type == 'best':
        # Get epoch checkpoints (exclude 'last.ckpt')
        epoch_files = [f for f in checkpoint_files if 'epoch=' in f.name]
        if epoch_files:
            # Take the most recently uploaded one
            target_file = max(epoch_files, key=lambda f: f.updated_at)
        else:
            target_file = None
    else:
        raise ValueError(f"Invalid checkpoint_type: {checkpoint_type}")

    if target_file is None:
        print(f"[WandB Utils] No {checkpoint_type} checkpoint found in run {run_path}")
        return None

    # Download the checkpoint
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = download_dir / target_file.name
    print(f"[WandB Utils] Downloading checkpoint: {target_file.name}")
    target_file.download(root=str(download_dir), replace=True)
    print(f"[WandB Utils] Downloaded to: {checkpoint_path}")

    return str(checkpoint_path)


def get_run_config(
    project: str,
    run_name: str,
    entity: Optional[str] = None
) -> Optional[dict]:
    """Fetch the config from a WandB run.

    Args:
        project: WandB project name
        run_name: WandB run name or run ID
        entity: WandB entity (username or team)

    Returns:
        Config dictionary from the run, or None if not found
    """
    api = wandb.Api()

    if entity is None:
        entity = api.default_entity
    run_path = f"{entity}/{project}/{run_name}"

    try:
        run = api.run(run_path)
        return dict(run.config)
    except Exception as e:
        print(f"[WandB Utils] Failed to fetch config from {run_path}: {e}")
        return None


def list_runs(
    project: str,
    entity: Optional[str] = None,
    filters: Optional[dict] = None,
    limit: int = 10
) -> list:
    """List runs in a WandB project.

    Args:
        project: WandB project name
        entity: WandB entity (username or team)
        filters: Dictionary of filters to apply (e.g., {"state": "finished"})
        limit: Maximum number of runs to return

    Returns:
        List of run objects

    Examples:
        >>> # List last 10 finished runs
        >>> runs = list_runs('jgnn-npe', filters={"state": "finished"})
        >>> for run in runs:
        ...     print(f"{run.name}: {run.summary.get('val/loss')}")
    """
    api = wandb.Api()

    if entity is None:
        entity = api.default_entity

    runs = api.runs(f"{entity}/{project}", filters=filters)

    return list(runs)[:limit]
