"""Optuna hyperparameter optimization for Neural Posterior Estimation (NPE).

The search_space uses nested dict structure matching your config hierarchy.
Any dict with a 'type' key is treated as a tunable parameter.

Example:
    config.optuna.search_space = {
        'optimizer': {
            'lr': {'type': 'float', 'low': 1e-5, 'high': 1e-3, 'log': True},
            'weight_decay': {'type': 'float', 'low': 1e-6, 'high': 1e-2, 'log': True}
        },
        'model': {
            'flows': {
                'num_transforms': {'type': 'int', 'low': 4, 'high': 10},
                'hidden_size': {'type': 'categorical', 'choices': [64, 128, 256]},
                'num_hidden_layers': {'type': 'int', 'low': 1, 'high': 3}
            }
        }
    }

Special handling for hidden_size + num_hidden_layers:
    - If both 'hidden_size' and 'num_hidden_layers' are found together,
      they are automatically combined into a list: [hidden_size] * num_hidden_layers
"""

import os
import sys
import copy
from pathlib import Path
from typing import Any

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
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from absl import flags
from ml_collections import config_flags

from jgnn import datasets
from jgnn.models import NPE
from jgnn.transforms import build_transformation
from jgnn.callbacks.visualization import NPEVisualizationCallback
from train_npe import (
    setup_workdir,
    load_embedding_network,
    prepare_data,
    create_model,
)


def suggest_parameter(trial: optuna.Trial, param_name: str, param_config: dict) -> Any:
    """Suggest a parameter value based on its configuration.

    Args:
        trial: Optuna trial object
        param_name: Name of the parameter
        param_config: Dictionary with keys: 'type', and type-specific keys

    Supported types:
        - 'float': {'low': float, 'high': float, 'log': bool (optional)}
        - 'int': {'low': int, 'high': int, 'step': int (optional), 'log': bool (optional)}
        - 'categorical': {'choices': list}
        - 'loguniform': {'low': float, 'high': float} (deprecated but supported)

    Returns:
        Suggested parameter value
    """
    param_type = param_config['type']

    if param_type == 'float':
        log = param_config.get('log', False)
        return trial.suggest_float(param_name, param_config['low'], param_config['high'], log=log)

    elif param_type == 'loguniform':
        # Legacy support - maps to float with log=True
        return trial.suggest_float(param_name, param_config['low'], param_config['high'], log=True)

    elif param_type == 'int':
        log = param_config.get('log', False)
        step = param_config.get('step', 1)
        return trial.suggest_int(param_name, param_config['low'], param_config['high'], step=step, log=log)

    elif param_type == 'categorical':
        return trial.suggest_categorical(param_name, param_config['choices'])

    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


def is_parameter_config(obj: Any) -> bool:
    """Check if object is a parameter configuration (has 'type' key)."""
    return isinstance(obj, (dict, ml_collections.ConfigDict)) and 'type' in obj


def traverse_and_suggest(
    trial: optuna.Trial,
    search_space: dict,
    path_prefix: str = ''
) -> dict:
    """Recursively traverse nested search_space dict and suggest parameters.

    Args:
        trial: Optuna trial object
        search_space: Nested dict defining search space
        path_prefix: Current path prefix (for recursion)

    Returns:
        Dictionary mapping full parameter paths to suggested values
    """
    suggestions = {}

    for key, value in search_space.items():
        current_path = f"{path_prefix}.{key}" if path_prefix else key
        if is_parameter_config(value):
            # This is a parameter to optimize
            suggested_value = suggest_parameter(trial, current_path, value)
            suggestions[current_path] = suggested_value
        elif isinstance(value, (dict, ml_collections.ConfigDict)):
            # This is a nested dict, recurse deeper
            nested_suggestions = traverse_and_suggest(trial, value, current_path)
            suggestions.update(nested_suggestions)

    return suggestions


def update_config_from_trial(config: ml_collections.ConfigDict, trial: optuna.Trial) -> ml_collections.ConfigDict:
    """Update config with hyperparameters suggested by Optuna trial.

    The search_space can be a nested dict structure matching the config hierarchy.
    For example:
        search_space = {
            'optimizer': {
                'lr': {'type': 'float', 'low': 1e-5, 'high': 1e-3, 'log': True}
            },
            'model': {
                'flows': {
                    'num_transforms': {'type': 'int', 'low': 4, 'high': 10}
                }
            }
        }

    Args:
        config: Base configuration (will be copied, not modified)
        trial: Optuna trial object

    Returns:
        Updated config with trial suggestions
    """
    # Deep copy to avoid modifying original
    trial_config = copy.deepcopy(config)
    trial_config.unlock()

    # Check search space exists
    if not hasattr(trial_config.optuna, 'search_space'):
        raise ValueError("Config must have optuna.search_space defined")

    # Convert to regular dict if it's a ConfigDict
    search_space = dict(trial_config.optuna.search_space)

    # Recursively traverse search_space and get all parameter suggestions
    suggestions = traverse_and_suggest(trial, search_space)

    # Apply suggestions to config
    for param_path, suggested_value in suggestions.items():
        # Navigate nested config using dot notation
        parts = param_path.split('.')
        current = trial_config

        # Navigate to parent of target
        for part in parts[:-1]:
            if not hasattr(current, part):
                # Create intermediate ConfigDict if it doesn't exist
                setattr(current, part, ml_collections.ConfigDict())
            current = getattr(current, part)

        # Set the final value
        setattr(current, parts[-1], suggested_value)

    # Post-process: Handle parameterized lists (hidden_size + num_hidden_layers)
    # These get converted to hidden_sizes/hidden_features lists

    # Handle model.flows.hidden_features
    if ('model.flows.hidden_size' in suggestions and
        'model.flows.num_hidden_layers' in suggestions):
        hidden_size = trial_config.model.flows.hidden_size
        num_layers = trial_config.model.flows.num_hidden_layers
        trial_config.model.flows.hidden_features = [hidden_size] * num_layers
        print(f"[Optuna] Constructed flows.hidden_features: [{hidden_size}] * {num_layers} = {trial_config.model.flows.hidden_features}")
        # Remove the temporary params
        delattr(trial_config.model.flows, 'hidden_size')
        delattr(trial_config.model.flows, 'num_hidden_layers')

    # Handle model.embedding.mlp.hidden_sizes
    if ('model.embedding.mlp.hidden_size' in suggestions and
        'model.embedding.mlp.num_hidden_layers' in suggestions):
        hidden_size = trial_config.model.embedding.mlp.hidden_size
        num_layers = trial_config.model.embedding.mlp.num_hidden_layers
        trial_config.model.embedding.mlp.hidden_sizes = [hidden_size] * num_layers
        print(f"[Optuna] Constructed embedding.mlp.hidden_sizes: [{hidden_size}] * {num_layers} = {trial_config.model.embedding.mlp.hidden_sizes}")
        # Remove the temporary params
        delattr(trial_config.model.embedding.mlp, 'hidden_size')
        delattr(trial_config.model.embedding.mlp, 'num_hidden_layers')

    return trial_config


def create_callbacks(
    config: ml_collections.ConfigDict,
    trial: optuna.Trial,
    wandb_logger: WandbLogger
) -> list:
    """Create PyTorch Lightning callbacks for Optuna trial.

    Args:
        config: Configuration dictionary
        trial: Optuna trial (for pruning callback)
        wandb_logger: WandB logger instance

    Returns:
        List of callback instances
    """
    callbacks = [
        PyTorchLightningPruningCallback(trial, monitor="val/loss"),
        EarlyStopping(
            monitor='val/loss',
            mode='min',
            patience=config.patience,
            verbose=True
        ),
        ModelCheckpoint(
            filename=f"trial_{trial.number}_best",
            monitor='val/loss',
            mode='min',
            save_top_k=1,
            save_weights_only=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Optionally add visualization callback (may want to disable for optuna)
    if config.get('enable_visualization_callback', False):
        callbacks.append(
            NPEVisualizationCallback(
                plot_every_n_epochs=config.visualization.get('plot_every_n_epochs', 10),
                n_posterior_samples=config.visualization.get('n_posterior_samples', 1000),
                n_val_samples=config.visualization.get('n_val_samples', 100),
                plot_median_v_true=config.visualization.get('plot_median_v_true', True),
                plot_tarp=config.visualization.get('plot_tarp', False),
                plot_rank=config.visualization.get('plot_rank', False),
                use_default_mplstyle=config.visualization.get('use_default_mplstyle', True),
            )
        )
    return callbacks


def objective(trial: optuna.Trial, base_config: ml_collections.ConfigDict) -> float:
    """Optuna objective function for a single trial.

    Args:
        trial: Optuna trial object
        base_config: Base configuration (fixed parameters)

    Returns:
        Validation loss (metric to minimize)
    """
    # Update config with trial suggestions
    config = update_config_from_trial(base_config, trial)

    # Create trial-specific workdir
    trial_workdir = Path(config.workdir) / f"trial_{trial.number}"
    trial_workdir.mkdir(parents=True, exist_ok=True)

    # Setup wandb logger for this trial
    wandb_mode = 'disabled' if config.get('debug', False) else 'online'
    wandb_logger = WandbLogger(
        project=config.get("wandb_project", "jgnn-npe-optuna"),
        name=f"{config.get('name', 'optuna')}_trial_{trial.number}",
        save_dir=str(trial_workdir),
        log_model=False,  # Don't log all models to wandb during optuna
        config={
            **config.to_dict(),
            'trial_number': trial.number,
        },
        mode=wandb_mode,
        tags=['optuna'],
    )


    # Load embedding network if specified (to potentially extract norm_dict)
    embedding_norm_dict = None
    embedding_checkpoint = config.model.embedding.get('checkpoint', None)
    if embedding_checkpoint is not None:
        _, embedding_norm_dict = load_embedding_network(
            config,
            embedding_checkpoint,
            freeze=False
        )

    # Prepare data
    train_loader, val_loader, norm_dict = prepare_data(config, embedding_norm_dict)

    # Build pre-transforms
    pre_transforms = build_transformation(
        norm_dict=norm_dict, **config.pre_transforms)

    # Create model
    model = create_model(config, pre_transforms, norm_dict)

    # Log model info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Trial {trial.number}] Trainable parameters: {trainable_params:,}")

    # Create callbacks
    callbacks = create_callbacks(config, trial, wandb_logger)

    # Create trainer
    trainer = pl.Trainer(
        default_root_dir=str(trial_workdir),
        max_epochs=config.num_epochs,
        max_steps=config.num_steps,
        accelerator=config.accelerator,
        # callbacks=callbacks,
        logger=wandb_logger,
        enable_progress_bar=config.get("enable_progress_bar", False),
        gradient_clip_val=config.get('gradient_clip_val', None),
        enable_model_summary=False,  # Reduce logging
    )
    # weird bug: if pruning callback is in the list at init, it fails
    # need to add it after trainer is created
    trainer.callbacks.extend(callbacks)

    # Set random seed for this tria
    # Use base seed (if provided) + trial number to ensure different but reproducible seeds
    base_seed = config.get('seed_training', 42)
    trial_seed = base_seed + trial.number
    pl.seed_everything(trial_seed, workers=True)
    print(f"[Trial {trial.number}] Using random seed: {trial_seed}")

    # Train model
    try:
        trainer.fit(model, train_loader, val_loader)

        # Get best validation loss
        best_val_loss = trainer.callback_metrics.get('val/loss', float('inf'))

        # Clean up
        wandb.finish(quiet=True)

        return best_val_loss.item() if torch.is_tensor(best_val_loss) else best_val_loss

    except optuna.TrialPruned as e:
        # Clean up on pruned trial
        wandb.finish(quiet=True)
        raise e

    except Exception as e:
        print(f"[Trial {trial.number}] Failed with error: {e}")
        wandb.finish(quiet=True)
        raise e


def main(config: ml_collections.ConfigDict, workdir: str = "./logging/optuna/"):
    """Run Optuna hyperparameter optimization.

    Args:
        config: Configuration dictionary with optuna settings
        workdir: Working directory for study storage and logs
    """
    # Setup working directory
    workdir_path = Path(workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)

    print(f"[Optuna] Starting hyperparameter optimization")
    print(f"[Optuna] Working directory: {workdir}")
    print(f"[Optuna] Study name: {config.optuna.study_name}")

    # Setup storage
    storage = config.optuna.get('storage', None)
    if storage is None:
        storage = f"sqlite:///{workdir_path / 'optuna_study.db'}"
    print(f"[Optuna] Storage: {storage}")

    # Create or load study
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=config.optuna.get('n_startup_trials', 10),
        multivariate=True,
        seed=config.get('seed_optuna', 42),
    )

    # Optional pruner
    use_pruner = config.optuna.get('use_pruner', True)
    pruner = None
    if use_pruner:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=config.optuna.get('n_startup_trials', 10),
            n_warmup_steps=config.optuna.get('n_warmup_steps', 5),
            interval_steps=config.optuna.get('interval_steps', 1),
        )

    study = optuna.create_study(
        study_name=config.optuna.study_name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=config.optuna.get('load_if_exists', True),
    )

    # Print search space
    print(f"\n[Optuna] Search space:")
    # for param_name, param_config in config.optuna.search_space.items():
        # print(f"  {param_name}: {param_config}")

    # Run optimization
    n_trials = config.optuna.get('n_trials', 100)
    timeout = config.optuna.get('timeout', None)  # in seconds

    print(f"\n[Optuna] Running {n_trials} trials")
    if timeout:
        print(f"[Optuna] Timeout: {timeout} seconds")

    study.optimize(
        lambda trial: objective(trial, config),
        n_trials=n_trials,
        timeout=timeout,
        catch=(Exception,),  # Continue on trial failures
        show_progress_bar=True,
    )

    # Print results
    print("\n" + "="*80)
    print("[Optuna] Optimization complete!")
    print("="*80)

    print(f"\nNumber of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")

    if study.best_trial:
        print(f"\nBest trial:")
        print(f"  Value (val/loss): {study.best_trial.value:.6f}")
        print(f"  Params:")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")

    # Save best hyperparameters to file
    best_params_file = workdir_path / "best_hyperparameters.txt"
    with open(best_params_file, 'w') as f:
        f.write(f"Best trial number: {study.best_trial.number}\n")
        f.write(f"Best validation loss: {study.best_trial.value:.6f}\n\n")
        f.write("Best hyperparameters:\n")
        for key, value in study.best_trial.params.items():
            f.write(f"  {key}: {value}\n")

    print(f"\n[Optuna] Best hyperparameters saved to: {best_params_file}")

    # Optionally save visualization
    try:
        import plotly

        # Optimization history
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(str(workdir_path / "optimization_history.html"))

        # Parameter importances
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(str(workdir_path / "param_importances.html"))

        # Parallel coordinate plot
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_html(str(workdir_path / "parallel_coordinate.html"))

        print(f"[Optuna] Visualizations saved to: {workdir_path}")

    except ImportError:
        print("[Optuna] Install plotly for visualizations: pip install plotly")


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config",
        None,
        "File path to the Optuna hyperparameter configuration.",
        lock_config=True,
    )
    FLAGS(sys.argv)
    main(config=FLAGS.config, workdir=FLAGS.config.workdir)
