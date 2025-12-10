"""Training script for Sequential Neural Posterior Estimation (SNPE).

This script implements multi-round SNPE with:
- Proper book-keeping to resume from any round
- Independent saving of each round
- Visualization of posterior at the end of each round
- WandB logging integration
"""

import os
import sys
import shutil
import pickle
import json
from pathlib import Path
from typing import Optional, Dict, Tuple

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import wandb
import ml_collections
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
import torch
from torch_geometric.data import Data, Batch
from absl import flags
from ml_collections import config_flags

from jgnn import datasets
from jgnn.models import SequentialNPE, GNNEmbedding, TransformerEmbedding
from jgnn.transforms import build_transformation
from jgnn.callbacks.visualization import NPEVisualizationCallback
from jgnn.priors import BoxUniform
from jgnn.sims import (
    run_simulation_batch,
    preprocess_simulations,
    samples_to_simulation_params
)
from jgnn.plotting import plot_corner_with_wandb, plot_posterior_statistics, close_all_figures


# ==================== Book-keeping Functions ====================

def save_round_state(
    round_num: int,
    round_dir: Path,
    model: SequentialNPE,
    norm_dict: Dict,
    config: ml_collections.ConfigDict,
    best_checkpoint_path: str
) -> None:
    """Save complete state for a round to enable resumption.

    Args:
        round_num: Current round number
        round_dir: Directory for this round
        model: Trained model
        norm_dict: Normalization dictionary
        config: Configuration
        best_checkpoint_path: Path to best checkpoint
    """
    state_file = round_dir / "round_state.pkl"

    state = {
        'round_num': round_num,
        'norm_dict': norm_dict,
        'best_checkpoint': best_checkpoint_path,
        'config': config.to_dict(),
    }

    with open(state_file, 'wb') as f:
        pickle.dump(state, f)

    print(f"[Book-keeping] Saved round {round_num} state to: {state_file}")


def load_round_state(round_dir: Path) -> Dict:
    """Load saved state for a round.

    Args:
        round_dir: Directory for the round

    Returns:
        Dictionary containing round state
    """
    state_file = round_dir / "round_state.pkl"

    if not state_file.exists():
        raise FileNotFoundError(f"Round state file not found: {state_file}")

    with open(state_file, 'rb') as f:
        state = pickle.load(f)

    print(f"[Book-keeping] Loaded round {state['round_num']} state from: {state_file}")
    return state


def find_latest_round(base_dir: Path) -> Optional[int]:
    """Find the latest completed round.

    Args:
        base_dir: Base directory containing round directories

    Returns:
        Latest round number or None if no rounds found
    """
    if not base_dir.exists():
        return None

    round_dirs = sorted([d for d in base_dir.glob("round_*") if d.is_dir()])

    for round_dir in reversed(round_dirs):
        state_file = round_dir / "round_state.pkl"
        if state_file.exists():
            round_num = int(round_dir.name.split('_')[1])
            print(f"[Book-keeping] Found completed round: {round_num}")
            return round_num

    return None


def setup_round_directory(base_dir: Path, round_num: int) -> Path:
    """Set up directory structure for a round.

    Args:
        base_dir: Base directory for all rounds
        round_num: Current round number

    Returns:
        Path to round directory
    """
    round_dir = base_dir / f"round_{round_num}"
    round_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (round_dir / "checkpoints").mkdir(exist_ok=True)
    (round_dir / "plots").mkdir(exist_ok=True)
    (round_dir / "logs").mkdir(exist_ok=True)

    return round_dir


# ==================== Data Functions ====================

def read_observation(
    config: ml_collections.ConfigDict,
    norm_dict: Optional[Dict] = None
) -> Batch:
    """Read observation data from CSV file.

    Args:
        config: Configuration dictionary
        norm_dict: Optional normalization dictionary

    Returns:
        Batch containing single observation
    """
    df = pd.read_csv(config.observation.path)

    # Apply membership probability cut
    mem_prob = df['mem_prob'].values.astype('float32')
    mask = mem_prob >= config.observation.get('prob_threshold', 0.8)
    if not mask.any():
        raise ValueError("No stars pass the membership probability threshold.")
    df = df[mask]

    # Extract relevant columns
    vr_sys = df['vr_sys'].values.astype('float32')
    vr = df['vr'].values.astype('float32')
    vr_err = df['vr_err'].values.astype('float32')
    ra = df['RA'].values.astype('float32')
    dec = df['DEC'].values.astype('float32')
    R_kin = df['R_kin'].values.astype('float32')
    log_R_kin = np.log10(R_kin + 1e-8).astype('float32')

    vr = vr - vr_sys  # Correct for systemic velocity

    print(f"[Observation] Loaded {len(df)} stars from {config.observation.path}")
    print(f"[Observation]   Velocity range: {vr.min():.2f} to {vr.max():.2f} km/s")
    print(f"[Observation]   Radius range: {R_kin.min():.2f} to {R_kin.max():.2f} kpc")

    # Handle conditioning labels
    if config.get('cond_labels') is not None:
        cond = []
        for label in config.cond_labels:
            cond.append(config.observation.meta['stellar_log_r_star'])
        cond = np.array(cond, dtype='float32')

        if norm_dict is not None:
            cond = (cond - norm_dict['cond_loc']) / norm_dict['cond_scale']
    else:
        cond = None

    # Create graph object (single observation)
    graph = Data(
        x=torch.tensor([log_R_kin, vr, vr_err]).T,
        pos=torch.tensor([ra, dec]).T,
        cond=torch.tensor(cond).unsqueeze(0) if cond is not None else None,
    )

    batch = Batch.from_data_list([graph])
    return batch


def sample_num_stars(config: ml_collections.ConfigDict, num_samples: int) -> np.ndarray:
    """Sample number of stars per galaxy.

    Args:
        config: Configuration dictionary
        num_samples: Number of samples needed

    Returns:
        Array of star counts
    """
    if config.simulation.num_stars_dist == 'poisson':
        return np.random.poisson(
            config.simulation.num_stars_mean,
            size=num_samples
        )
    elif config.simulation.num_stars_dist == 'uniform':
        return np.random.randint(
            config.simulation.num_stars_min,
            config.simulation.num_stars_max,
            size=num_samples
        )
    elif config.simulation.num_stars_dist == 'delta':
        return np.full(num_samples, config.simulation.num_stars_value)
    else:
        raise ValueError(f"Unknown num_stars_dist: {config.simulation.num_stars_dist}")


# ==================== Model Functions ====================

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
        )
    elif model_type == 'transformer':
        print("[Model] Creating Transformer Embedding model...")
        return TransformerEmbedding(
            input_size=config.model.input_size,
            transformer_args=config.model.embedding.transformer,
            loss_type=config.model.embedding.get('loss_type', 'mse'),
            loss_args=config.model.embedding.get('loss_args', None),
            mlp_args=config.model.embedding.get('mlp', None),
        )
    else:
        raise ValueError(f"Unsupported embedding model type: {model_type}")


def create_model(
    config: ml_collections.ConfigDict,
    pre_transforms,
    norm_dict: Dict,
    proposal=None,
    prior=None,
    current_round: int = 0
) -> SequentialNPE:
    """Create the SNPE model.

    Args:
        config: Configuration dictionary
        pre_transforms: Pre-transformation pipeline
        norm_dict: Normalization dictionary
        proposal: Proposal distribution from previous round
        prior: Prior distribution
        current_round: Current round number

    Returns:
        SequentialNPE model instance
    """
    print(f"[Model] Creating new embedding network for round {current_round}...")
    embedding_nn = create_embedding_network(config)

    print(f"[Model] Creating SequentialNPE model for round {current_round}...")
    if proposal is not None:
        print(f"[Model] Using posterior from round {current_round-1} as proposal")

    # Get num_atoms from config
    num_atoms = config.model.get('num_atoms', 10)
    print(f"[Model] Number of atoms: {num_atoms}")

    model = SequentialNPE(
        input_size=config.model.input_size,
        output_size=config.model.output_size,
        flows_args=config.model.flows,
        embedding_nn=embedding_nn,
        optimizer_args=config.optimizer,
        scheduler_args=config.scheduler,
        norm_dict=norm_dict,
        pre_transforms=pre_transforms,
        prior=prior,
        proposal=proposal,
        current_round=current_round,
        num_atoms=num_atoms
    )

    return model


def create_callbacks(config: ml_collections.ConfigDict) -> list:
    """Create PyTorch Lightning callbacks.

    Args:
        config: Configuration dictionary

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
            save_top_k=config.save_top_k,
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


# ==================== Visualization Functions ====================

def visualize_round_posterior(
    model: SequentialNPE,
    posterior,
    config: ml_collections.ConfigDict,
    round_num: int,
    round_dir: Path,
    norm_dict: Dict,
    wandb_logger
) -> None:
    """Create and log posterior visualization at end of round.

    Args:
        model: Trained model
        posterior: Posterior distribution
        config: Configuration dictionary
        round_num: Current round number
        round_dir: Directory for this round
        norm_dict: Normalization dictionary
        wandb_logger: WandB logger
    """
    if not config.post_round_visualization.enabled:
        print(f"[Visualization] Skipping round {round_num} visualization (disabled)")
        return

    print(f"\n[Visualization] Creating posterior plots for round {round_num}...")

    # Skip if round 0 and no observation loaded yet
    if round_num == 0 and not config.use_observation:
        print(f"[Visualization] Skipping round 0 (no observation)")
        return

    # Read observation
    obs_batch = read_observation(config, norm_dict=norm_dict)
    pre_transforms_obs = build_transformation(
        apply_projection=config.pre_transforms.apply_projection,
        apply_selection=False,
        apply_graph=False,
        apply_uncertainty=False,
        use_log_features=config.pre_transforms.use_log_features,
        projection_args=config.pre_transforms.projection_args,
        norm_dict=norm_dict
    )
    obs_batch = pre_transforms_obs(obs_batch)

    # Sample from posterior
    n_samples = config.post_round_visualization.n_samples
    print(f"[Visualization] Sampling {n_samples} from posterior...")

    model.eval()
    with torch.no_grad():
        samples = posterior.sample(
            n_samples,
            x_obs=obs_batch,
            norm_dict=norm_dict
        )

    # Print statistics
    plot_posterior_statistics(
        samples=samples,
        labels=config.labels,
        title=f"Round {round_num} Posterior Statistics",
        save_path=str(round_dir / "plots" / "statistics.txt") if config.post_round_visualization.save_plots else None
    )

    # Create corner plot
    save_dir = round_dir / "plots" if config.post_round_visualization.save_plots else None

    fig = plot_corner_with_wandb(
        samples=samples,
        labels=config.labels,
        round_num=round_num,
        wandb_logger=wandb_logger if config.post_round_visualization.log_to_wandb else None,
        save_dir=save_dir
    )

    # Clean up
    close_all_figures()
    print(f"[Visualization] Round {round_num} visualization complete\n")


# ==================== Main Training Loop ====================

def train_round(
    config: ml_collections.ConfigDict,
    round_num: int,
    round_dir: Path,
    prior: BoxUniform,
    model: Optional[SequentialNPE],
    proposal,
    norm_dict: Optional[Dict],
) -> Tuple[SequentialNPE, Dict, Dict]:
    """Train a single round of SNPE.

    Args:
        config: Configuration dictionary
        round_num: Current round number
        round_dir: Directory for this round
        prior: Prior distribution
        model: Existing model (None for round 0)
        proposal: Proposal distribution (None for round 0)
        norm_dict: Normalization dictionary (None for round 0)

    Returns:
        Tuple of (trained_model, posterior, norm_dict)
    """
    print(f"\n{'='*80}")
    print(f"ROUND {round_num} / {config.num_rounds - 1}")
    print(f"{'='*80}\n")

    # ========== STEP 1: Sample parameters and run simulations ==========
    # Determine how many samples to draw from prior vs proposal (combined loss)
    use_combined_loss = config.get('use_combined_loss', False) and round_num > 0
    prior_fraction = config.get('prior_fraction', 0.0) if use_combined_loss else 0.0

    if round_num == 0:
        print(f"[Round {round_num}] Sampling from prior...")
        samples = prior.sample(
            config.simulation.num_galaxies,
            seed=config.get('seed', 42) + round_num
        )
    elif use_combined_loss and prior_fraction > 0:
        # Combined loss: mix prior and proposal samples
        num_prior = int(config.simulation.num_galaxies * prior_fraction)
        num_proposal = config.simulation.num_galaxies - num_prior

        print(f"[Round {round_num}] Using combined loss scheme:")
        print(f"[Round {round_num}]   Prior samples: {num_prior} ({prior_fraction*100:.1f}%)")
        print(f"[Round {round_num}]   Proposal samples: {num_proposal} ({(1-prior_fraction)*100:.1f}%)")

        # Sample from prior
        prior_samples = prior.sample(num_prior, seed=config.get('seed', 42) + round_num)

        # Sample from proposal
        with torch.no_grad():
            obs_batch = read_observation(config, norm_dict=norm_dict)
            pre_transforms_obs = build_transformation(
                apply_projection=config.pre_transforms.apply_projection,
                apply_selection=False,
                apply_graph=False,
                apply_uncertainty=False,
                use_log_features=config.pre_transforms.use_log_features,
                projection_args=config.pre_transforms.projection_args,
                norm_dict=norm_dict
            )
            obs_batch = pre_transforms_obs(obs_batch)

            proposal_samples = proposal.sample(
                num_proposal,
                x_obs=obs_batch,
                norm_dict=norm_dict
            )

        # Combine samples
        samples = np.vstack([prior_samples, proposal_samples])
        print(f"[Round {round_num}] Combined {len(samples)} samples")
    else:
        print(f"[Round {round_num}] Sampling from proposal (posterior from round {round_num-1})...")
        with torch.no_grad():
            obs_batch = read_observation(config, norm_dict=norm_dict)
            pre_transforms_obs = build_transformation(
                apply_projection=config.pre_transforms.apply_projection,
                apply_selection=False,
                apply_graph=False,
                apply_uncertainty=False,
                use_log_features=config.pre_transforms.use_log_features,
                projection_args=config.pre_transforms.projection_args,
                norm_dict=norm_dict
            )
            obs_batch = pre_transforms_obs(obs_batch)

            samples = proposal.sample(
                config.simulation.num_galaxies,
                x_obs=obs_batch,
                norm_dict=norm_dict
            )

    # Convert samples to simulation parameters
    params_list = samples_to_simulation_params(
        samples, prior.labels,
        dm_type=config.simulation.dm_type,
        stellar_type=config.simulation.stellar_type,
        df_type=config.simulation.df_type,
        dm_params_default=config.simulation.get('dm_params_default', {}),
        stellar_params_default=config.simulation.get('stellar_params_default', {}),
        df_params_default=config.simulation.get('df_params_default', {})
    )

    # Run simulations
    print(f"[Round {round_num}] Running {config.simulation.num_galaxies} simulations...")
    num_stars_list = sample_num_stars(config, config.simulation.num_galaxies)
    node_features, graph_features = run_simulation_batch(
        params_list, num_stars_list,
        max_iter=config.simulation.get('max_iter', 1000),
        n_jobs=config.simulation.get('n_jobs', None),
        use_multiprocessing=config.simulation.get('use_multiprocessing', True)
    )
    node_features, graph_features = preprocess_simulations(
        node_features, graph_features, **config.preprocess
    )

    # ========== STEP 2: Create dataloaders ==========
    print(f"[Round {round_num}] Creating dataloaders...")
    train_loader, val_loader, round_norm_dict = datasets.prepare_dataloaders(
        node_features, graph_features, config.labels,
        cond_labels=config.get('cond_labels', None),
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        train_frac=config.train_frac,
        num_workers=config.num_workers,
        seed=config.seed_data,
        norm_dict=norm_dict if round_num > 0 else None
    )

    # Store norm_dict from round 0
    if round_num == 0:
        norm_dict = round_norm_dict
        print(f"[Round {round_num}] Created normalization dictionary (will be reused for all rounds)")
    else:
        print(f"[Round {round_num}] Reusing normalization dictionary from round 0")

    print(f"[Round {round_num}] Training samples: {len(train_loader.dataset)}")
    print(f"[Round {round_num}] Validation samples: {len(val_loader.dataset)}")

    # ========== STEP 3: Create or update model ==========
    if round_num == 0:
        print(f"[Round {round_num}] Creating model...")
        pre_transforms = build_transformation(norm_dict=norm_dict, **config.pre_transforms)
        model = create_model(
            config, pre_transforms, norm_dict,
            proposal=None,
            prior=prior,
            current_round=0
        )
    else:
        print(f"[Round {round_num}] Updating model with new proposal and round...")
        model.set_proposal(proposal)
        model.set_round(round_num)

    # ========== STEP 4: Setup training ==========
    # Close previous wandb run if this is not the first round
    if round_num > 0:
        wandb.finish()  # Critical: finish previous wandb run before creating new one
        print(f"[WandB] Closed previous wandb run")

    wandb_mode = 'disabled' if config.get('debug', False) else 'online'
    wandb_logger = WandbLogger(
        project=config.get("wandb_project", "jgnn-snpe"),
        name=f"{config.get('name')}_round{round_num}",
        id=None,
        save_dir=str(round_dir / "logs"),
        log_model=config.get("log_model", "all"),
        config=config.to_dict(),
        mode=wandb_mode,
        resume="allow",
    )

    # Only watch model in round 0 to avoid duplicate watching
    if round_num == 0:
        wandb_logger.watch(model, log="all", log_freq=500, log_graph=True)
        print(f"[WandB] Watching model parameters and gradients")

    callbacks = create_callbacks(config)

    trainer = pl.Trainer(
        default_root_dir=str(round_dir),
        max_epochs=config.num_epochs,
        max_steps=config.num_steps,
        accelerator=config.accelerator,
        callbacks=callbacks,
        logger=wandb_logger,
        enable_progress_bar=config.get("enable_progress_bar", True),
        gradient_clip_val=config.get('gradient_clip_val', None),
    )

    # ========== STEP 5: Train model ==========
    print(f"[Round {round_num}] Starting training...")
    trainer.fit(model, train_loader, val_loader)
    print(f"[Round {round_num}] Training completed!")

    best_checkpoint = trainer.checkpoint_callback.best_model_path
    print(f"[Round {round_num}] Best checkpoint: {best_checkpoint}")

    # ========== STEP 6: Build posterior for next round ==========
    posterior = model.build_posterior()

    # ========== STEP 7: Visualize posterior ==========
    visualize_round_posterior(
        model, posterior, config, round_num, round_dir, norm_dict, wandb_logger
    )

    # ========== STEP 8: Save round state ==========
    save_round_state(round_num, round_dir, model, norm_dict, config, best_checkpoint)

    # Clean up wandb
    wandb.finish()

    print(f"\n[Round {round_num}] Round complete!\n")

    return model, posterior, norm_dict


def main(config: ml_collections.ConfigDict, workdir: str = "./logging/"):
    """Main SNPE training loop with multi-round support.

    Args:
        config: Configuration dictionary
        workdir: Working directory for saving results
    """
    print(f"[Setup] Starting Sequential NPE training")
    print(f"[Setup] Working directory: {workdir}")
    print(f"[Setup] Number of rounds: {config.num_rounds}")

    # Setup base directory
    base_dir = Path(workdir) / config.name
    base_dir.mkdir(parents=True, exist_ok=True)

    # Initialize prior
    prior = BoxUniform(config.prior)

    # Check if resuming from previous run
    start_round = 0
    model = None
    proposal = None
    norm_dict = None

    if config.get('resume', False):
        resume_round = config.get('resume_round', None)

        if resume_round is None:
            resume_round = find_latest_round(base_dir)

        if resume_round is not None:
            print(f"\n[Resume] Resuming from round {resume_round}")
            round_dir = base_dir / f"round_{resume_round}"
            state = load_round_state(round_dir)

            norm_dict = state['norm_dict']
            start_round = resume_round + 1

            # Load model and build posterior
            print(f"[Resume] Loading model from: {state['best_checkpoint']}")
            model = SequentialNPE.load_from_checkpoint(
                state['best_checkpoint'],
                strict=False
            )
            model.set_round(resume_round)
            proposal = model.build_posterior()

            print(f"[Resume] Will start training from round {start_round}")
        else:
            print(f"[Resume] No previous rounds found, starting from round 0")

    # Train each round
    for current_round in range(start_round, config.num_rounds):
        round_dir = setup_round_directory(base_dir, current_round)

        model, proposal, norm_dict = train_round(
            config=config,
            round_num=current_round,
            round_dir=round_dir,
            prior=prior,
            model=model,
            proposal=proposal,
            norm_dict=norm_dict
        )

    print(f"\n{'='*80}")
    print(f"SNPE TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"[Summary] Trained {config.num_rounds} rounds")
    print(f"[Summary] Results saved to: {base_dir}")
    print(f"{'='*80}\n")


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
