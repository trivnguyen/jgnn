"""Script for running simulations with prior and/or proposal model.

This script generates parameter samples from a prior distribution and/or a trained
proposal (SNPE) model, then simulates stellar kinematics for dwarf galaxies.

Usage:
    python run_simulations.py --config=configs/sim_config.py
"""
import os
import sys
from pathlib import Path
import warnings


import yaml
import numpy as np
import h5py
import torch
from tqdm import tqdm
from absl import flags
from ml_collections import config_flags
import ml_collections

from jgnn import datasets
from jgnn.models import SequentialNPE, GNNEmbedding, TransformerEmbedding
from jgnn.priors import BoxUniform
from jgnn.sims import (
    run_simulation_batch,
    preprocess_simulation,
    samples_to_simulation_params,
    write_graph_dataset,
    load_observation,
)


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
        raise ValueError(f"Unsupported embedding model type: {config.model.type}")


def load_proposal_from_checkpoint(
    checkpoint_path: str, config: ml_collections.ConfigDict = None):
    """Load SequentialNPE proposal model from checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the SequentialNPE checkpoint file

    Returns
    -------
    model : SequentialNPE
        Loaded SequentialNPE model in eval mode
    norm_dict : dict
        Normalization dictionary from checkpoint
    labels : list
        List of parameter labels
    """
    print(f"[Proposal] Loading proposal model from: {checkpoint_path}")

    # Create new embedding network
    print("[Model] Creating new embedding network...")
    embedding_nn = create_embedding_network(config)

    # Create SNPE model
    print("[Model] Creating SNPE model...")
    model = SequentialNPE(
        input_size=config.model.input_size,
        output_size=config.model.output_size,
        flows_args=config.model.flows,
        embedding_nn=embedding_nn,
        optimizer_args=config.optimizer,
        scheduler_args=config.scheduler,
        norm_dict=norm_dict,
        pre_transforms=pre_transforms,
    )

    # Load checkpoint and hyperparameters
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    hparams = checkpoint['hyper_parameters']
    norm_dict = hparams.get('norm_dict')

    # Load model state
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    print(f"[Proposal] Model loaded successfully")
    print(f"[Proposal] Output size: {model.output_size}")

    return model, norm_dict


def sample_from_proposal(
    model,
    observation,
    num_samples: int = 1000,
    batch_size: int = 100,
    device: str = 'cpu',
):
    """Sample parameters from the proposal posterior.

    Parameters
    ----------
    model : SequentialNPE
        Trained SequentialNPE model
    observation : PyG Data object
        Observation data for conditioning
    num_samples : int
        Number of samples to draw
    batch_size : int
        Batch size for sampling
    device : str
        Device to use for computation

    Returns
    -------
    samples : np.ndarray, shape (num_samples, output_size)
        Posterior samples (normalized)
    """
    model = model.to(device)
    model.eval()

    all_samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    print(f"[Sampling] Sampling {num_samples} parameters from proposal posterior...")

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Sampling"):
            # Determine batch size for this iteration
            current_batch_size = min(batch_size, num_samples - i * batch_size)

            # Move observation to device
            obs_batch = observation.to(device)

            # Get flow context from the observation
            flow_context = model.forward(
                obs_batch.x,
                obs_batch.edge_index,
                batch=obs_batch.batch,
                edge_attr=obs_batch.edge_attr,
                edge_weight=obs_batch.edge_weight,
                cond=obs_batch.cond if hasattr(obs_batch, 'cond') else None,
            )

            # Sample from the flow
            samples = model.flows(flow_context).sample((current_batch_size,))

            # Move to CPU and store
            all_samples.append(samples.cpu().numpy())

    # Concatenate all samples
    all_samples = np.concatenate(all_samples, axis=0)

    return all_samples


def sample_num_stars(config: ml_collections.ConfigDict, num_samples: int):
    """Sample number of stars per galaxy based on configuration. """

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


def denormalize_samples(samples, norm_dict):
    """Denormalize samples back to original scale.

    Parameters
    ----------
    samples : np.ndarray
        Normalized samples
    norm_dict : dict
        Normalization dictionary

    Returns
    -------
    denormalized_samples : np.ndarray
        Samples in original scale
    """
    theta_loc = np.array(norm_dict['theta_loc'])
    theta_scale = np.array(norm_dict['theta_scale'])

    return samples * theta_scale + theta_loc


def main(config: ml_collections.ConfigDict):
    """Main function for running simulations.

    Parameters
    ----------
    config : ConfigDict
        Configuration dictionary with keys:
        - prior: Prior configuration (required if proposal is not given or for truncation)
        - proposal: Proposal configuration (optional)
        - observation: Observation configuration (required if proposal is given)
        - simulation: Simulation configuration
        - output: Output configuration
    """
    print("[Setup] Starting simulation run")
    print(f"[Setup] Output: {config.output.path}")

    # Set random seed
    if config.get('seed') is not None:
        torch.manual_seed(config.seed)
        print(f"[Setup] Random seed: {config.seed}")

    # Determine sampling mode
    use_proposal = config.get('proposal') is not None and config.proposal.get('checkpoint') is not None
    use_prior = config.get('prior') is not None

    print(f"[Mode] Use proposal: {use_proposal}")
    print(f"[Mode] Use prior: {use_prior}")

    # Load proposal if specified
    proposal_model = None
    norm_dict = None

    if use_proposal:
        if config.get('observation') is None or config.observation.get('path') is None:
            raise ValueError("Observation data is required when using proposal model")

        proposal_model, norm_dict = load_proposal_from_checkpoint(
            config.proposal.checkpoint
        )

        # Load observation
        observation = load_observation(
            config.observation.path,
            proposal_model,
            norm_dict
        )

        device = config.proposal.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Proposal] Using device: {device}")

    # Initialize prior if specified
    prior = None
    if use_prior:
        print("[Prior] Initializing BoxUniform prior")
        prior = BoxUniform(config.prior)
        print(f"[Prior] {prior}")

    # Sample parameters
    num_samples = config.simulation.num_galaxies
    print(f"\n[Sampling] Generating {num_samples} parameter samples...")

    if use_proposal:
        # Sample from proposal with truncation if needed
        samples = []
        total_generated = 0

        while len(samples) < num_samples:
            # Determine batch size for this iteration
            remaining = num_samples - len(samples)
            current_batch = min(remaining * 2, config.proposal.get('batch_size', 100))

            # Sample from proposal
            batch_samples = sample_from_proposal(
                proposal_model,
                observation,
                num_samples=current_batch,
                batch_size=config.proposal.get('batch_size', 100),
                device=device
            )
            total_generated += current_batch

            # Denormalize
            if norm_dict is not None:
                batch_samples = denormalize_samples(batch_samples, norm_dict)

            # Apply truncation if prior is given
            if use_prior:
                within_bounds = prior.is_within_bounds(batch_samples)
                acceptance_fraction = np.sum(within_bounds) / len(batch_samples)
                print(f"[Truncation] Batch acceptance fraction: {acceptance_fraction:.3f} "
                      f"({np.sum(within_bounds)}/{len(batch_samples)} samples)")
                batch_samples = batch_samples[within_bounds]

            samples.append(batch_samples)

        # Concatenate and trim to exact number
        samples = np.concatenate(samples, axis=0)[:num_samples]

        # Print acceptance statistics
        if use_prior:
            acceptance_fraction = len(samples) / total_generated
            print(f"[Truncation] Acceptance fraction: {acceptance_fraction:.3f} "
                  f"({len(samples)}/{total_generated} samples)")

    elif use_prior:
        # Sample from prior only
        print("[Prior] Sampling from prior distribution...")
        samples = prior.sample(num_samples=num_samples, seed=config.get('seed'))

    else:
        raise ValueError("Either prior or proposal must be specified")

    print(f"[Sampling] Generated {len(samples)} parameter samples")

    # Convert samples to simulation parameters
    params_list = samples_to_simulation_params(
        samples,
        labels,
        dm_type=config.simulation.dm_type,
        stellar_type=config.simulation.stellar_type,
        df_type=config.simulation.df_type,
        dm_params_default=config.simulation.get('dm_params_default', {}),
        stellar_params_default=config.simulation.get('stellar_params_default', {}),
        df_params_default=config.simulation.get('df_params_default', {}),
    )
    num_stars_list = sample_num_stars(config, num_samples)

    print(f"[Sampling] Sampled number of stars: mean={num_stars_list.mean():.1f}, std={num_stars_list.std():.1f}")

    # Run simulations and preprocess
    node_features, graph_features = run_simulation_batch(
        params_list,
        num_stars_list,
        max_iter=config.simulation.get('max_iter', 1000)
    )

    print("\n[Preprocessing] Applying preprocessing transformations...")
    node_features, graph_features = preprocess_simulation(
        node_features, graph_features, **config.preprocess)

    # Save preprocessed results
    metadata = {
        'use_proposal': use_proposal,
        'use_prior': use_prior,
        'num_galaxies': len(graph_features['num_stars']),
        'dm_type': config.simulation.dm_type,
        'stellar_type': config.simulation.stellar_type,
        'df_type': config.simulation.df_type,
    }

    if use_proposal:
        metadata['proposal_checkpoint'] = config.proposal.checkpoint
        metadata['observation_path'] = config.observation.path

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    write_graph_dataset(
        output_path,
        node_features,
        graph_features,
        num_stars,
        headers=metadata.copy() if metadata is not None else {}
    )

    print("\n[Done] Simulation complete!")


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config",
        None,
        "File path to the simulation configuration.",
        lock_config=True,
    )
    FLAGS(sys.argv)
    main(config=FLAGS.config)
