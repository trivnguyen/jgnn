"""Script for running simulations with prior and/or proposal model.

This script generates parameter samples from a prior distribution and/or a trained
proposal (SNPE) model, then simulates stellar kinematics for dwarf galaxies.

Usage:
    python run_simulations_from_priors.py --config=configs/sim_config.py
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
from jgnn.priors import BoxUniform
from jgnn.sims import (
    run_simulation_batch,
    preprocess_simulations,
    samples_to_simulation_params,
    write_graph_dataset,
)

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


def main(config: ml_collections.ConfigDict):
    """Main function for running simulations.

    Parameters
    ----------
    config : ConfigDict
        Configuration dictionary with keys:
        - prior: Prior configuration
        - simulation: Simulation configuration
        - output: Output configuration
    """
    print("[Setup] Starting simulation run")
    print(f"[Setup] Output: {config.output.path}")

    # Set random seed
    if config.get('seed') is not None:
        torch.manual_seed(config.seed)
        print(f"[Setup] Random seed: {config.seed}")

    # Initialize prior distribution
    print("[Prior] Initializing BoxUniform prior")
    prior = BoxUniform(config.prior)
    print(f"[Prior] {prior}")

    # Sample parameters
    num_samples = config.simulation.num_galaxies
    print(f"\n[Sampling] Generating {num_samples} parameter samples...")
    samples = prior.sample(num_samples=num_samples, seed=config.get('seed'))

    # Convert samples to simulation parameters
    params_list = samples_to_simulation_params(
        samples,
        config.prior.labels,
        dm_type=config.simulation.dm_type,
        stellar_type=config.simulation.stellar_type,
        df_type=config.simulation.df_type,
        dm_params_default=config.simulation.get('dm_params_default', {}),
        stellar_params_default=config.simulation.get('stellar_params_default', {}),
        df_params_default=config.simulation.get('df_params_default', {}),
    )
    num_stars_list = sample_num_stars(config, num_samples)

    print(f"[Sampling] Sampled number of stars: mean={num_stars_list.mean():.1f}, std={num_stars_list.std():.1f}")

    print(params_list[0])


    # Run simulations and preprocess
    node_features, graph_features = run_simulation_batch(
        params_list,
        num_stars_list,
        max_iter=config.simulation.get('max_iter', 1000),
        use_multiprocessing=config.simulation.get('use_multiprocessing', False),
        n_jobs=config.simulation.get('n_jobs', 1),
    )

    print("\n[Preprocessing] Applying preprocessing transformations...")

    if config.get('preprocess') is not None:
        node_features, graph_features = preprocess_simulations(
            node_features, graph_features, **config.preprocess)

    # Save preprocessed results
    metadata = {
        'num_galaxies': len(graph_features['num_stars']),
        'dm_type': config.simulation.dm_type,
        'stellar_type': config.simulation.stellar_type,
        'df_type': config.simulation.df_type,
    }
    output_path = config.output.path
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    write_graph_dataset(
        output_path,
        node_features,
        graph_features,
        graph_features['num_stars'],
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
