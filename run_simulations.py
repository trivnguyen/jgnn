"""Script for running simulations with prior and/or proposal model.

This script generates parameter samples from a prior distribution and/or a trained
proposal (SNPE) model, then simulates stellar kinematics for dwarf galaxies.

Usage:
    # Sample from prior only
    python run_simulations.py --config=configs/sim_config.py

    # Sample from proposal (requires observation data)
    python run_simulations.py --config=configs/sim_config.py

    # Sample from proposal with prior truncation
    python run_simulations.py --config=configs/sim_config.py
"""
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import yaml
import numpy as np
import h5py
import torch
from tqdm import tqdm
from absl import flags
from ml_collections import config_flags
import ml_collections

import datasets
from jgnn.models import SequentialNPE
from jgnn.priors import BoxUniform
from jgnn.sims import run_simulation as simulate_galaxy
from jgnn.sims.simulator import preprocess
from jgnn.sims.utils import write_graph_dataset


def load_proposal_from_checkpoint(checkpoint_path: str) -> tuple:
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

    # Load full SequentialNPE model from checkpoint
    model = SequentialNPE.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Extract hyperparameters
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    hparams = checkpoint['hyper_parameters']
    norm_dict = hparams.get('norm_dict')

    # Get labels from norm_dict or checkpoint
    labels = None
    if norm_dict is not None and 'labels' in norm_dict:
        labels = norm_dict['labels']

    print(f"[Proposal] Model loaded successfully")
    print(f"[Proposal] Output size: {model.output_size}")
    print(f"[Proposal] Labels: {labels}")

    return model, norm_dict, labels


def load_observation(
    observation_path: str,
    model: SequentialNPE,
    norm_dict: dict = None,
) -> torch.Tensor:
    """Load and preprocess observation data for proposal conditioning.

    Parameters
    ----------
    observation_path : str
        Path to observation HDF5 file
    model : SequentialNPE
        SequentialNPE model (used to apply pre-transforms)
    norm_dict : dict, optional
        Normalization dictionary

    Returns
    -------
    observation_batch : PyG Data object
        Preprocessed observation ready for conditioning
    """
    from datasets.preprocess import create_graph_from_posvel
    from torch_geometric.data import Batch

    print(f"[Observation] Loading observation from: {observation_path}")

    # Read observation
    node_feats, graph_feats, headers = datasets.read_graph_dataset(
        observation_path, concat=False, to_array=True
    )

    # Get the first (and should be only) graph
    pos = node_feats['pos'][0]
    vel = node_feats['vel'][0]
    vel_error = node_feats.get('vel_error', [None])[0]

    # Create a dummy label (will not be used)
    dummy_label = [0.0] * model.output_size

    # Get conditioning if it exists
    if 'cond' in graph_feats:
        cond = graph_feats['cond'][0]
    else:
        cond = None

    # Create graph
    graph = create_graph_from_posvel(
        pos, vel, vel_error=vel_error, label=dummy_label, cond=cond
    )

    # Normalize if norm_dict is provided
    if norm_dict is not None:
        theta_loc = torch.tensor(norm_dict['theta_loc'], dtype=torch.float32)
        theta_scale = torch.tensor(norm_dict['theta_scale'], dtype=torch.float32)
        graph.theta = (graph.theta - theta_loc) / theta_scale

    # Create batch
    observation_batch = Batch.from_data_list([graph])

    print(f"[Observation] Observation loaded: {pos.shape[0]} stars")

    return observation_batch


def sample_from_proposal(
    model: SequentialNPE,
    observation: torch.Tensor,
    num_samples: int = 1000,
    batch_size: int = 100,
    device: str = 'cpu',
) -> np.ndarray:
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


def denormalize_samples(
    samples: np.ndarray,
    norm_dict: dict,
) -> np.ndarray:
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


def samples_to_simulation_params(
    samples: np.ndarray,
    labels: list,
    sim_config: ml_collections.ConfigDict,
) -> list:
    """Convert parameter samples to simulation parameter format.

    Handles log-space parameters: labels containing 'log_' will be converted
    from log10 space to linear space (e.g., 'dm_log_r_dm' -> 'dm_r_dm').

    Parameters
    ----------
    samples : np.ndarray, shape (num_samples, num_params)
        Parameter samples
    labels : list of str
        Parameter names (may include 'log_' prefix for log-space parameters)
    sim_config : ConfigDict
        Simulation configuration with keys:
        - dm_type, stellar_type, df_type
        - default values for parameters not in labels

    Returns
    -------
    params_list : list of dict
        List of parameter dictionaries ready for simulator
    """
    num_samples = len(samples)
    params_list = []

    for i in range(num_samples):
        # Create parameter dictionaries
        dm_params = {}
        stellar_params = {}
        df_params = {}

        # Fill in sampled parameters
        for j, label in enumerate(labels):
            value = samples[i, j]

            # Handle log-space parameters
            # If label contains 'log_', convert from log10 to linear
            if 'log_' in label:
                # Remove 'log_' from the label
                linear_label = label.replace('log_', '')
                linear_value = 10 ** value
            else:
                linear_label = label
                linear_value = value

            # Parse label to determine which dict it belongs to
            if linear_label.startswith('dm_'):
                param_name = linear_label[3:]  # Remove 'dm_' prefix
                dm_params[param_name] = linear_value
            elif linear_label.startswith('stellar_'):
                param_name = linear_label[8:]  # Remove 'stellar_' prefix
                stellar_params[param_name] = linear_value
            elif linear_label.startswith('df_'):
                param_name = linear_label[3:]  # Remove 'df_' prefix
                df_params[param_name] = linear_value
            else:
                # Try to infer from parameter name
                param_name = linear_label
                if param_name in ['rho_0', 'r_dm', 'alpha', 'beta', 'gamma', 'q', 'p']:
                    dm_params[param_name] = linear_value
                    print(dm_params)
                elif param_name in ['r_star', 'r_star_r_dm']:
                    stellar_params[param_name] = linear_value
                elif param_name in ['r_a', 'r_a_r_dm', 'r_a_r_star', 'beta0']:
                    df_params[param_name] = linear_value

        # Fill in default values from config if not already set
        for key, val in sim_config.get('dm_params_default', {}).items():
            if key not in dm_params:
                dm_params[key] = val

        for key, val in sim_config.get('stellar_params_default', {}).items():
            if key not in stellar_params:
                stellar_params[key] = val

        for key, val in sim_config.get('df_params_default', {}).items():
            if key not in df_params:
                df_params[key] = val

        # Construct full parameter dict
        params = {
            'dm_type': sim_config.dm_type,
            'stellar_type': sim_config.stellar_type,
            'df_type': sim_config.df_type,
            'dm_params': dm_params,
            'stellar_params': stellar_params,
            'df_params': df_params,
        }

        params_list.append(params)

    return params_list


def run_simulations_batch(
    params_list: list,
    num_stars_list: list,
    max_iter: int = 1000,
) -> tuple:
    """Run simulations for a batch of parameter sets.

    Parameters
    ----------
    params_list : list of dict
        List of parameter dictionaries
    num_stars_list : list of int
        Number of stars for each galaxy
    max_iter : int
        Maximum iterations for each simulation

    Returns
    -------
    node_features : dict
        Node features as lists (not concatenated), ready for preprocessing
    graph_features : dict
        Graph features as arrays
    """

    num_galaxies = len(params_list)
    all_pos = []
    all_vel = []
    graph_feat_lists = {key: [] for key in []}
    successful_sims = []

    print(f"[Simulations] Running {num_galaxies} simulations...")

    for i in tqdm(range(num_galaxies), desc="Simulating galaxies"):
        try:
            node_feat, graph_feat = simulate_galaxy(
                params_list[i],
                num_stars_list[i],
                max_iter=max_iter
            )

            all_pos.append(node_feat['pos'])
            all_vel.append(node_feat['vel'])

            # Initialize graph feature lists on first success
            if len(graph_feat_lists) == 0:
                graph_feat_lists = {key: [] for key in graph_feat.keys()}

            # Append graph features
            for key in graph_feat.keys():
                graph_feat_lists[key].append(graph_feat[key])

            successful_sims.append(i)

        except Exception as e:
            print(f"\n[Warning] Simulation {i} failed: {str(e)}")
            raise e

    # Combine results
    if len(successful_sims) == 0:
        raise RuntimeError("All simulations failed")

    print(f"[Simulations] Completed {len(successful_sims)}/{num_galaxies} simulations successfully")

    # Return node features as lists (not concatenated) for preprocessing
    node_features = {
        'pos': all_pos,
        'vel': all_vel,
    }

    graph_features = {
        key: np.array(values) for key, values in graph_feat_lists.items()
    }

    return node_features, graph_features


def save_simulations(
    output_path: str,
    node_features: dict,
    graph_features: dict,
    num_stars: list,
    metadata: dict = None,
):
    """Save preprocessed simulation results to HDF5 file using jgnn.sims.utils.

    Parameters
    ----------
    output_path : str
        Path to output HDF5 file
    node_features : dict
        Node features with concatenated arrays (pos, vel, vel_error, etc.)
    graph_features : dict
        Graph features (parameters)
    num_stars : list or np.ndarray
        Number of stars per galaxy
    metadata : dict, optional
        Additional metadata to save as headers
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Prepare headers
    headers = metadata.copy() if metadata is not None else {}

    # Use write_graph_dataset from jgnn.sims.utils
    write_graph_dataset(
        output_path,
        node_features,
        graph_features,
        num_stars,
        headers=headers
    )

    print(f"[Save] Preprocessed simulations saved to: {output_path}")


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
        np.random.seed(config.seed)
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
    labels = None

    if use_proposal:
        if config.get('observation') is None or config.observation.get('path') is None:
            raise ValueError("Observation data is required when using proposal model")

        proposal_model, norm_dict, labels = load_proposal_from_checkpoint(
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

        # If labels not set from proposal, get from prior
        if labels is None:
            labels = prior.labels

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
        config.simulation
    )

    # Sample number of stars per galaxy
    if config.simulation.num_stars_dist == 'poisson':
        num_stars_list = np.random.poisson(
            config.simulation.num_stars_mean,
            size=num_samples
        )
    elif config.simulation.num_stars_dist == 'uniform':
        num_stars_list = np.random.randint(
            config.simulation.num_stars_min,
            config.simulation.num_stars_max,
            size=num_samples
        )
    elif config.simulation.num_stars_dist == 'delta':
        num_stars_list = np.full(num_samples, config.simulation.num_stars_value)
    else:
        raise ValueError(f"Unknown num_stars_dist: {config.simulation.num_stars_dist}")

    print(f"[Sampling] Sampled number of stars: mean={num_stars_list.mean():.1f}, std={num_stars_list.std():.1f}")

    # Run simulations
    node_features, graph_features = run_simulations_batch(
        params_list,
        num_stars_list,
        max_iter=config.simulation.get('max_iter', 1000)
    )

    # Preprocess simulations
    print("\n[Preprocessing] Applying preprocessing transformations...")
    preprocess_config = config.get('preprocess', ml_collections.ConfigDict())

    node_features, graph_features = preprocess(
        node_features,
        graph_features,
        vrange=preprocess_config.get('vrange', (0, np.inf)),
        vdisp_range=preprocess_config.get('vdisp_range', (0, np.inf)),
        r_range=preprocess_config.get('r_range', (0, np.inf)),
        r_rstar_range=preprocess_config.get('r_rstar_range', (0, np.inf)),
        apply_projection=preprocess_config.get('apply_projection', True),
        projection_axis=preprocess_config.get('projection_axis', None),
        use_proper_motions=preprocess_config.get('use_proper_motions', False),
        norm_rstar=preprocess_config.get('norm_rstar', False),
        seed=config.get('seed'),
    )

    num_stars = graph_features['num_stars']
    print(f"[Preprocessing] Kept {len(num_stars)} galaxies after preprocessing")

    # Save preprocessed results
    metadata = {
        'use_proposal': use_proposal,
        'use_prior': use_prior,
        'num_galaxies': len(num_stars),
        'dm_type': config.simulation.dm_type,
        'stellar_type': config.simulation.stellar_type,
        'df_type': config.simulation.df_type,
    }

    if use_proposal:
        metadata['proposal_checkpoint'] = config.proposal.checkpoint
        metadata['observation_path'] = config.observation.path

    save_simulations(
        config.output.path,
        node_features,
        graph_features,
        num_stars,
        metadata=metadata
    )

    # Save parameter samples separately
    # samples_path = config.output.path.replace('.hdf5', '_samples.npy')
    # np.save(samples_path, samples)
    # print(f"[Save] Parameter samples saved to: {samples_path}")

    # Save config
    # config_path = config.output.path.replace('.hdf5', '_config.yaml')
    # with open(config_path, 'w') as f:
    #     yaml.dump(config.to_dict(), f)
    # print(f"[Save] Config saved to: {config_path}")

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
