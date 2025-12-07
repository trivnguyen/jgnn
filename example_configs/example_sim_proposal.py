"""Example configuration for simulation with proposal (SNPE) model.

This config samples parameters from a trained SNPE proposal conditioned on
observation data, then runs simulations.

Usage:
    python run_simulations.py --config=example_configs/example_sim_proposal.py
"""
from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    # Random seed
    config.seed = 42

    ### PROPOSAL CONFIGURATION ###
    config.proposal = proposal = ConfigDict()

    # Path to trained NPE checkpoint
    proposal.checkpoint = '/mnt/ceph/users/tnguyen/jeans_gnn/trained_models-v2/example_npe_run/jgnn_v2.0_test/abc123/checkpoints/best.ckpt'

    # Sampling parameters
    proposal.batch_size = 100  # Batch size for sampling from proposal

    # Device for inference
    proposal.device = 'cuda'  # or 'cpu'

    ### OBSERVATION CONFIGURATION ###
    # Required when using proposal
    config.observation = observation = ConfigDict()

    # Path to observation HDF5 file (must contain pos, vel, and optionally vel_error)
    observation.path = '/mnt/home/tnguyen/projects/jeans_gnn/observations/draco/draco_processed.hdf5'

    ### SIMULATION CONFIGURATION ###
    config.simulation = simulation = ConfigDict()

    # Galaxy model types (should match the types used in training data)
    simulation.dm_type = 'Spheroid'
    simulation.stellar_type = 'Plummer'
    simulation.df_type = 'QuasiSpherical'

    # Number of galaxies to simulate
    simulation.num_galaxies = 1000

    # Number of stars per galaxy distribution
    simulation.num_stars_dist = 'poisson'
    simulation.num_stars_mean = 100

    # Maximum iterations for each simulation
    simulation.max_iter = 1000

    # Default parameter values (for parameters not in proposal output)
    simulation.dm_params_default = {}
    simulation.stellar_params_default = {}
    simulation.df_params_default = {}

    ### OUTPUT CONFIGURATION ###
    config.output = output = ConfigDict()
    output.path = '/mnt/home/tnguyen/projects/jeans_gnn/datasets/raw_datasets/test_sim_proposal.hdf5'

    return config
