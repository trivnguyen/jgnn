"""Example configuration for simulation with proposal + prior truncation.

This config samples parameters from a trained SNPE proposal conditioned on
observation data, truncates samples to prior bounds, then runs simulations.

This is useful for sequential NPE where you want to ensure samples stay within
the prior support while still being informed by the posterior.

Usage:
    python run_simulations.py --config=example_configs/example_sim_proposal_truncated.py
"""
from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    # Random seed
    config.seed = 42

    ### PRIOR CONFIGURATION ###
    # BoxUniform prior for truncation
    config.prior = prior = ConfigDict()

    # Parameter labels (must match proposal output labels)
    prior.labels = [
        'dm_gamma',
        'dm_log_r_dm',
        'dm_log_rho_0',
        'stellar_r_star_r_dm',
        'df_beta0',
        'df_log_r_a',
    ]

    # Prior bounds - samples outside these bounds will be rejected
    prior.min = [
        -1.0,      # dm_gamma
        -2.0,      # dm_log_r_dm
        3.0,       # dm_log_rho_0
        0.2,       # stellar_r_star_r_dm
        -0.499,    # df_beta0
        -1.0,      # df_log_r_a
    ]

    prior.max = [
        2.0,       # dm_gamma
        2.0,       # dm_log_r_dm
        10.0,      # dm_log_rho_0
        1.0,       # stellar_r_star_r_dm
        0.999,     # df_beta0
        1.0,       # df_log_r_a
    ]

    ### PROPOSAL CONFIGURATION ###
    config.proposal = proposal = ConfigDict()

    # Path to trained NPE checkpoint from previous round
    proposal.checkpoint = '/mnt/ceph/users/tnguyen/jeans_gnn/trained_models-v2/example_npe_run/jgnn_v2.0_test/abc123/checkpoints/best.ckpt'

    # Sampling parameters
    proposal.batch_size = 100

    # Device for inference
    proposal.device = 'cuda'

    ### OBSERVATION CONFIGURATION ###
    config.observation = observation = ConfigDict()

    # Path to observation HDF5 file
    observation.path = '/mnt/home/tnguyen/projects/jeans_gnn/observations/draco/draco_processed.hdf5'

    ### SIMULATION CONFIGURATION ###
    config.simulation = simulation = ConfigDict()

    # Galaxy model types
    simulation.dm_type = 'Spheroid'
    simulation.stellar_type = 'Plummer'
    simulation.df_type = 'QuasiSpherical'

    # Number of galaxies to simulate
    simulation.num_galaxies = 1000

    # Number of stars per galaxy
    simulation.num_stars_dist = 'poisson'
    simulation.num_stars_mean = 100

    # Maximum iterations for each simulation
    simulation.max_iter = 1000

    # Default parameter values
    simulation.dm_params_default = {}
    simulation.stellar_params_default = {}
    simulation.df_params_default = {}

    ### OUTPUT CONFIGURATION ###
    config.output = output = ConfigDict()
    output.path = '/mnt/home/tnguyen/projects/jeans_gnn/datasets/raw_datasets/test_sim_proposal_truncated.hdf5'

    return config
