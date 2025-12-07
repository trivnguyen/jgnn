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
    proposal.batch_size = 100
    proposal.device = 'cuda'

    ### OBSERVATION CONFIGURATION ###
    config.observation = observation = ConfigDict()

    # Path to observation HDF5 file
    observation.path = '/mnt/home/tnguyen/projects/jeans_gnn/observations/draco/draco_processed.hdf5'

    ### SIMULATION CONFIGURATION ###
    config.simulation = simulation = ConfigDict()
    simulation.dm_type = 'Spheroid'          # Dark matter potential type
    simulation.stellar_type = 'Plummer'      # Stellar density profile type
    simulation.df_type = 'QuasiSpherical'   # Distribution function type
    simulation.num_galaxies = 100
    simulation.num_stars_dist = 'poisson'  # Options: 'poisson', 'uniform', 'delta'
    simulation.num_stars_mean = 100        # Mean for Poisson
    simulation.max_iter = 1000
    simulation.dm_params_default = {
        'alpha': 1.0,    # Fixed gFNW
        'beta': 3.0,     # Fixed gNFW
    }
    simulation.stellar_params_default = {}
    simulation.df_params_default = {}

    ### PREPROCESSING CONFIGURATION ###
    config.preprocess = preprocess = ConfigDict()
    preprocess.vrange = (0, 1000)         # 3D velocity range in km/s
    preprocess.vdisp_range = (0, 1000)    # Velocity dispersion range
    preprocess.r_range = (0, 100.)        # Radius range in kpc
    preprocess.r_rstar_range = (0, 10.)  # Radius range in units of r_star
    preprocess.apply_projection = False      # Apply 2D projection
    preprocess.projection_axis = None       # Random projection (or 0, 1, 2 for fixed axis)
    preprocess.use_proper_motions = False   # Include proper motions
    preprocess.norm_rstar = False           # Normalize positions by r_star

    ### OUTPUT CONFIGURATION ###
    config.output = output = ConfigDict()
    output.path = '/mnt/home/tnguyen/projects/jeans_gnn/datasets/raw_datasets/test_sim_proposal_truncated.hdf5'

    return config
