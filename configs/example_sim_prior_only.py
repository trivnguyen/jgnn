"""Example configuration for simulation with prior only.

This config samples parameters from a BoxUniform prior and runs simulations.
No proposal model is used.

Usage:
    python run_simulations.py --config=example_configs/example_sim_prior_only.py
"""

from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    # Random seed
    config.seed = 42

    ### PRIOR CONFIGURATION ###
    # BoxUniform prior specification
    config.prior = prior = ConfigDict()

    # Parameter labels (must match simulation parameter names with prefixes)
    prior.labels = [
        'dm_gamma',        # DM density profile slope (for gNFW/Spheroid)
        'dm_log_r_dm',     # log10 of DM scale radius (kpc)
        'dm_log_rho_0',    # log10 of DM density normalization (Msun/kpc^3)
        'stellar_r_star_r_dm',  # Stellar scale radius as ratio to r_dm
        'df_beta0',        # Velocity anisotropy parameter
        'df_log_r_a',      # log10 of anisotropy radius (kpc) - will be converted to r_a_r_star
    ]

    # Prior bounds (must match length of labels)
    prior.min = [
        -1.0,      # dm_gamma: range from -1 to 2 (cusp to core)
        -2.0,      # dm_log_r_dm: 10^-2 to 10^2 kpc
        3.0,       # dm_log_rho_0: 10^3 to 10^10 Msun/kpc^3
        0.2,       # stellar_r_star_r_dm: 0.2 to 1.0
        0.0,    # df_beta0: -0.499 to 0.999 (isotropic to radial)
        -1.0,      # df_log_r_a: 10^-1 to 10^1 kpc (will convert to ratio)
    ]

    prior.max = [
        2.0,       # dm_gamma
        2.0,       # dm_log_r_dm
        10.0,      # dm_log_rho_0
        1.0,       # stellar_r_star_r_dm
        0.0,     # df_beta0
        1.0,       # df_log_r_a
    ]

    ### SIMULATION CONFIGURATION ###
    config.simulation = simulation = ConfigDict()

    # Galaxy model types
    simulation.dm_type = 'Spheroid'          # Dark matter potential type
    simulation.stellar_type = 'Plummer'      # Stellar density profile type
    simulation.df_type = 'QuasiSpherical'   # Distribution function type

    # Number of galaxies to simulate
    simulation.num_galaxies = 100

    # Number of stars per galaxy distribution
    simulation.num_stars_dist = 'poisson'  # Options: 'poisson', 'uniform', 'delta'
    simulation.num_stars_mean = 100        # Mean for Poisson
    # simulation.num_stars_min = 50        # Min for uniform
    # simulation.num_stars_max = 150       # Max for uniform
    # simulation.num_stars_value = 100     # Value for delta

    # Maximum iterations for each simulation
    simulation.max_iter = 1000

    # Default parameter values (for parameters not sampled)
    simulation.dm_params_default = {
        'alpha': 1.0,    # Fixed gFNW
        'beta': 3.0,     # Fixed gNFW
    }
    simulation.stellar_params_default = {}
    simulation.df_params_default = {}

    ### PREPROCESSING CONFIGURATION ###
    config.preprocess = preprocess = ConfigDict()

    # Velocity and radius ranges
    preprocess.vrange = (0, 1000)         # 3D velocity range in km/s
    preprocess.vdisp_range = (0, 1000)    # Velocity dispersion range
    preprocess.r_range = (0, 100.)        # Radius range in kpc
    preprocess.r_rstar_range = (0, 10.)  # Radius range in units of r_star

    # Projection
    preprocess.apply_projection = False      # Apply 2D projection
    preprocess.projection_axis = None       # Random projection (or 0, 1, 2 for fixed axis)
    preprocess.use_proper_motions = False   # Include proper motions

    # Normalization
    preprocess.norm_rstar = False           # Normalize positions by r_star

    ### OUTPUT CONFIGURATION ###
    config.output = output = ConfigDict()
    output.path = '/mnt/home/tnguyen/ceph/jeans_gnn/datasets/example/test_sim_prior_only.hdf5'

    return config
