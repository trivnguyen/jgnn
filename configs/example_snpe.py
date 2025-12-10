"""Example configuration for Sequential Neural Posterior Estimation (SNPE) training."""

import numpy as np
import pandas as pd
from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    # Seeding
    config.seed_data = 4042
    config.seed = 42

    # Parameter labels
    config.labels = [
        'dm_gamma', 'dm_log_r_dm', 'dm_log_rho_0',
        'df_beta0', 'df_log_r_a_r_star',
    ]
    config.cond_labels = None  # ['stellar_log_r_star'] if using conditioning

    # Data configuration
    config.train_frac = 0.9
    config.train_batch_size = 128
    config.eval_batch_size = 256
    config.num_workers = 0

    ### LOGGING AND WANDB CONFIGURATION ###
    config.workdir = '/mnt/ceph/users/tnguyen/jeans_gnn/trained_models-snpe'
    config.name = 'example_snpe_run'
    config.debug = False
    config.wandb_project = 'jgnn_snpe'
    config.enable_progress_bar = True
    config.log_model = 'all'

    # Resume configuration
    config.resume = False  # Set to True to resume from a previous run
    config.resume_round = None  # Specific round to resume from (None = auto-detect latest)
    config.resume_dir = None  # Directory to resume from (None = use workdir/name)

    ### OBSERVATION CONFIGURATION (for SNPE) ###
    config.use_observation = True
    config.observation = observation = ConfigDict()
    observation.path = '/mnt/home/tnguyen/projects/jeans_gnn/datasets/processed_data/draco_desi_p0.80.csv'
    observation.key = 'draco_1'
    observation.prob_threshold = 0.8

    # Load metadata for the observation
    meta = pd.read_csv('/mnt/home/tnguyen/projects/jeans_gnn/datasets/tables/dwarfs.csv')
    rstar = meta[meta.key == observation.key].rhalf_sph_physical.values[0] / 1000
    observation.meta = {
        'stellar_log_r_star': np.log10(rstar)
    }

    ### SIMULATION AND PRIOR CONFIGURATION ###
    config.simulation = simulation = ConfigDict()
    simulation.dm_type = 'Spheroid'
    simulation.stellar_type = 'Plummer'
    simulation.df_type = 'QuasiSpherical'
    simulation.num_stars_dist = 'poisson'  # Options: 'poisson', 'uniform', 'delta'
    simulation.num_stars_mean = 100
    simulation.num_galaxies = 10_000  # Simulations per round
    simulation.max_iter = 1000

    # Multiprocessing for simulations
    simulation.use_multiprocessing = True
    simulation.n_jobs = None  # None = use all CPUs

    # Default parameters for AGAMA
    simulation.dm_params_default = {'alpha': 1.0, 'beta': 3.0}  # generalized NFW
    simulation.stellar_params_default = {'r_star': rstar}
    simulation.df_params_default = {}

    # Prior configuration
    config.prior = prior = ConfigDict()
    prior.labels = config.labels.copy()
    prior.min = [-1.0, -2.0, 3.0, -0.499, -1.0]
    prior.max = [2.0, 2.0, 10.0, 0.0, 3.0]

    # Preprocessing configuration
    config.preprocess = preprocess = ConfigDict()
    preprocess.vrange = (0, 1000)
    preprocess.vdisp_range = (0, 1000)
    preprocess.r_range = (0, 100.)
    preprocess.r_rstar_range = (0, 10.)
    preprocess.apply_projection = False
    preprocess.projection_axis = None
    preprocess.use_proper_motions = False

    ### MODEL CONFIGURATION ###
    config.model = model = ConfigDict()
    model.input_size = 3
    model.output_size = len(config.labels)

    # Embedding network configuration
    model.embedding = ConfigDict()
    model.embedding.type = 'transformer'  # 'gnn' or 'transformer'

    # Transformer embedding configuration
    model.embedding.transformer = ConfigDict()
    model.embedding.transformer.d_in = model.input_size
    model.embedding.transformer.d_model = 64
    model.embedding.transformer.d_mlp = 64
    model.embedding.transformer.n_layers = 4
    model.embedding.transformer.n_heads = 4
    model.embedding.transformer.concat_conditioning = False
    model.embedding.transformer.d_pos = None
    model.embedding.transformer.use_pos_enc = False
    model.embedding.transformer.pooling = 'mean'

    # MLP configuration (on top of transformer)
    model.embedding.mlp = ConfigDict()
    model.embedding.mlp.hidden_sizes = [64]
    model.embedding.mlp.output_size = 10
    model.embedding.mlp.act_name = 'relu'
    model.embedding.mlp.act_args = {}
    model.embedding.mlp.dropout = 0.0
    model.embedding.mlp.batch_norm = False

    # NPE Normalizing Flows configuration
    model.flows = ConfigDict()
    model.flows.num_transforms = 6
    model.flows.hidden_features = [64, 64]
    model.flows.num_bins = 8
    model.flows.activation = 'tanh'
    model.flows.randperm = True

    # SNPE-specific configuration
    model.num_atoms = 10  # Number of atoms for atomic loss computation

    # Pre-transformation configuration
    config.pre_transforms = pre_transforms = ConfigDict()
    pre_transforms.apply_graph = False  # Disable for Transformer
    pre_transforms.apply_projection = True
    pre_transforms.apply_selection = False
    pre_transforms.apply_uncertainty = True
    pre_transforms.use_log_features = True
    pre_transforms.projection_args = {'axis': 2}
    pre_transforms.uncertainty_args = {
        'distribution_type': 'jeffreys',
        'low': 0.01,
        'high': 20.0,
        'feature_idx': 1
    }

    ### OPTIMIZER AND SCHEDULER CONFIGURATION ###
    config.optimizer = optimizer = ConfigDict()
    optimizer.name = "AdamW"
    optimizer.lr = 5e-4
    optimizer.betas = [0.9, 0.999]
    optimizer.weight_decay = 0.01

    config.scheduler = scheduler = ConfigDict()
    scheduler.name = "WarmUpCosineAnnealingLR"
    scheduler.decay_steps = int(10_000 / 128 * 20)
    scheduler.warmup_steps = int(0.05 * scheduler.decay_steps)
    scheduler.eta_min = 1e-6
    scheduler.interval = 'step'
    scheduler.restart = False
    scheduler.T_mult = 1

    ### TRAINING CONFIGURATION ###
    config.num_rounds = 3  # Number of SNPE rounds
    config.num_epochs = 20
    config.num_steps = -1
    config.accelerator = 'auto'
    config.patience = 10
    config.gradient_clip_val = 1.0
    config.save_top_k = 3

    # Combined loss configuration (mix prior and proposal samples)
    config.use_combined_loss = False  # Enable combined loss for rounds > 0
    config.prior_fraction = 0.1  # Fraction of samples from prior (0.0 to 1.0)

    ### VISUALIZATION CONFIGURATION ###
    config.enable_visualization_callback = False  # In-training visualization
    config.visualization = ConfigDict()
    config.visualization.plot_every_n_epochs = 5
    config.visualization.n_posterior_samples = 1000
    config.visualization.n_val_samples = 100
    config.visualization.plot_median_v_true = True
    config.visualization.plot_tarp = True
    config.visualization.plot_rank = True
    config.visualization.use_default_mplstyle = True

    # Post-round visualization
    config.post_round_visualization = ConfigDict()
    config.post_round_visualization.enabled = True
    config.post_round_visualization.n_samples = 10_000  # Samples for corner plot
    config.post_round_visualization.save_plots = True
    config.post_round_visualization.log_to_wandb = True

    return config
