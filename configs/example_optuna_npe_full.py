"""Comprehensive Optuna configuration for NPE hyperparameter optimization - V2 Style.

This config demonstrates the flexible search space system with many hyperparameters.
Use this as a template and customize the search_space for your needs.

Key Features:
- Parameters in `optuna.search_space` will be optimized by Optuna
- Parameters NOT in search_space will use their fixed values defined below
- Use dot notation to access any nested parameter (e.g., 'optimizer.lr')
- Easily move parameters between fixed and optimized by adding/removing from search_space

Usage:
    python jgnn/train_optuna_npe.py --config=jgnn/example_configs/example_optuna_npe_full.py
"""

import os
from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    # =========================================================================
    # Seeding
    # =========================================================================
    config.seed_data = 42
    config.seed_training = 24
    config.seed_optuna = 42

    # =========================================================================
    # Data Configuration
    # =========================================================================
    config.data_root = '/mnt/ceph/users/tnguyen/jeans_gnn/datasets/processed_datasets/'
    config.data_name = 'gnfw_profiles-v2/gnfw_beta_priorlarge_pois100'
    config.num_datasets = 20
    config.init = 0
    config.labels = (
        'dm_gamma', 'dm_log_r_dm', 'dm_log_rho_0',
        'df_beta0', 'df_log_r_a',
    )
    config.cond_labels = ('stellar_log_r_star',)
    config.train_frac = 0.9
    config.num_workers = 0

    # =========================================================================
    # Logging Configuration
    # =========================================================================
    config.workdir = '/mnt/ceph/users/tnguyen/jeans_gnn/optuna_runs/npe_full'
    config.name = None
    config.wandb_project = 'JGNN-Optuna-NPE'
    config.debug = False
    config.checkpoint = None
    config.reset_optimizer = False
    config.enable_progress_bar = False
    config.log_model = 'all'

    # Reuse norm_dict from embedding checkpoint
    config.reuse_embedding_norm_dict = False

    # =========================================================================
    # Optuna Configuration
    # =========================================================================
    config.optuna = ConfigDict()
    config.optuna.n_trials = 50
    config.optuna.n_startup_trials = 10  # Random trials before TPE sampler
    config.optuna.n_warmup_steps = 5     # Warmup steps for pruner
    config.optuna.interval_steps = 1     # Check pruning every N steps
    config.optuna.study_name = "npe_hyperopt_study"
    config.optuna.load_if_exists = True
    config.optuna.use_pruner = True
    config.optuna.timeout = None  # Optional timeout in seconds
    config.optuna.storage = "sqlite:///" + os.path.join(config.workdir, 'optuna_study.db')

    # =========================================================================
    # Search Space Definition
    # =========================================================================
    # Define which hyperparameters to optimize and their search spaces.
    #
    # Supported parameter types:
    #   - 'float': {'type': 'float', 'low': X, 'high': Y, 'log': True/False}
    #   - 'int': {'type': 'int', 'low': X, 'high': Y, 'step': Z, 'log': True/False}
    #   - 'categorical': {'type': 'categorical', 'choices': [A, B, C]}
    #
    # To OPTIMIZE a parameter: Add it to search_space
    # To KEEP FIXED: Remove from search_space and set the value below
    # =========================================================================

    config.optuna.search_space = ConfigDict()

    # --- Optimizer hyperparameters ---
    config.optuna.search_space['optimizer.lr'] = {
        'type': 'float',
        'low': 1e-5,
        'high': 1e-3,
        'log': True,  # Log scale for learning rate
    }
    config.optuna.search_space['optimizer.weight_decay'] = {
        'type': 'float',
        'low': 1e-6,
        'high': 1e-2,
        'log': True,
    }

    # --- Flow architecture ---
    config.optuna.search_space['model.flows.num_transforms'] = {
        'type': 'int',
        'low': 4,
        'high': 10,
    }
    config.optuna.search_space['model.flows.num_bins'] = {
        'type': 'categorical',
        'choices': [4, 8, 16],
    }
    # --- Flow hidden features (parameterized as size + num_layers) ---
    # Instead of optimizing the list directly, optimize two parameters:
    # 1. The size of each hidden layer
    # 2. The number of hidden layers
    # Then construct: hidden_features = [hidden_size] * num_layers
    #
    # Example: If hidden_size=64 and num_hidden_layers=2
    #          -> hidden_features = [64, 64]
    #
    # To KEEP FIXED: Comment out both parameters below and set
    #                config.model.flows.hidden_features = [64, 64] (at line ~188)
    config.optuna.search_space['model.flows.hidden_size'] = {
        'type': 'categorical',
        'choices': [32, 64, 128, 256],
    }
    config.optuna.search_space['model.flows.num_hidden_layers'] = {
        'type': 'int',
        'low': 1,
        'high': 3,
    }

    # --- Embedding MLP hidden sizes (parameterized as size + num_layers) ---
    # Same pattern: optimize size and num_layers separately
    # Then construct: hidden_sizes = [hidden_size] * num_hidden_layers
    #
    # Example: If hidden_size=128 and num_hidden_layers=3
    #          -> hidden_sizes = [128, 128, 128]
    config.optuna.search_space['model.embedding.mlp.hidden_size'] = {
        'type': 'categorical',
        'choices': [32, 64, 128, 256],
    }
    config.optuna.search_space['model.embedding.mlp.num_hidden_layers'] = {
        'type': 'int',
        'low': 1,
        'high': 4,
    }

    # --- Transformer architecture (if using transformer embedding) ---
    config.optuna.search_space['model.embedding.transformer.d_model'] = {
        'type': 'categorical',
        'choices': [32, 64, 128, 256],
    }

    config.optuna.search_space['model.embedding.transformer.d_mlp'] = {
        'type': 'categorical',
        'choices': [32, 64, 128, 256],
    }

    config.optuna.search_space['model.embedding.transformer.n_layers'] = {
        'type': 'int',
        'low': 2,
        'high': 6,
    }

    config.optuna.search_space['model.embedding.transformer.n_heads'] = {
        'type': 'categorical',
        'choices': [2, 4, 8],
    }

    # --- Embedding MLP output size ---
    config.optuna.search_space['model.embedding.mlp.output_size'] = {
        'type': 'categorical',
        'choices': [5, 10, 20, 32],
    }

    # =========================================================================
    # Model Configuration
    # =========================================================================
    config.model = ConfigDict()
    config.model.input_size = 3
    config.model.output_size = len(config.labels)

    model.embedding = ConfigDict()
    model.embedding.type = 'transformer'  # 'gnn' or 'transformer'
    model.embedding.transformer = ConfigDict()
    model.embedding.transformer.d_in = model.get_ref('input_size')
    model.embedding.transformer.d_model = 64
    model.embedding.transformer.d_mlp = 64
    model.embedding.transformer.n_layers = 4
    model.embedding.transformer.n_heads = 4
    model.embedding.transformer.d_cond = 1
    model.embedding.transformer.concat_conditioning = False
    model.embedding.transformer.d_pos = None  # No positional encoding
    model.embedding.transformer.use_pos_enc = False
    model.embedding.transformer.pooling = 'mean'  # 'mean', 'max', 'sum', 'cls'
    model.embedding.mlp.hidden_sizes = [64, ]
    model.embedding.mlp.output_size = 10
    model.embedding.mlp.act_name = 'relu'
    model.embedding.mlp.act_args = {}
    model.embedding.mlp.dropout = 0.0
    model.embedding.mlp.batch_norm = False

    # NPE Normalizing Flows configuration
    config.model.flows = ConfigDict()
    config.model.flows.hidden_features = [64, 64]
    config.model.flows.num_transforms = 6  # Will be optimized
    config.model.flows.num_bins = 8  # Will be optimized
    config.model.flows.activation = 'tanh'
    config.model.flows.randperm = True

    # Initialize NPE flows from embedding network flow (if embedding has a flow)
    config.model.init_flows_from_embedding = False

    # =========================================================================
    # Pre-Transformation Configuration (V2 Style)
    # =========================================================================
    config.pre_transforms = pre_transforms = ConfigDict()
    pre_transforms.apply_graph = False # disable graph construction for Transformer models
    pre_transforms.apply_projection = True
    pre_transforms.apply_selection = False
    pre_transforms.apply_uncertainty = True
    pre_transforms.use_log_features = True
    pre_transforms.projection_args = {
        'axis': 2
    }
    pre_transforms.uncertainty_args = {
        'distribution_type': 'jeffreys',
        'low': 0.01,
        'high': 20.0,
        'feature_idx': 1
    }  # ignore if apply_uncertainty is False

    # =========================================================================
    # Optimizer Configuration
    # =========================================================================
    config.optimizer = ConfigDict()
    config.optimizer.name = "AdamW"
    config.optimizer.lr = 5e-4  # Will be optimized
    config.optimizer.betas = [0.9, 0.999]
    config.optimizer.weight_decay = 1e-3  # Will be optimized

    # =========================================================================
    # Scheduler Configuration
    # =========================================================================
    config.scheduler = ConfigDict()
    config.scheduler.name = "WarmUpCosineAnnealingLR"
    config.scheduler.decay_steps = 20_000
    config.scheduler.warmup_steps = 1_000
    config.scheduler.eta_min = 1e-6
    config.scheduler.interval = 'step'
    config.scheduler.restart = False
    config.scheduler.T_mult = 1

    # =========================================================================
    # Training Configuration
    # =========================================================================
    config.accelerator = 'gpu'
    config.train_batch_size = 128
    config.eval_batch_size = 256
    config.num_epochs = 50
    config.num_steps = -1
    config.patience = 20
    config.gradient_clip_val = 0.5
    config.save_top_k = 3

    # =========================================================================
    # Visualization Configuration
    # =========================================================================
    config.enable_visualization_callback = True
    config.visualization = ConfigDict()
    config.visualization.plot_every_n_epochs = 1
    config.visualization.n_posterior_samples = 500
    config.visualization.n_val_samples = 1000
    config.visualization.plot_median_v_true = True
    config.visualization.plot_tarp = True
    config.visualization.plot_rank = True

    return config