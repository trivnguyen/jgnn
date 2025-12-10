"""Minimal test configuration for Optuna NPE optimization - V2 Style.

This is a quick test config to verify the Optuna implementation works correctly.
It uses:
- Small number of trials (3)
- Small model architecture
- Few training epochs
- Minimal search space (just 2 parameters)

Use this to test before running a full optimization.

Usage:
    python jgnn/train_optuna_npe.py --config=jgnn/example_configs/example_optuna_npe.py
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
    config.data_name = 'gnfw_profiles/gnfw_beta_priorlarge_pois100'
    config.num_datasets = 1  # Small for testing
    config.init = 0
    config.labels = (
        'dm_gamma', 'dm_log_r_dm', 'dm_log_rho_0',
        'df_beta0', 'df_log_r_a',
    )
    config.cond_labels = None
    config.train_frac = 0.9
    config.num_workers = 0

    # =========================================================================
    # Logging Configuration
    # =========================================================================
    config.workdir = './test_optuna_npe'
    config.name = 'test_optuna'
    config.wandb_project = 'jgnn-npe-optuna-test'
    config.debug = True  # Set to True to disable wandb
    config.checkpoint = None  # Path to NPE checkpoint for resuming
    config.reset_optimizer = False
    config.enable_progress_bar = True
    config.log_model = 'all'

    # Reuse norm_dict from embedding checkpoint
    config.reuse_embedding_norm_dict = True

    # =========================================================================
    # Optuna Configuration - MINIMAL FOR TESTING
    # =========================================================================
    config.optuna = ConfigDict()
    config.optuna.n_trials = 3  # Just 3 trials for quick test
    config.optuna.n_startup_trials = 2
    config.optuna.n_warmup_steps = 2
    config.optuna.interval_steps = 1
    config.optuna.study_name = "npe_test_study"
    config.optuna.load_if_exists = True
    config.optuna.use_pruner = True
    config.optuna.timeout = None
    config.optuna.storage = "sqlite:///" + os.path.join(config.workdir, 'test_optuna.db')

    # =========================================================================
    # Search Space - MINIMAL (just 2 parameters to test)
    # =========================================================================
    config.optuna.search_space = ConfigDict()

    # Test 1: Optimize learning rate (log scale)
    config.optuna.search_space['optimizer.lr'] = {
        'type': 'float',
        'low': 1e-4,
        'high': 1e-3,
        'log': True,
    }

    # Test 2: Optimize number of flow transforms
    config.optuna.search_space['model.flows.num_transforms'] = {
        'type': 'int',
        'low': 3,
        'high': 6,
    }

    # =========================================================================
    # Model Configuration - SMALL FOR TESTING
    # =========================================================================
    config.model = ConfigDict()
    config.model.input_size = 3
    config.model.output_size = len(config.labels)
    config.model.init_flows_from_embedding = False

    # Embedding network configuration
    # Set to checkpoint path to load pre-trained embedding, or None to train from scratch
    config.model.embedding_checkpoint = None  # Example: '/path/to/embedding.ckpt'
    config.model.freeze_embedding = False

    # NPE Normalizing Flows configuration
    config.model.flows = ConfigDict()
    config.model.flows.num_transforms = 4  # Will be optimized
    config.model.flows.hidden_features = [64, 64]  # V2 style: list of sizes
    config.model.flows.num_bins = 8
    config.model.flows.activation = 'tanh'
    config.model.flows.randperm = True

    # =========================================================================
    # Pre-Transformation Configuration (V2 Style)
    # =========================================================================
    config.pre_transforms = ConfigDict()
    config.pre_transforms.apply_projection = True
    config.pre_transforms.apply_selection = False
    config.pre_transforms.apply_uncertainty = True
    config.pre_transforms.use_log_features = True

    config.pre_transforms.uncertainty_args = {
        'distribution_type': 'jeffreys',
        'low': 0.01,
        'high': 20.0,
        'feature_idx': 1
    }

    config.pre_transforms.graph_name = 'KNN'
    config.pre_transforms.graph_args = {'k': 20, 'loop': True}

    # =========================================================================
    # Optimizer Configuration
    # =========================================================================
    config.optimizer = ConfigDict()
    config.optimizer.name = "AdamW"
    config.optimizer.lr = 5e-4  # Will be optimized
    config.optimizer.betas = [0.9, 0.999]
    config.optimizer.weight_decay = 1e-2

    # =========================================================================
    # Scheduler Configuration
    # =========================================================================
    config.scheduler = ConfigDict()
    config.scheduler.name = None  # Disable for quick test
    # config.scheduler.name = "WarmUpCosineAnnealingLR"
    # config.scheduler.decay_steps = 10_000
    # config.scheduler.warmup_steps = 500
    # config.scheduler.eta_min = 1e-6
    # config.scheduler.interval = 'step'

    # =========================================================================
    # Training Configuration - FAST FOR TESTING
    # =========================================================================
    config.train_batch_size = 128
    config.eval_batch_size = 256
    config.num_epochs = 5  # Very few epochs for quick test
    config.num_steps = -1
    config.patience = 10  # Won't trigger in 5 epochs
    config.gradient_clip_val = 0.5
    config.save_top_k = 3

    # =========================================================================
    # Hardware Configuration
    # =========================================================================
    config.accelerator = 'gpu'

    # =========================================================================
    # Visualization Configuration
    # =========================================================================
    config.enable_visualization_callback = False  # Disable for quick test

    config.visualization = ConfigDict()
    config.visualization.plot_every_n_epochs = 10
    config.visualization.n_posterior_samples = 100
    config.visualization.n_val_samples = 50
    config.visualization.plot_median_v_true = False
    config.visualization.plot_tarp = False
    config.visualization.plot_rank = False

    return config


# =============================================================================
# Expected Output
# =============================================================================
#
# This test should:
# 1. Create 3 trials with different hyperparameters
# 2. Each trial trains for 5 epochs
# 3. Return validation loss for each trial
# 4. Select best trial
# 5. Save results to ./test_optuna_npe/
#
# To verify it's working:
# - Check that trials have different learning rates (1e-4 to 1e-3)
# - Check that trials have different num_transforms (3 to 6)
# - Check that best_hyperparameters.txt is created
# - Check that other parameters stay fixed
#
# =============================================================================
