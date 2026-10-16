"""Example NPE training config for ICRS sky-plane datasets.

Produced by sample_galaxies_target.py (target-matched mock catalogs).

Key differences from Cartesian configs
---------------------------------------
dataset_type  = 'icrs'      — uses jgnn.datasets.icrs dataloader
is_directory  = False       — single HDF5 file (not a sharded directory)
input_size    = 2           — node features are [log10(R_proj), vlos]
apply_projection  = False   — already in projected sky plane
apply_uncertainty = False   — observed vlos_err already present in data
"""

import numpy as np
from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder


def get_config():
    config = ConfigDict()

    # -------------------------------------------------------------------------
    # Seeding
    # -------------------------------------------------------------------------
    config.seed_data = 42
    config.seed_training = np.random.randint(0, 1_000_000)

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    # is_directory=False → reads root/data_name.hdf5 directly
    config.dataset_type = 'icrs'
    config.is_directory = False
    config.data_root = '/mnt/ceph/users/tnguyen/jeans_gnn/datasets/raw_datasets'
    config.data_name = 'default_draco1_gnfw_beta_target_config'
    config.num_datasets = 1  # unused for single-file mode, kept for API compat
    config.train_frac = 0.9
    config.num_workers = 0

    # Parameters to infer (theta).  These must be keys in graph_features.
    config.labels = (
        'dm_gamma',
        'dm_r_dm',
        'dm_rho_0',
        'df_beta0',
        'df_log_r_a',
    )

    # Conditioned-on observable summary (optional).
    # stellar_r_star = rhalf drawn per galaxy; log_mwolf = Wolf mass draw.
    config.cond_labels = ('stellar_log_r_star', )

    # -------------------------------------------------------------------------
    # Logging / WandB
    # -------------------------------------------------------------------------
    config.workdir = '/mnt/ceph/users/tnguyen/jeans_gnn/trained_models/npe-v3/draco1_icrs'
    config.wandb_project = 'draco1_icrs_npe'
    config.entity = 'sbi_dsph'
    config.name = placeholder(str)
    config.id = placeholder(str)
    config.tags = ['npe', 'icrs', 'draco1']
    config.checkpoint = placeholder(str)
    config.reset_optimizer = False
    config.debug = True
    config.enable_progress_bar = True
    config.reuse_embedding_norm_dict = False

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    config.model = model = ConfigDict()
    # input_size=2: [log10(R_proj), vlos] — built by datasets.icrs
    model.input_size = 2
    model.output_size = len(config.labels)

    # Embedding — GNN on pos=(ra, dec) with node features x=[log10 R_proj, vlos]
    model.embedding = ConfigDict()
    model.embedding.type = 'gnn'

    model.embedding.gnn = ConfigDict()
    model.embedding.gnn.graph_layer = 'ChebConv'
    model.embedding.gnn.graph_layer_params = {'K': 8}
    model.embedding.gnn.hidden_sizes = [128] * 5
    model.embedding.gnn.act_name = 'relu'
    model.embedding.gnn.pooling = 'mean'
    model.embedding.gnn.layer_norm = False
    model.embedding.gnn.norm_first = False

    model.embedding.mlp = ConfigDict()
    model.embedding.mlp.hidden_sizes = [128]
    model.embedding.mlp.output_size = 128
    model.embedding.mlp.act_name = 'relu'
    model.embedding.mlp.dropout = 0.0

    # Conditional MLP — encodes cond_labels (e.g. stellar_r_star)
    model.embedding.conditional_mlp = ConfigDict()
    model.embedding.conditional_mlp.input_size = len(config.cond_labels)
    model.embedding.conditional_mlp.hidden_sizes = [128]
    model.embedding.conditional_mlp.output_size = 128
    model.embedding.conditional_mlp.act_name = 'relu'

    # Normalizing flows (NSF)
    model.flows = ConfigDict()
    model.flows.type = 'nsf'
    model.flows.num_transforms = 6
    model.flows.hidden_features = [128, 128]
    model.flows.activation = 'tanh'
    model.flows.num_bins = 8
    model.flows.randperm = True

    # -------------------------------------------------------------------------
    # Pre-transforms
    # Note: projection and uncertainty augmentation are OFF — the data is already
    # on the sky plane and carries observed velocity errors.
    # -------------------------------------------------------------------------
    config.pre_transforms = pre_transforms = ConfigDict()
    pre_transforms.apply_graph = True          # build knn graph on pos=(ra, dec)
    pre_transforms.apply_projection = False    # already projected
    pre_transforms.apply_selection = False     # no dropout augmentation
    pre_transforms.apply_uncertainty = False   # vlos_err already in the data
    pre_transforms.use_log_features = False    # x=[log10 R_proj, vlos] already set
    pre_transforms.graph_name = 'adaptive_knn'
    pre_transforms.graph_args = {'ratio': 0.2, 'loop': True}

    # -------------------------------------------------------------------------
    # Visualization callbacks
    # -------------------------------------------------------------------------
    config.visualization = visualization = ConfigDict()
    visualization.enabled = True

    # Simulation diagnostics (TARP / rank / median-vs-true on val set)
    visualization.n_posterior_samples = 500
    visualization.n_val_samples = 200
    visualization.plot_every_n_epochs = 1
    visualization.plot_tarp = True
    visualization.plot_median_v_true = True
    visualization.plot_rank = True
    visualization.use_default_mplstyle = True

    # Target posterior — corner plot on real observed data each epoch.
    # Remove or set visualization.target = None to disable.
    # cond_values are auto-derived from meta; override only if needed.
    visualization.target = target_vis = ConfigDict()
    target_vis.catalog_path = (
        '/mnt/home/tnguyen/projects/jeans_gnn/dsph_datasets/vr_catalogs/'
        'walker23-ting/draco_all_3000_walker_04012025.csv')
    target_vis.meta_key = 'draco_1'
    target_vis.source = 'walker23'
    target_vis.loader_kwargs = ConfigDict({
        'target_system': 'Draco_1',
        'mem_prob_min': 0.8,
        'vlos_abs_max': 50.0,
        'apply_perspective_corr': True,
    })
    target_vis.n_posterior_samples = 2000
    target_vis.plot_every_n_epochs = 1
    target_vis.param_names = (
        r'$\gamma$',
        r'$r_\mathrm{dm}$ [kpc]',
        r'$\rho_0$ [M$_\odot$ kpc$^{-3}$]',
        r'$\beta_0$',
        r'$\log r_a / r_\star$',
    )

    # -------------------------------------------------------------------------
    # Optimizer and scheduler
    # -------------------------------------------------------------------------
    config.optimizer = optimizer = ConfigDict()
    optimizer.name = 'AdamW'
    optimizer.lr = 5e-4
    optimizer.betas = [0.9, 0.999]
    optimizer.weight_decay = 0.01

    config.scheduler = scheduler = ConfigDict()
    scheduler.name = 'WarmUpCosineAnnealingLR'
    scheduler.decay_steps = 200_000
    scheduler.warmup_steps = int(0.05 * scheduler.decay_steps)
    scheduler.eta_min = 1e-6
    scheduler.interval = 'step'
    scheduler.restart = False
    scheduler.T_mult = 1

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    config.accelerator = 'gpu'
    config.train_batch_size = 64
    config.eval_batch_size = 64
    config.num_epochs = -1
    config.num_steps = scheduler.decay_steps
    config.patience = 50
    config.gradient_clip_val = 0.5

    return config
