
import optuna
from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    # seeding
    config.seed_data = 1923871
    config.seed_training = 19318

    # data configuration
    config.data_root = '/mnt/ceph/users/tnguyen/jeans_gnn/datasets/processed_datasets/'
    config.data_name = 'gnfw_profiles/gnfw_beta_priorlarge_pois100'
    config.num_datasets = 1
    config.labels = (
        'dm_gamma', 'dm_log_r_dm', 'dm_log_rho_0',
        'df_beta0', 'df_log_r_a',
    )
    config.train_frac = 0.8
    config.num_workers = 0

    ## LOGGING AND WANDB CONFIGURATION ###
    config.workdir = '/mnt/ceph/users/tnguyen/jeans_gnn/trained_models-v2'
    config.name = 'example_embed_run'
    config.debug = False
    config.wandb_project = 'jgnn_v2.0_test'
    config.checkpoint = None
    config.reset = True
    config.reset_optimizer = True
    config.enable_progress_bar = True
    config.log_model = 'all'  # Log model checkpoints to WandB

    ### MODEL CONFIGURATION ###
    config.model = model = ConfigDict()
    model.input_size = 3

    # GNN configuration
    model.gnn = ConfigDict()
    model.gnn.projection_size = 32
    model.gnn.hidden_sizes = [64, ] * 3
    model.gnn.graph_layer = 'GATConv'
    model.gnn.graph_layer_params = ConfigDict()
    model.gnn.graph_layer_params.heads = 2
    model.gnn.graph_layer_params.concat = False
    model.gnn.pooling = 'mean'
    model.gnn.layer_norm = True
    model.gnn.norm_first = False
    model.gnn.act_name = 'leaky_relu'
    model.gnn.act_args = {'negative_slope': 0.01}

    # MLP configuration
    model.mlp = ConfigDict()
    model.mlp.hidden_sizes = [64, ] * 3
    model.mlp.output_size = 5
    model.mlp.dropout = 0.4
    model.mlp.batch_norm = True
    model.mlp.act_name = 'leaky_relu'
    model.mlp.act_args = {'negative_slope': 0.01}

    # Conditional MLP
    model.conditional_mlp = ConfigDict()
    model.conditional_mlp.input_size = 1
    model.conditional_mlp.hidden_sizes = [16, ] * 3
    model.conditional_mlp.output_size = 5
    model.conditional_mlp.dropout = 0.0
    model.conditional_mlp.batch_norm = True
    model.conditional_mlp.act_name = 'leaky_relu'
    model.conditional_mlp.act_args = {'negative_slope': 0.01}

    # Loss configuration
    model.loss_type = 'vmim'
    model.loss_args = ConfigDict()
    model.loss_args.features = 5
    model.loss_args.context_features = 5
    model.loss_args.num_transforms = 6
    model.loss_args.hidden_features = [64, 64]
    model.loss_args.num_bins = 8
    model.loss_args.activation = 'tanh'

    # Pre-transformation configuration
    # this is the transformation that is applied to the data before forward pass
    config.pre_transforms = pre_transforms = ConfigDict()
    pre_transforms.apply_projection = True
    pre_transforms.apply_selection = False
    pre_transforms.apply_uncertainty = True
    pre_transforms.use_log_features = True
    pre_transforms.uncertainty_args = {
        'distribution_type': 'jeffreys',
        'low': 0.01,
        'high': 20.0,
        'feature_idx': 1
    }
    pre_transforms.graph_name = 'KNN'
    pre_transforms.graph_args = {'k': 20, 'loop': True}

    ### OPTIMIZER AND SCHEDULER CONFIGURATION ###
    config.optimizer = optimizer = ConfigDict()
    optimizer.name = "AdamW"
    optimizer.lr = 1e-4
    optimizer.betas = [0.9, 0.999]
    optimizer.weight_decay = 0.01

    config.scheduler = scheduler = ConfigDict()
    scheduler.name = "WarmUpCosineAnnealingLR"
    scheduler.decay_steps = 10_000
    scheduler.warmup_steps = int(0.05 * scheduler.decay_steps)
    scheduler.eta_min = 1e-6
    scheduler.interval = 'step'

    ### TRAINING CONFIGURATION ###
    config.accelerator = 'gpu'
    config.train_batch_size = 128
    config.eval_batch_size = 128
    config.num_epochs = 20  # example run; set appropriately in real runs
    config.num_steps = -1
    config.patience = 15
    config.gradient_clip_val = 0.5
    config.save_top_k = 5

    return config
