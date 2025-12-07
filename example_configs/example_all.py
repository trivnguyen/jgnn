
import optuna
from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    # seeding
    config.seed_data = 42
    config.seed_training = 24

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
    config.workdir = './example'
    config.wandb_project = 'jgnn_v2.0_example'
    config.debug = False
    config.checkpoint = None  # Path to NPE checkpoint for resuming
    config.reset = True
    config.reset_optimizer = True
    config.enable_progress_bar = True
    config.log_model = 'all'  # Log model checkpoints to WandB

    # Reuse norm_dict from embedding checkpoint
    # if import embedding_nn from a pre-trained model, this should always be True
    config.reuse_embedding_norm_dict = False

    ### MODEL CONFIGURATION ###
    config.model = model = ConfigDict()
    model.input_size = 3
    model.output_size = len(config.labels)  # Number of parameters to infer (5)

    # Embedding network configuration
    model.embedding = ConfigDict()
    model.embedding.gnn = ConfigDict()
    model.embedding.gnn.projection_size = 32
    model.embedding.gnn.hidden_sizes = [64, ] * 3
    model.embedding.gnn.graph_layer = 'GATConv'
    model.embedding.gnn.graph_layer_params = ConfigDict()
    model.embedding.gnn.graph_layer_params.heads = 2
    model.embedding.gnn.graph_layer_params.concat = False
    model.embedding.gnn.pooling = 'mean'
    model.embedding.gnn.layer_norm = True
    model.embedding.gnn.norm_first = False
    model.embedding.gnn.act_name = 'leaky_relu'
    model.embedding.gnn.act_args = {'negative_slope': 0.01}

    # MLP configuration
    model.embedding.mlp = ConfigDict()
    model.embedding.mlp.hidden_sizes = [64, ] * 3
    model.embedding.mlp.output_size = 5
    model.embedding.mlp.dropout = 0.4
    model.embedding.mlp.batch_norm = True
    model.embedding.mlp.act_name = 'leaky_relu'
    model.embedding.mlp.act_args = {'negative_slope': 0.01}

    # Conditional MLP
    model.embedding.conditional_mlp = ConfigDict()
    model.embedding.conditional_mlp.input_size = 1
    model.embedding.conditional_mlp.hidden_sizes = [16, ] * 3
    model.embedding.conditional_mlp.output_size = 5
    model.embedding.conditional_mlp.dropout = 0.0
    model.embedding.conditional_mlp.batch_norm = True
    model.embedding.conditional_mlp.act_name = 'leaky_relu'
    model.embedding.conditional_mlp.act_args = {'negative_slope': 0.01}

    # NPE Normalizing Flows configuration
    model.flows = ConfigDict()
    model.flows.num_transforms = 6
    model.flows.hidden_features = [64, 64]
    model.flows.num_bins = 8
    model.flows.activation = 'tanh'
    model.flows.randperm = True

    # Pre-transformation configuration
    # Note: For NPE, pre_transforms are passed to NPE, not to embedding_nn
    config.pre_transforms = pre_transforms = ConfigDict()
    pre_transforms.apply_graph = True
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
    pre_transforms.graph_name = 'KNN'
    pre_transforms.graph_args = {'k': 20, 'loop': True}

    ### VISUALIZATION CALLBACK CONFIGURATION ###
    config.enable_visualization_callback = True
    config.visualization = visualization = ConfigDict()
    visualization.n_posterior_samples = 500
    visualization.n_val_samples = 1000
    visualization.plot_every_n_epochs = 1
    visualization.plot_tarp = True
    visualization.plot_median_v_true = True
    visualization.plot_rank = True

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
    config.num_epochs = 10
    config.num_steps = -1
    config.patience = 15
    config.gradient_clip_val = 0.5
    config.save_top_k = 5

    return config
