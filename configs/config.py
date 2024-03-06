
from ml_collections import ConfigDict

def get_config():
    config = ConfigDict()

    config.seed = 10

    # data configuration
    config.data_root = '/mnt/ceph/users/tnguyen/jeans_gnn/datasets/processed_datasets'
    config.data_name = 'gnfw_profiles/gnfw_beta_priorlarge_pois100_full'
    config.is_directory = True
    config.num_datasets = 1
    config.labels = (
        'dm_gamma', 'dm_log_r_dm', 'dm_log_rho_0',
        'df_beta0', 'df_log_r_a','stellar_log_r_star'
    )

    # logging configuration
    config.workdir = '/mnt/ceph/users/tnguyen/jeans_gnn/trained_models'
    config.progress_bar_refresh_rate = 1000
    config.resume = False

    # training configuration
    config.train_frac = 0.8
    config.train_batch_size = 128
    config.num_workers = 1
    config.eval_batch_size = 128
    config.num_posterior_samples = 2000

    # model configuration
    config.model = model = ConfigDict()
    model.input_size = 2
    model.output_size = len(config.labels)
    model.featurizer = ConfigDict()
    model.featurizer.name = 'gnn'
    model.featurizer.activation = ConfigDict()
    model.featurizer.activation.name = 'relu'
    model.featurizer.hidden_sizes = [128, 128, 128, 128]
    model.featurizer.graph_layer = 'ChebConv'
    model.featurizer.graph_layer_params = ConfigDict()
    model.featurizer.graph_layer_params.K = 4
    model.featurizer.pooling = 'mean'
    model.featurizer.layer_norm = True
    model.featurizer.norm_first = False
    model.mlp = ConfigDict()
    model.mlp.activation = ConfigDict()
    model.mlp.activation.name = 'relu'
    model.mlp.hidden_sizes = [128, 64]
    model.mlp.output_size = 32
    model.flows = ConfigDict()
    model.flows.name = 'maf'
    model.flows.hidden_size = 64
    model.flows.num_blocks = 2
    model.flows.num_layers = 4
    model.flows.batch_norm = True
    model.flows.activation = ConfigDict()
    model.flows.activation.name = 'tanh'

    # pre-transformation configuration
    model.pre_transform = ConfigDict()
    model.pre_transform.graph_name = 'KNN'
    model.pre_transform.graph_params = ConfigDict()
    model.pre_transform.graph_params.k = 20
    model.pre_transform.graph_params.loop = True

    # optimizer and scheduler configuration
    config.optimizer = optimizer = ConfigDict()
    optimizer.name = "AdamW"
    optimizer.lr = 5e-4
    optimizer.betas = [0.9, 0.999]
    optimizer.weight_decay = 0.01
    config.scheduler = scheduler = ConfigDict()
    scheduler.name = None
    # scheduler.name = "ReduceLROnPlateau"
    # scheduler.factor = 0.5
    # scheduler.patience = 10
    # scheduler.interval = "epoch"
    # scheduler.name = "WarmUpCosineAnnealingLR"
    # scheduler.decay_steps = 100_000
    # scheduler.warmup_steps = 5_000
    # scheduler.name = "CosineAnnealingLR"
    # scheduler.T_max = 100_000
    # scheduler.eta_min = 1e-6
    # scheduler.interval = "step"
cl
    # training loop configuration
    config.num_epochs = 1_000
    config.patience = 1_000
    config.gradient_clip_val = 0.5
    config.save_top_k = 20
    config.accelerator = 'gpu'
    config.strategy = 'auto'
    config.monitor = 'val_loss'
    config.mode = 'min'
    config.num_nodes = 1
    config.devices = 1
    config.overfit_batches = 0  # debug option, leave at 0.0 in production runs

    return config
