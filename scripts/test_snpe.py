import os

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'

import shutil
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import wandb
import ml_collections
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
import torch
from torch_geometric.data import Data, Batch
from absl import flags
from ml_collections import config_flags
from ml_collections.config_dict import ConfigDict

from jgnn import datasets
from jgnn.datasets.preprocess import create_graph_from_posvel
from jgnn.models import SequentialNPE, GNNEmbedding, TransformerEmbedding
from jgnn.transforms import build_transformation
from jgnn.callbacks.visualization import NPEVisualizationCallback
from jgnn.priors import BoxUniform
from jgnn.sims import (
    run_simulation_batch,
    preprocess_simulations,
    samples_to_simulation_params
)

def get_config():
    config = ConfigDict()

    config.seed_data = 4042
    config.seed = 2131
    config.workdir = '/mnt/ceph/users/tnguyen/jeans_gnn/runs'

    config.labels = [
        'dm_gamma', 'dm_log_r_dm', 'dm_log_rho_0',
        'df_beta0', 'df_log_r_a_r_star',
    ]
    # config.cond_labels = ['stellar_log_r_star',]
    config.train_frac = 0.9
    config.train_batch_size = 128
    config.eval_batch_size = 256
    config.num_workers = 0

    ### OBSERVATION (for SNPE) ###
    config.use_observation = True
    config.observation = observation = ConfigDict()
    observation.path = '/mnt/home/tnguyen/projects/jeans_gnn/datasets/processed_data/'\
        'draco_desi_p0.80.csv'
    observation.key = 'draco_1'
    observation.prob_threshold = 0.8

    meta = pd.read_csv('/mnt/home/tnguyen/projects/jeans_gnn/datasets/tables/dwarfs.csv')
    rstar = meta[meta.key=='draco_1'].rhalf_sph_physical.values[0] / 1000
    observation.meta = {
        'stellar_log_r_star': np.log10(rstar)
    }

    ### SIMULATION AND PRIOR CONFIGURATION (for SNPE) ###
    config.simulation = simulation = ConfigDict()
    simulation.dm_type = 'Spheroid'          # Dark matter potential type
    simulation.stellar_type = 'Plummer'      # Stellar density profile type
    simulation.df_type = 'QuasiSpherical'   # Distribution function type
    simulation.num_stars_dist = 'poisson'  # Options: 'poisson', 'uniform', 'delta'
    simulation.num_stars_mean = 100        # Mean for Poisson
    simulation.num_galaxies = 50_000   # per round of simulation
    simulation.max_iter = 1000
    simulation.use_multiprocessing = False
    simulation.n_jobs = 1
    simulation.dm_params_default = {'alpha': 1.0, 'beta': 3.0}  # generalized NFW
    simulation.stellar_params_default = {'r_star': rstar}
    simulation.df_params_default = {}

    config.prior = prior = ConfigDict()
    prior.labels = [
        'dm_gamma', 'dm_log_r_dm', 'dm_log_rho_0',
        'df_beta0', 'df_log_r_a_r_star',
    ]
    prior.min = [-1.0, np.log10(rstar), 3.0, -0.499, -1.0]
    prior.max = [2.0, np.log10(rstar) + 2.0, 10.0, 0.999, 3.0]

    config.preprocess = preprocess = ConfigDict()
    preprocess.vrange = (0, 1000)         # 3D velocity range in km/s
    preprocess.vdisp_range = (0, 1000)    # Velocity dispersion range
    preprocess.r_range = (0, 100.)        # Radius range in kpc
    preprocess.r_rstar_range = (0, 10.)  # Radius range in units of r_star
    preprocess.apply_projection = False      # Apply 2D projection
    preprocess.projection_axis = None       # Random projection (or 0, 1, 2 for fixed axis)
    preprocess.use_proper_motions = False   # Include proper motions

    ### MODEL CONFIGURATION ###
    config.model = model = ConfigDict()
    model.input_size = 3
    model.output_size = len(config.labels)  # Number of parameters to infer (5)

    # Embedding network configuration
    model.embedding = ConfigDict()
    model.embedding.type = 'transformer'  # 'gnn' or 'transformer'

    # Transformer configuration
    model.embedding.transformer = ConfigDict()
    model.embedding.transformer.d_in = 3
    model.embedding.transformer.d_model = 64
    model.embedding.transformer.d_mlp = 64
    model.embedding.transformer.n_layers = 4
    model.embedding.transformer.n_heads = 4
    model.embedding.transformer.d_cond = None
    model.embedding.transformer.concat_conditioning = False
    model.embedding.transformer.d_pos = None  # No positional encoding
    model.embedding.transformer.use_pos_enc = False
    model.embedding.transformer.pooling = 'mean'  # 'mean', 'max', 'sum', 'cls'
    model.embedding.mlp = ConfigDict()
    model.embedding.mlp.hidden_sizes = [64, ]
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

    # Pre-transformation configuration
    # Note: For NPE, pre_transforms are passed to NPE, not to embedding_nn
    config.pre_transforms = pre_transforms = ConfigDict()
    pre_transforms.apply_graph = False # disable graph construction for Transformer models
    pre_transforms.apply_projection = True
    pre_transforms.apply_selection = False
    pre_transforms.apply_uncertainty = True
    pre_transforms.use_log_features = True
    pre_transforms.projection_args = {'axis': 2}
    pre_transforms.uncertainty_args = {
        'distribution_type': 'jeffreys',
        'low': 0.01,
        'high': 10.0,
        'feature_idx': 1
    }

    ### OPTIMIZER AND SCHEDULER CONFIGURATION ###
    config.optimizer = optimizer = ConfigDict()
    optimizer.name = "AdamW"
    optimizer.lr = 1e-3
    optimizer.betas = [0.9, 0.999]
    optimizer.weight_decay = 0.01

    config.scheduler = scheduler = ConfigDict()
    scheduler.name = "WarmUpCosineAnnealingLR"
    scheduler.decay_steps = int(50_000 / 128 * 50)  # num_epochs estimated
    scheduler.warmup_steps = int(0.05 * scheduler.decay_steps)
    scheduler.eta_min = 1e-6
    scheduler.interval = 'step'
    scheduler.restart = True
    scheduler.T_mult = 1

    ### TRAINING CONFIGURATION ###
    config.num_rounds = 10  # Number of SNPE rounds
    config.num_epochs = -1
    config.num_steps = scheduler.decay_steps
    config.accelerator = 'auto'
    config.patience = 100
    config.gradient_clip_val = 1.0
    config.enable_progress_bar = False
    config.debug = False

    ### WANDB CONFIGURATION ###
    config.wandb_project = "JGNN-SNPE-Test"
    config.name = 'Test-Draco'
    config.id = None

    ### VISUALIZATION CALLBACK CONFIGURATION ###
    config.enable_visualization_callback = False

    return config

config = get_config()

print(f"[Config] Multiprocessing: {config.simulation.use_multiprocessing}")
if config.simulation.use_multiprocessing:
    if config.simulation.n_jobs is None:
        import multiprocessing
        print(f"[Config] Using all {multiprocessing.cpu_count()} CPUs")
    else:
        print(f"[Config] Using {config.simulation.n_jobs} CPUs")
else:
    print(f"[Config] Sequential mode (no multiprocessing)")

def read_observation(
    config: ml_collections.ConfigDict, norm_dict=None) -> Batch:
    """ Read observation data from CSV file and apply membership probability cut.

    Returns a batch containing a single observation (the dwarf galaxy with all its stars).
    """
    df = pd.read_csv(config.observation.path)

    # apply membership probability cut
    mem_prob = df['mem_prob'].values.astype('float32')
    mask = mem_prob >= config.observation.get('prob_threshold', 0.8)
    if not mask.any():
        raise ValueError("No stars pass the membership probability threshold.")
    df = df[mask]

    # extract relevant columns
    vr_sys = df['vr_sys'].values.astype('float32')
    vr = df['vr'].values.astype('float32')
    vr_err = df['vr_err'].values.astype('float32')
    ra = df['RA'].values.astype('float32')
    dec = df['DEC'].values.astype('float32')
    R_kin = df['R_kin'].values.astype('float32')
    log_R_kin = np.log10(R_kin + 1e-8).astype('float32')  # avoid log(0)

    vr = vr - vr_sys  # correct for systemic velocity

    print(f"[Data] Loaded observation from {config.observation.path} with {len(df)} stars.")
    print(f"[Data]   Systemic velocity: {vr_sys[0]:.2f} km/s")
    print(f"[Data]   Velocity range: {vr.min():.2f} to {vr.max():.2f} km/s")
    print(f"[Data]   Velocity error range: {vr_err.min():.2f} to {vr_err.max():.2f} km/s")
    print(f"[Data]   Radius range: {R_kin.min():.2f} to {R_kin.max():.2f} kpc")

    if config.get('cond_labels') is not None:
        # get conditioning labels from metadata
        cond = []
        for label in config.cond_labels:
            cond.append(config.observation.meta['stellar_log_r_star'])
        cond = np.array(cond, dtype='float32')

        if norm_dict is not None:
            cond = (cond - norm_dict['cond_loc']) / norm_dict['cond_scale']
    else:
        cond = None

    # create a graph object (single observation)
    graph = Data(
        x=torch.tensor([log_R_kin, vr, vr_err]).T,
        pos=torch.tensor([ra, dec]).T,
        cond=torch.tensor(cond).unsqueeze(0) if cond is not None else None,
    )

    batch = Batch.from_data_list([graph])  # create a batch with a single graph (batch_size=1)
    return batch

def create_embedding_network(config: ml_collections.ConfigDict):
    """Create a new embedding network.

    Args:
        config: Configuration dictionary

    Returns:
        Embedding network instance
    """
    model_type = config.model.embedding.get('type', 'gnn')
    if model_type == 'gnn':
        print("[Model] Creating GNN Embedding model...")
        return GNNEmbedding(
            input_size=config.model.input_size,
            gnn_args=config.model.embedding.gnn,
            mlp_args=config.model.embedding.mlp,
            loss_type=config.model.embedding.get('loss_type', 'mse'),
            loss_args=config.model.embedding.get('loss_args', None),
            conditional_mlp_args=config.model.embedding.get('conditional_mlp', None),
        )
    elif model_type == 'transformer':
        print("[Model] Creating Transformer Embedding model...")
        return TransformerEmbedding(
            input_size=config.model.input_size,
            transformer_args=config.model.embedding.transformer,
            loss_type=config.model.embedding.get('loss_type', 'mse'),
            loss_args=config.model.embedding.get('loss_args', None),
            mlp_args=config.model.embedding.get('mlp', None),
        )
    else:
        raise ValueError(f"Unsupported embedding model type: {config.model.type}")

def create_model(
    config: ml_collections.ConfigDict,
    pre_transforms,
    norm_dict,
    proposal=None,
    prior=None,
    current_round=0
) -> SequentialNPE:
    """Create the NPE model with optional pre-trained embedding network.

    Args:
        config: Configuration dictionary
        pre_transforms: Pre-transformation pipeline (passed to NPE)
        norm_dict: Normalization dictionary to pass to NPE
        proposal: Proposal distribution (posterior from previous round)
        current_round: Current SNPE round

    Returns:
        NPE model instance and prior
    """
    # Check if we should load a pre-trained embedding network
    embedding_checkpoint = config.model.get('embedding_checkpoint', None)
    freeze_embedding = config.model.get('freeze_embedding', False)

    if embedding_checkpoint is not None:
        # Load pre-trained embedding network
        embedding_nn, _ = load_embedding_network(
            embedding_checkpoint,
            freeze=freeze_embedding
        )
    else:
        # Create new embedding network
        print(f"[Model] Creating new embedding network for round {current_round}...")
        embedding_nn = create_embedding_network(config)

    # Create NPE model
    # Note: pre_transforms goes to NPE, not embedding_nn
    print(f"[Model] Creating NPE model for round {current_round}...")
    if proposal is not None:
        print(f"[Model] Using posterior from round {current_round-1} as proposal")

    model = SequentialNPE(
        input_size=config.model.input_size,
        output_size=config.model.output_size,
        flows_args=config.model.flows,
        embedding_nn=embedding_nn,
        optimizer_args=config.optimizer,
        scheduler_args=config.scheduler,
        norm_dict=norm_dict,
        pre_transforms=pre_transforms,
        prior=prior,
        proposal=proposal,
        current_round=current_round,
        num_atoms=32
    )

    return model

def create_callbacks(config: ml_collections.ConfigDict, wandb_logger: WandbLogger) -> list:
    """Create PyTorch Lightning callbacks.

    Args:
        config: Configuration dictionary
        wandb_logger: WandB logger instance (used to get checkpoint directory)

    Returns:
        List of callback instances
    """
    # default callbacks
    callbacks = [
        EarlyStopping(
            monitor='val/loss',
            mode='min',
            patience=config.patience,
            verbose=True
        ),
        ModelCheckpoint(
            filename="epoch={epoch}-step={step}-loss={val/loss:.4f}",
            monitor='val/loss',
            mode='min',
            save_top_k=3,  # saves last 3 best checkpoints
            save_weights_only=False,
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(
            filename="last",
            save_weights_only=False,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    if config.get('enable_visualization_callback', True):
        print("[Callbacks] Adding NPE Visualization Callback")
        callbacks.append(
            NPEVisualizationCallback(
                plot_every_n_epochs=config.visualization.get('plot_every_n_epochs', 1),
                n_posterior_samples=config.visualization.get('n_posterior_samples', 1000),
                n_val_samples=config.visualization.get('n_val_samples', 100),
                plot_median_v_true=config.visualization.get('plot_median_v_true', True),
                plot_tarp=config.visualization.get('plot_tarp', True),
                plot_rank=config.visualization.get('plot_rank', True),
                use_default_mplstyle=config.visualization.get('use_default_mplstyle', True),
            )
        )

    return callbacks

def sample_num_stars(config: ml_collections.ConfigDict, num_samples: int):
    """Sample number of stars per galaxy based on configuration. """

    if config.simulation.num_stars_dist == 'poisson':
        return np.random.poisson(
            config.simulation.num_stars_mean,
            size=num_samples
        )
    elif config.simulation.num_stars_dist == 'uniform':
        return np.random.randint(
            config.simulation.num_stars_min,
            config.simulation.num_stars_max,
            size=num_samples
        )
    elif config.simulation.num_stars_dist == 'delta':
        return np.full(num_samples, config.simulation.num_stars_value)
    else:
        raise ValueError(f"Unknown num_stars_dist: {config.simulation.num_stars_dist}")

# Initialize storage for multi-round SNPE
prior = BoxUniform(config.prior)
norm_dict = None  # Will be created in round 0 and reused for all rounds

# %% [markdown]
# ### Multi-Round SNPE Training

# Loop over SNPE rounds
model = None  # Model will be created in round 0
proposal = None  # Proposal for sampling (posterior from previous round)

for current_round in range(config.num_rounds):
    print(f"\n{'='*80}")
    print(f"ROUND {current_round} / {config.num_rounds - 1}")
    print(f"{'='*80}\n")

    # ========== STEP 1: Sample parameters and run simulations ==========
    if current_round == 0:
        print(f"[Round {current_round}] Sampling from prior...")
        samples = prior.sample(config.simulation.num_galaxies, seed=config.get('seed', 42) + current_round)
    else:
        print(f"[Round {current_round}] Sampling from proposal (posterior from round {current_round-1})...")
        # Sample from the proposal (posterior from previous round)
        with torch.no_grad():
            # Read observation to condition the posterior
            obs_batch = read_observation(config, norm_dict=norm_dict)
            pre_transforms_obs = build_transformation(
                apply_projection=False,
                apply_selection=False,
                apply_graph=False,
                apply_uncertainty=False,
                use_log_features=True,
                projection_args={'axis': 2},
                norm_dict=norm_dict
            )
            obs_batch = pre_transforms_obs(obs_batch)

            # Sample from posterior
            samples = proposal.sample(
                config.simulation.num_galaxies,
                x_obs=obs_batch,
                norm_dict=norm_dict,
            ).cpu().numpy()

    # Convert samples to simulation parameters
    params_list = samples_to_simulation_params(
        samples, prior.labels,
        dm_type=config.simulation.dm_type,
        stellar_type=config.simulation.stellar_type,
        df_type=config.simulation.df_type,
        dm_params_default=config.simulation.get('dm_params_default', {}),
        stellar_params_default=config.simulation.get('stellar_params_default', {}),
        df_params_default=config.simulation.get('df_params_default', {})
    )

    # Run simulations with multiprocessing
    print(f"[Round {current_round}] Running {config.simulation.num_galaxies} simulations...")
    num_stars_list = sample_num_stars(config, config.simulation.num_galaxies)
    node_features, graph_features = run_simulation_batch(
        params_list, num_stars_list,
        max_iter=config.simulation.get('max_iter', 1000),
        n_jobs=config.simulation.get('n_jobs', None),
        use_multiprocessing=config.simulation.get('use_multiprocessing', True)
    )
    node_features, graph_features = preprocess_simulations(
        node_features, graph_features, **config.preprocess
    )

    # ========== STEP 2: Create dataloaders ==========
    print(f"[Round {current_round}] Creating dataloaders...")
    train_loader, val_loader, round_norm_dict = datasets.prepare_dataloaders(
        node_features, graph_features, config.labels,
        cond_labels=config.get('cond_labels', None),
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        train_frac=config.train_frac,
        num_workers=config.num_workers,
        seed=config.seed_data,
        norm_dict=norm_dict if current_round > 0 else None  # Only use norm_dict from round 0
    )

    # Store norm_dict from round 0
    if current_round == 0:
        norm_dict = round_norm_dict
        print(f"[Round {current_round}] Created normalization dictionary (will be reused for all rounds)")
    else:
        print(f"[Round {current_round}] Reusing normalization dictionary from round 0")

    print(f"[Round {current_round}] Training samples: {len(train_loader.dataset)}")
    print(f"[Round {current_round}] Validation samples: {len(val_loader.dataset)}")

    # ========== STEP 3: Create or update model ==========
    if current_round == 0:
        # Create model only in round 0
        print(f"[Round {current_round}] Creating model...")
        pre_transforms = build_transformation(norm_dict=norm_dict, **config.pre_transforms)
        model = create_model(
            config, pre_transforms, norm_dict,
            proposal=None,
            prior=prior,
            current_round=0
        )
    else:
        # Update model for subsequent rounds
        print(f"[Round {current_round}] Updating model with new proposal and round...")
        model.set_proposal(proposal)
        model.set_round(current_round)

    # ========== STEP 4: Setup training ==========
    run_dir = Path(config.workdir) / f"{config.name}_round{current_round}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if current_round > 0:
        wandb.finish()  # This is the key!

    wandb_mode = 'disabled' if config.get('debug', False) else 'online'
    wandb_logger = WandbLogger(
        project=config.get("wandb_project", "jgnn-npe"),
        name=f"{config.get('name')}_round{current_round}",
        id=config.get("id", None),
        save_dir=str(run_dir),
        log_model="all",
        config=config.to_dict(),
        mode=wandb_mode,
        resume="allow",
    )
    if current_round == 0:
        wandb_logger.watch(model, log="all", log_freq=500, log_graph=True)

    callbacks = create_callbacks(config, wandb_logger)

    trainer = pl.Trainer(
        default_root_dir=str(run_dir),
        max_epochs=config.num_epochs,
        max_steps=config.num_steps,
        accelerator=config.accelerator,
        callbacks=callbacks,
        logger=wandb_logger,
        enable_progress_bar=config.get("enable_progress_bar", True),
        gradient_clip_val=config.get('gradient_clip_val', None),
    )

    # ========== STEP 5: Train model ==========
    print(f"[Round {current_round}] Starting training...")
    trainer.fit(model, train_loader, val_loader)
    print(f"[Round {current_round}] Training completed!")
    print(f"[Round {current_round}] Best checkpoint: {trainer.checkpoint_callback.best_model_path}")

    # ========== STEP 6: Build posterior for next round ==========
    # The posterior from this round becomes the proposal for the next round
    proposal = model.build_posterior()

    print(f"\n[Round {current_round}] Round complete! Proposal built for round {current_round + 1}.\n")

# Store final model and posterior
final_model = model
final_posterior = proposal
models = [model]  # Keep for compatibility

# ### Final Inference on Observation

# Use the final round's posterior for inference
print(f"[Inference] Using posterior from final round {config.num_rounds - 1}")
print(f"[Inference] Model trained through {config.num_rounds} rounds")

# Read and prepare observation data (single observation: the dwarf galaxy)
print("\n[Observation] Loading observation data...")
observation_batch = read_observation(config, norm_dict=norm_dict)

# Apply transformations
pre_transforms_obs = build_transformation(
    apply_projection=False,
    apply_selection=False,
    apply_graph=False,
    apply_uncertainty=False,
    use_log_features=True,
    projection_args={'axis': 2},
    norm_dict=norm_dict
)
observation_batch = pre_transforms_obs(observation_batch)


# Sample from final posterior
print("\n[Inference] Sampling from final posterior...")
final_model.eval()
with torch.no_grad():
    posterior_samples = final_posterior.sample(
        1000,
        x_obs=observation_batch,
        norm_dict=norm_dict,
    )
posterior_samples_np = posterior_samples.cpu().numpy()

print(f"\n[Inference] Posterior samples shape: {posterior_samples_np.shape}")
print(f"[Inference] Parameter labels: {config.labels}")
print("\n[Inference] Posterior statistics:")
print(f"{'Parameter':<25} {'Median':>10} {'16th %':>10} {'84th %':>10} {'Mean':>10} {'Std':>10}")
print("-" * 80)
for i, label in enumerate(config.labels):
    mean = posterior_samples_np[:, i].mean()
    std = posterior_samples_np[:, i].std()
    q16, q50, q84 = np.percentile(posterior_samples_np[:, i], [16, 50, 84])
    print(f"{label:<25} {q50:10.4f} {q16:10.4f} {q84:10.4f} {mean:10.4f} {std:10.4f}")

try:
    import corner
    import matplotlib.pyplot as plt

    fig = corner.corner(
        posterior_samples_np,
        labels=config.labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt='.4f',
        title_kwargs={"fontsize": 12},
    )
    plt.suptitle(f"Posterior from Round {config.num_rounds - 1}", y=1.02, fontsize=14)
    plt.savefig("posterior_corner_plot.png", bbox_inches='tight', dpi=300)
    plt.show()

except ImportError:
    print("Install 'corner' package to visualize posterior samples: pip install corner")
except Exception as e:
    print(f"Visualization error: {e}")
