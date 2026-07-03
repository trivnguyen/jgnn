"""Training script for Neural Posterior Estimation (NPE)."""

import sys

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import wandb
import ml_collections
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import summarize
import torch
from absl import flags
from ml_collections import config_flags

from jgnn import datasets, training
from jgnn.models import NPE, GNNEmbedding, TransformerEmbedding
from jgnn.transforms import build_transformation
from jgnn.callbacks.visualization import NPEVisualizationCallback, TargetPosteriorCallback


def load_embedding_network(
    config: ml_collections.ConfigDict, checkpoint_path: str, freeze: bool = False):
    """Load a pre-trained embedding network from checkpoint.

    Args:
        config: Configuration dictionary
        checkpoint_path: Path to checkpoint
        freeze: If True, freeze all parameters of the embedding network

    Returns:
        Tuple of (embedding_network, norm_dict)
        - embedding_network: The loaded GNNEmbedding model
        - norm_dict: The normalization dictionary from the checkpoint (if available)
    """
    print(f"[Embedding] Loading pre-trained embedding network from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if config.model.embedding.type == 'transformer':
        print(f"[Embedding] Detected TransformerEmbedding model type")
        embedding_nn = TransformerEmbedding.load_from_checkpoint(checkpoint_path)
    elif config.model.embedding.type == 'gnn':
        print(f"[Embedding] Detected GNNEmbedding model type")
        embedding_nn = GNNEmbedding.load_from_checkpoint(checkpoint_path)
    else:
        raise ValueError(
            f"Unsupported embedding model type: {config.model.embedding.type}")

    norm_dict = None
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        if 'norm_dict' in hparams:
            norm_dict = hparams['norm_dict']
            print(f"[Embedding] Loaded norm_dict from embedding checkpoint")

    if freeze:
        for param in embedding_nn.parameters():
            param.requires_grad = False
        embedding_nn.eval()
        print(f"[Embedding] Froze all parameters in embedding network")

    print(f"[Embedding] Embedding network loaded successfully")
    print(f"[Embedding] Output size: {embedding_nn.output_size}")

    return embedding_nn, norm_dict


def prepare_data(config: ml_collections.ConfigDict, embedding_norm_dict=None):
    """Load and prepare datasets with transformations.

    Args:
        config: Configuration dictionary
        embedding_norm_dict: Optional norm_dict from embedding checkpoint.
                           If provided, this will be used instead of computing from data.

    Returns:
        Tuple of (train_loader, val_loader, norm_dict)

    Config fields
    -------------
    dataset_type : str, default 'cartesian'
        'cartesian' — 3-D Cartesian phase-space (sample_galaxies.py output)
                      node features: pos, vel, vel_error
        'icrs'      — sky-plane ICRS observables (sample_galaxies_target.py output)
                      node features: ra, dec, vlos, R_proj, vlos_err
    """
    dataset_type = config.get('dataset_type', 'cartesian')
    is_directory = config.get('is_directory', True)

    node_feats, graph_feats = datasets.read_datasets(
        config.data_root,
        config.data_name,
        config.num_datasets,
        init=config.get('init', 0),
        is_directory=is_directory,
        concat=True
    )

    if embedding_norm_dict is not None and config.get('reuse_embedding_norm_dict', True):
        print("[Data] Reusing normalization dict from embedding checkpoint")
        norm_dict = embedding_norm_dict
    else:
        norm_dict = None

    if dataset_type == 'icrs':
        print("[Data] Using ICRS dataset stream (ra/dec/vlos/R_proj)")
        stream = datasets.icrs
    else:
        print("[Data] Using Cartesian dataset stream (pos/vel)")
        stream = datasets.cartesian

    train_loader, val_loader, norm_dict = stream.prepare_dataloaders(
        node_feats,
        graph_feats,
        config.labels,
        cond_labels=config.get('cond_labels', None),
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        train_frac=config.train_frac,
        num_workers=config.num_workers,
        seed=config.seed_data,
        norm_dict=norm_dict
    )

    return train_loader, val_loader, norm_dict


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
            # NPE handles optimizer, scheduler, and pre_transforms
            optimizer_args=None,
            scheduler_args=None,
            pre_transforms=None,
        )
    elif model_type == 'transformer':
        print("[Model] Creating Transformer Embedding model...")
        return TransformerEmbedding(
            input_size=config.model.input_size,
            transformer_args=config.model.embedding.transformer,
            loss_type=config.model.embedding.get('loss_type', 'mse'),
            loss_args=config.model.embedding.get('loss_args', None),
            mlp_args=config.model.embedding.get('mlp', None),
            optimizer_args=None,
            scheduler_args=None,
            pre_transforms=None,
        )
    else:
        raise ValueError(f"Unsupported embedding model type: {config.model.type}")


def create_model(
    config: ml_collections.ConfigDict,
    pre_transforms,
    norm_dict
) -> NPE:
    """Create the NPE model with optional pre-trained embedding network.

    Args:
        config: Configuration dictionary
        pre_transforms: Pre-transformation pipeline (passed to NPE)
        norm_dict: Normalization dictionary to pass to NPE

    Returns:
        NPE model instance
    """
    embedding_checkpoint = config.model.embedding.get('checkpoint', None)
    freeze_embedding = config.model.embedding.get('freeze', False)

    if embedding_checkpoint is not None:
        embedding_nn, _ = load_embedding_network(
            config,
            embedding_checkpoint,
            freeze=freeze_embedding
        )
    else:
        print("[Model] Creating new embedding network...")
        embedding_nn = create_embedding_network(config)

    print("[Model] Creating NPE model...")
    init_flows_from_embedding = config.model.get('init_flows_from_embedding', False)

    return NPE(
        input_size=config.model.input_size,
        output_size=config.model.output_size,
        flows_args=config.model.flows,
        embedding_nn=embedding_nn,
        optimizer_args=config.optimizer,
        scheduler_args=config.scheduler,
        norm_dict=norm_dict,
        pre_transforms=pre_transforms,
        init_flows_from_embedding=init_flows_from_embedding,
    )


def create_callbacks(config: ml_collections.ConfigDict) -> list:
    """Create PyTorch Lightning callbacks, including NPE-specific visualization.

    Args:
        config: Configuration dictionary

    Returns:
        List of callback instances
    """
    callbacks = training.create_base_callbacks(config)

    if config.get('enable_visualization_callback', False):
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

        target_vis_cfg = config.visualization.get('target', None)
        if target_vis_cfg is not None:
            print("[Callbacks] Adding Target Posterior Callback")
            callbacks.append(
                TargetPosteriorCallback(
                    catalog_path=target_vis_cfg.catalog_path,
                    meta_key=target_vis_cfg.meta_key,
                    source=target_vis_cfg.source,
                    loader_kwargs=dict(target_vis_cfg.get('loader_kwargs', {})),
                    cond_values=dict(target_vis_cfg.get('cond_values', {})),
                    cond_labels=list(config.get('cond_labels', [])),
                    n_posterior_samples=target_vis_cfg.get('n_posterior_samples', 2000),
                    plot_every_n_epochs=target_vis_cfg.get('plot_every_n_epochs', 1),
                    param_names=list(target_vis_cfg.get('param_names', config.labels)),
                    meta_path=target_vis_cfg.get('meta_path', None),
                )
            )

    return callbacks


def main(config: ml_collections.ConfigDict, workdir: str = "./logging/"):
    """Train the NPE model with wandb logging.

    Args:
        config: Configuration dictionary containing model and training parameters
        workdir: Working directory for logging and checkpoints
    """
    resume_training = config.get('checkpoint') is not None
    print(f"[Setup] Resume training: {resume_training}")
    print(f"[Setup] Working directory: {workdir}")

    run_dir = training.setup_workdir(workdir)
    print(f"[Setup] Run directory: {run_dir}")

    wandb_logger = training.create_wandb_logger(config, run_dir, tag='npe')

    print("[Data] Loading datasets...")
    # norm_dict is None here; if an embedding checkpoint is configured,
    # create_model() below loads it and returns its norm_dict for reuse.
    train_loader, val_loader, norm_dict = prepare_data(config)
    print(f"[Data] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    print("[Transforms] Building pre-transforms...")
    pre_transforms = build_transformation(
        norm_dict=norm_dict, **config.pre_transforms)

    print("[Model] Creating NPE model...")
    model = create_model(config, pre_transforms, norm_dict)
    summarize(model, max_depth=3)
    training.report_param_counts(model)

    # this watches all parameters and gradients
    wandb_logger.watch(model, log="all", log_freq=1000, log_graph=False)

    checkpoint_path = None
    if resume_training:
        checkpoint_path = training.get_checkpoint_path(config, run_dir)
        print(f"[Checkpoint] Resuming from: {checkpoint_path}")
        print(f"[Checkpoint] Reset optimizer: {config.get('reset_optimizer', False)}")

    callbacks = create_callbacks(config)
    print(f"[Callbacks] Created {len(callbacks)} callbacks")

    trainer = training.build_trainer(
        config, run_dir, callbacks, wandb_logger, num_sanity_val_steps=0)

    pl.seed_everything(config.seed_training, workers=True)
    print(f"[Seed] Training seed set to: {config.seed_training}")

    training.fit(
        trainer, model, train_loader, val_loader,
        checkpoint_path=checkpoint_path,
        reset_optimizer=config.get('reset_optimizer', False),
    )

    wandb.finish()
    print("[WandB] Finished")


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config",
        None,
        "File path to the training hyperparameter configuration.",
        lock_config=True,
    )
    FLAGS(sys.argv)
    main(config=FLAGS.config, workdir=FLAGS.config.workdir)
