
from typing import Optional, Tuple

import os
import wandb
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from pytorch_lightning.utilities.model_summary import summarize

from ml_collections import ConfigDict

def fetch_wandb_checkpoint(
    run_path: Optional[str]=None, entity: Optional[str]=None, project: Optional[str]=None,
    run_id: Optional[str]=None, run_name: Optional[str]=None, run_index: Optional[int]=None,
    version: str="best"
) -> Tuple[str, ConfigDict]:
    """ Fetch checkpoint from wandb

    Parameters
    ----------
    run_path : str, optional
        Full wandb run path in the format "entity/project/run_id" or "entity/project/run_name".
        If provided, `entity`, `project`, `run_id`, and `run_name` are ignored.
    entity : str, optional
        Wandb entity name. Must be given if `run_path` is not provided.
    project : str, optional
        Wandb project name. Must be given if `run_path` is not provided.
    run_id : str, optional
        Wandb run id, by default None. Either run_name or run_id must be provided if `run_path` is not given.
    run_name : str, optional
        Wandb run name, by default None. Either run_name or run_id must be provided if `run_path` is not given.
    run_index: int, optional
        If multiple runs have the same name, specify which one to use (0-indexed).
        Raise error if not specified and multiple runs found.
    version : str, optional
        Version of the artifact to fetch, by default "best"

    Returns:
        checkpoint_path : str
            Path to the downloaded checkpoint file
        config : ConfigDict
            Configuration dictionary from the wandb run
    """
    if run_path is None:
        if entity is None or project is None:
            raise ValueError("Either `run_path` or both `entity` and `project` must be provided.")
        if run_name is None and run_id is None:
            raise ValueError("Either `run_name` or `run_id` must be provided.")

    api = wandb.Api()
    if run_path is not None:
        run = api.run(run_path)
        config = ConfigDict(run.config)
        entity = run.entity
        project = run.project
        run_id = run.id
    else:
        if run_id is not None:
            run = api.run(f"{entity}/{project}/{run_id}")
            config = ConfigDict(run.config)
        else:
            runs = api.runs(f"{entity}/{project}", filters={"display_name": run_name})
            if len(runs) == 0:
                raise ValueError(f"No run found with name: {run_name}")
            if len(runs) > 1 and run_index is None:
                raise ValueError(
                    f"Multiple runs found with name: {run_name}. Please specify `run_index`.")
            run = runs[run_index or 0]
            run_id = run.id
            config = ConfigDict(run.config)

    artifact = api.artifact(f'{entity}/{project}/model-{run_id}:{version}')
    artifact_dir = artifact.download()
    checkpoint_path = os.path.join(artifact_dir, 'model.ckpt')
    print(f"Downloaded checkpoint to: {checkpoint_path}")

    return checkpoint_path, config


def fetch_local_checkpoint(
    checkpoint_dir: str, filename: str='model.ckpt'
) -> str:
    """ Load checkpoint from local directory

    Parameters
    ----------
    checkpoint_dir : str
        Directory containing the checkpoint file
    filename : str, optional
        Checkpoint filename, by default 'model.ckpt'

    Returns:
        checkpoint_path : str
            Path to the checkpoint file
    """
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")
    print(f"Loaded checkpoint from: {checkpoint_path}")
    return checkpoint_path


def fetch_config_from_wandb(
    entity: str, project: str, run_name: Optional[str]=None, run_id: Optional[str]=None,
    run_index: Optional[int]=None
) -> ConfigDict:
    """ Load configuration from wandb run

    Parameters
    ----------
    entity : str
        Wandb entity name
    project : str
        Wandb project name
    run_name : str, optional
        Wandb run name, by default None. Either run_name or run_id must be provided.
    run_id : str, optional
        Wandb run id, by default None. Either run_name or run_id must be provided.
    run_index: int, optional
        If multiple runs have the same name, specify which one to use (0-indexed).
        Raise error if not specified and multiple runs found.

    Returns:
        config : ConfigDict
            Configuration dictionary from the wandb run
    """
    if run_name is None and run_id is None:
        raise ValueError("Either `run_name` or `run_id` must be provided.")

    api = wandb.Api()
    if run_name is not None:
        runs = api.runs(f"{entity}/{project}", filters={"display_name": run_name})
        if len(runs) == 0:
            raise ValueError(f"No run found with name: {run_name}")
        if len(runs) > 1 and run_index is None:
            raise ValueError(
                f"Multiple runs found with name: {run_name}. Please specify `run_index`.")
        run = runs[run_index or 0]
        run_id = run.id
        config = ConfigDict(run.config)
    else:
        run = api.run(f"{entity}/{project}/{run_id}")
        config = ConfigDict(run.config)

    return config, run_id



def load_npe_from_checkpoint(
    config, checkpoint_path: str, return_norm_dict: bool = True,
    map_location: str = 'cpu', verbose: bool = True
):
    """Load Neural Probability Estimator (NPE) model from checkpoint.

    Parameters
    ----------
    config : ConfigDict
        Model configuration
    checkpoint_path : str
        Path to the checkpoint file
    return_norm_dict : bool, optional
        Whether to return normalization dictionary, by default True
    map_location : str, optional
        Device to load the model on, by default 'cpu'
    verbose : bool, optional
        Whether to print model summary, by default True

    Returns
    -------
    npe : NPE
        Loaded NPE model
    norm_dict : dict, optional
        Normalization dictionary (if return_norm_dict=True)
    """
    # need to import here to avoid circular dependencies
    from jgnn.models import NPE, GNNEmbedding, TransformerEmbedding

    embedding_type = config.model.embedding.get('type', 'gnn')
    if embedding_type == 'transformer':
        embedding_nn = TransformerEmbedding(
            input_size=config.model.input_size,
            transformer_args=config.model.embedding.transformer,
            mlp_args=config.model.embedding.mlp,
        )
    elif embedding_type == 'gnn':
        embedding_nn = GNNEmbedding(
            input_size=config.model.input_size,
            gnn_args=config.model.embedding.gnn,
            mlp_args=config.model.embedding.mlp,
            conditional_mlp_args=config.model.embedding.get('conditional_mlp', None),
        )
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")

    npe = NPE(
        input_size=config.model.input_size,
        output_size=config.model.output_size,
        flows_args=config.model.flows,
        embedding_nn=embedding_nn,
    )
    npe.eval()

    if verbose:
        print(summarize(npe, max_depth=3))

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    npe.load_state_dict(checkpoint['state_dict'])

    if return_norm_dict:
        norm_dict = checkpoint['hyper_parameters']['norm_dict']
        return npe, norm_dict

    return npe
