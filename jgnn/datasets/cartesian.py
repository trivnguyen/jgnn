"""Cartesian dataset helpers.

Handles HDF5 files produced by sample_galaxies.py whose node features are
3-D Cartesian phase-space coordinates (pos, vel, vel_error).

Graph layout
------------
  pos      : (N, 3)  3-D position, kpc
  vel      : (N, 3)  3-D velocity, km/s
  x        : (N, 2)  [log10(||pos||), ||vel||]  — assembled by the model
  theta    : (1, L)  graph-level labels (parameters)
  cond     : (1, C)  graph-level conditionals (optional)
"""

import numpy as np
import torch
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from . import preprocess


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def create_graph_from_posvel(pos, vel, vel_error=None, label=None, cond=None):
    """Alias kept for backward compatibility; delegates to preprocess."""
    return preprocess.create_graph_from_posvel(
        pos, vel, vel_error=vel_error, label=label, cond=cond)


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------

def prepare_dataloaders(
    node_feats, graph_feats, labels, train_frac=0.8, train_batch_size=32,
    eval_batch_size=32, num_workers=1, norm_dict=None, seed=0, cond_labels=None
):
    """Prepare train/val dataloaders from Cartesian phase-space node features."""
    pl.seed_everything(seed)

    num_graphs = len(graph_feats['num_stars'])
    ptr = np.cumsum(graph_feats['num_stars'])
    ptr = np.insert(ptr, 0, 0)

    graphs = []
    loop = tqdm(range(num_graphs), miniters=num_graphs // 100,
                desc='Creating dataloader')
    for i in loop:
        pos = node_feats['pos'][ptr[i]:ptr[i+1]]
        vel = node_feats['vel'][ptr[i]:ptr[i+1]]
        if node_feats.get('vel_error') is not None:
            vel_error = node_feats['vel_error'][ptr[i]:ptr[i+1]]
        else:
            vel_error = np.zeros_like(vel)

        flow_labels = [graph_feats[k][i] for k in labels]
        cond = [graph_feats[k][i] for k in cond_labels] if cond_labels else None

        graph = preprocess.create_graph_from_posvel(
            pos, vel, vel_error=vel_error, label=flow_labels, cond=cond)
        graphs.append(graph)

    num_train = int(num_graphs * train_frac)
    np.random.shuffle(graphs)
    train_graphs = graphs[:num_train]
    val_graphs = graphs[num_train:]

    device = train_graphs[0].x.device

    if norm_dict is not None:
        x_loc = torch.tensor(norm_dict['x_loc'], dtype=torch.float32, device=device)
        x_scale = torch.tensor(norm_dict['x_scale'], dtype=torch.float32, device=device)
        theta_loc = torch.tensor(norm_dict['theta_loc'], dtype=torch.float32, device=device)
        theta_scale = torch.tensor(norm_dict['theta_scale'], dtype=torch.float32, device=device)
        if cond_labels:
            cond_loc = torch.tensor(norm_dict['cond_loc'], dtype=torch.float32, device=device)
            cond_scale = torch.tensor(norm_dict['cond_scale'], dtype=torch.float32, device=device)
    else:
        pos_train = torch.cat([g.pos for g in train_graphs])
        vel_train = torch.cat([g.vel for g in train_graphs])
        theta_train = torch.cat([g.theta for g in train_graphs], dim=0)

        rad3d = torch.norm(pos_train, dim=1).view(-1, 1)
        vel3d = torch.norm(vel_train, dim=1).view(-1, 1)
        log_rad = torch.log10(rad3d + 1e-6)
        x_loc = torch.cat([log_rad, vel3d], dim=1).mean(dim=0)
        x_scale = torch.cat([log_rad, vel3d], dim=1).std(dim=0)

        theta_min = theta_train.min(dim=0)[0]
        theta_max = theta_train.max(dim=0)[0]
        theta_loc = (theta_max + theta_min) / 2
        theta_scale = (theta_max - theta_min) / 2

        if cond_labels:
            cond_train = torch.cat([g.cond for g in train_graphs], dim=0)
            cond_min = cond_train.min(dim=0)[0]
            cond_max = cond_train.max(dim=0)[0]
            cond_loc = (cond_max + cond_min) / 2
            cond_scale = (cond_max - cond_min) / 2

    norm_dict = {
        'x_loc': list(x_loc.cpu().numpy()),
        'x_scale': list(x_scale.cpu().numpy()),
        'theta_loc': list(theta_loc.cpu().numpy()),
        'theta_scale': list(theta_scale.cpu().numpy()),
    }
    if cond_labels:
        norm_dict['cond_loc'] = list(cond_loc.cpu().numpy())
        norm_dict['cond_scale'] = list(cond_scale.cpu().numpy())

    for g in train_graphs + val_graphs:
        g.theta = (g.theta - theta_loc) / theta_scale
        if cond_labels:
            g.cond = (g.cond - cond_loc) / cond_scale

    train_loader = PyGDataLoader(
        train_graphs, batch_size=train_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False)
    val_loader = PyGDataLoader(
        val_graphs, batch_size=eval_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader, norm_dict


def prepare_test_dataloader(
    node_feats, graph_feats, labels, batch_size=32, num_workers=1,
    norm_dict=None, seed=0, max_graphs=None, cond_labels=None
):
    """Prepare a test dataloader from Cartesian phase-space node features."""
    pl.seed_everything(seed)

    num_graphs = len(graph_feats['num_stars'])
    if max_graphs is not None:
        num_graphs = min(num_graphs, max_graphs)

    ptr = np.cumsum(graph_feats['num_stars'])
    ptr = np.insert(ptr, 0, 0)

    graphs = []
    loop = tqdm(range(num_graphs), miniters=num_graphs // 100,
                desc='Creating dataloader')
    for i in loop:
        pos = node_feats['pos'][ptr[i]:ptr[i+1]]
        vel = node_feats['vel'][ptr[i]:ptr[i+1]]
        vel_error = node_feats['vel_error'][ptr[i]:ptr[i+1]]
        flow_labels = [graph_feats[k][i] for k in labels]
        cond = [graph_feats[k][i] for k in cond_labels] if cond_labels else None

        graph = preprocess.create_graph_from_posvel(
            pos, vel, vel_error=vel_error, label=flow_labels, cond=cond)
        graphs.append(graph)

    device = graphs[0].x.device

    if norm_dict is not None:
        x_loc = torch.tensor(norm_dict['x_loc'], dtype=torch.float32, device=device)
        x_scale = torch.tensor(norm_dict['x_scale'], dtype=torch.float32, device=device)
        theta_loc = torch.tensor(norm_dict['theta_loc'], dtype=torch.float32, device=device)
        theta_scale = torch.tensor(norm_dict['theta_scale'], dtype=torch.float32, device=device)
        if cond_labels:
            cond_loc = torch.tensor(norm_dict['cond_loc'], dtype=torch.float32, device=device)
            cond_scale = torch.tensor(norm_dict['cond_scale'], dtype=torch.float32, device=device)
    else:
        pos_all = torch.cat([g.pos for g in graphs])
        vel_all = torch.cat([g.vel for g in graphs])
        theta_all = torch.cat([g.theta for g in graphs], dim=0)

        rad3d = torch.norm(pos_all, dim=1).view(-1, 1)
        vel3d = torch.norm(vel_all, dim=1).view(-1, 1)
        log_rad = torch.log10(rad3d + 1e-6)
        x_loc = torch.cat([log_rad, vel3d], dim=1).mean(dim=0)
        x_scale = torch.cat([log_rad, vel3d], dim=1).std(dim=0)

        theta_min = theta_all.min(dim=0)[0]
        theta_max = theta_all.max(dim=0)[0]
        theta_loc = (theta_max + theta_min) / 2
        theta_scale = (theta_max - theta_min) / 2

        if cond_labels:
            cond_all = torch.cat([g.cond for g in graphs], dim=0)
            cond_min = cond_all.min(dim=0)[0]
            cond_max = cond_all.max(dim=0)[0]
            cond_loc = (cond_max + cond_min) / 2
            cond_scale = (cond_max - cond_min) / 2

    norm_dict = {
        'x_loc': list(x_loc.cpu().numpy()),
        'x_scale': list(x_scale.cpu().numpy()),
        'theta_loc': list(theta_loc.cpu().numpy()),
        'theta_scale': list(theta_scale.cpu().numpy()),
    }
    if cond_labels:
        norm_dict['cond_loc'] = list(cond_loc.cpu().numpy())
        norm_dict['cond_scale'] = list(cond_scale.cpu().numpy())

    for g in graphs:
        g.theta = (g.theta - theta_loc) / theta_scale
        if cond_labels:
            g.cond = (g.cond - cond_loc) / cond_scale

    loader = PyGDataLoader(
        graphs, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False)

    return loader, norm_dict
