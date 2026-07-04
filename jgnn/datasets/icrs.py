"""ICRS dataset helpers.

Handles HDF5 files produced by sample_galaxies_target.py whose node features
are sky-plane observables (ra, dec, vlos, vlos_err, R_proj).

Graph layout
------------
  pos       : (N, 2)  [ra, dec], deg
  vel       : (N, 1)  [vlos], km/s
  vel_error : (N, 1)  [vlos_err], km/s  (optional)
  x         : (N, 2)  [log10(R_proj + eps), vlos]  — model input
  theta     : (1, L)  graph-level labels (parameters)
  cond      : (1, C)  graph-level conditionals (optional)
"""

import numpy as np
import torch
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

_LOG_EPS = 1e-6


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def create_graph_from_icrs(
    ra, dec, vlos, R_proj, vlos_err=None, label=None, cond=None
):
    """Create a PyG graph from ICRS sky-plane observables.

    Parameters
    ----------
    ra, dec  : array (N,)   sky position in degrees
    vlos     : array (N,)   line-of-sight velocity, km/s
    R_proj   : array (N,)   projected radius, kpc
    vlos_err : array (N,)   LOS velocity uncertainty, km/s (optional)
    label    : array-like   graph-level parameter labels (theta)
    cond     : array-like   graph-level conditionals (optional)
    """
    pos = torch.tensor(np.column_stack([ra, dec]), dtype=torch.float32)
    vel = torch.tensor(vlos, dtype=torch.float32).view(-1, 1)
    R = torch.tensor(R_proj, dtype=torch.float32).view(-1, 1)
    log_R = torch.log10(R + _LOG_EPS)
    vel_error = None
    if vlos_err is not None:
        vel_error = torch.tensor(vlos_err, dtype=torch.float32).view(-1, 1)
        x = torch.cat([log_R, vel, vel_error], dim=1)
    else:
        x = torch.cat([log_R, vel], dim=1)

    if label is not None:
        label = torch.tensor(label, dtype=torch.float32).view(1, -1)
    if cond is not None:
        cond = torch.tensor(cond, dtype=torch.float32).view(1, -1)

    return Data(x=x, pos=pos, vel=vel, vel_error=vel_error, theta=label, cond=cond)


# ---------------------------------------------------------------------------
# Internal helper: build graph list from node/graph feature dicts
# ---------------------------------------------------------------------------

def _build_graphs(node_feats, graph_feats, labels, cond_labels=None,
                  max_graphs=None):
    num_graphs = len(graph_feats['num_stars'])
    if max_graphs is not None:
        num_graphs = min(num_graphs, max_graphs)

    ptr = np.cumsum(graph_feats['num_stars'])
    ptr = np.insert(ptr, 0, 0)

    graphs = []
    loop = tqdm(range(num_graphs), miniters=max(1, num_graphs // 100),
                desc='Creating dataloader')
    for i in loop:
        sl = slice(ptr[i], ptr[i + 1])
        ra = node_feats['ra'][sl]
        dec = node_feats['dec'][sl]
        vlos = node_feats['vlos'][sl]
        R_proj = node_feats['R_proj'][sl]
        vlos_err = node_feats.get('vlos_err')
        if vlos_err is not None:
            vlos_err = vlos_err[sl]

        flow_labels = [graph_feats[k][i] for k in labels]
        cond = ([graph_feats[k][i] for k in cond_labels]
                if cond_labels else None)

        graph = create_graph_from_icrs(
            ra, dec, vlos, R_proj,
            vlos_err=vlos_err, label=flow_labels, cond=cond)
        graphs.append(graph)

    return graphs


def _compute_norm(graphs, cond_labels=None):
    """Compute normalization stats from a list of graphs."""
    x_all = torch.cat([g.x for g in graphs])
    x_loc = x_all.mean(dim=0)
    x_scale = x_all.std(dim=0)

    theta_all = torch.cat([g.theta for g in graphs], dim=0)
    theta_min = theta_all.min(dim=0)[0]
    theta_max = theta_all.max(dim=0)[0]
    theta_loc = (theta_max + theta_min) / 2
    theta_scale = (theta_max - theta_min) / 2

    norm = {
        'x_loc': x_loc.tolist(),
        'x_scale': x_scale.tolist(),
        'theta_loc': theta_loc.tolist(),
        'theta_scale': theta_scale.tolist(),
    }

    if cond_labels:
        cond_all = torch.cat([g.cond for g in graphs], dim=0)
        cond_min = cond_all.min(dim=0)[0]
        cond_max = cond_all.max(dim=0)[0]
        norm['cond_loc'] = ((cond_max + cond_min) / 2).tolist()
        norm['cond_scale'] = ((cond_max - cond_min) / 2).tolist()

    return norm


def _apply_norm(graphs, norm_dict, device, cond_labels=None):
    """Normalise theta (and cond) in-place from a norm_dict."""
    # x_loc = torch.tensor(norm_dict['x_loc'], dtype=torch.float32, device=device)
    # x_scale = torch.tensor(norm_dict['x_scale'], dtype=torch.float32, device=device)
    theta_loc = torch.tensor(norm_dict['theta_loc'], dtype=torch.float32, device=device)
    theta_scale = torch.tensor(norm_dict['theta_scale'], dtype=torch.float32, device=device)
    if cond_labels:
        cond_loc = torch.tensor(norm_dict['cond_loc'], dtype=torch.float32, device=device)
        cond_scale = torch.tensor(norm_dict['cond_scale'], dtype=torch.float32, device=device)

    for g in graphs:
        # g.x = (g.x - x_loc) / x_scale
        g.theta = (g.theta - theta_loc) / theta_scale
        if cond_labels:
            g.cond = (g.cond - cond_loc) / cond_scale


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------

def prepare_dataloaders(
    node_feats, graph_feats, labels, train_frac=0.8, train_batch_size=32,
    eval_batch_size=32, num_workers=1, norm_dict=None, seed=0, cond_labels=None
):
    """Prepare train/val dataloaders from ICRS sky-plane node features.

    node_feats must contain: 'ra', 'dec', 'vlos', 'R_proj'
    optionally: 'vlos_err'
    """
    pl.seed_everything(seed)

    graphs = _build_graphs(node_feats, graph_feats, labels, cond_labels)

    num_train = int(len(graphs) * train_frac)
    np.random.shuffle(graphs)
    train_graphs = graphs[:num_train]
    val_graphs = graphs[num_train:]

    device = train_graphs[0].x.device

    if norm_dict is None:
        norm_dict = _compute_norm(train_graphs, cond_labels)

    _apply_norm(train_graphs, norm_dict, device, cond_labels)
    _apply_norm(val_graphs, norm_dict, device, cond_labels)

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
    """Prepare a test dataloader from ICRS sky-plane node features.

    node_feats must contain: 'ra', 'dec', 'vlos', 'R_proj'
    optionally: 'vlos_err'
    """
    pl.seed_everything(seed)

    graphs = _build_graphs(node_feats, graph_feats, labels, cond_labels,
                           max_graphs=max_graphs)

    device = graphs[0].x.device

    if norm_dict is None:
        norm_dict = _compute_norm(graphs, cond_labels)

    _apply_norm(graphs, norm_dict, device, cond_labels)

    loader = PyGDataLoader(
        graphs, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False)

    return loader, norm_dict
