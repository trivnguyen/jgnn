
import os
import h5py
from tqdm import tqdm

import numpy as np
import torch
import pytorch_lightning as pl
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from . import preprocess

def read_graph_dataset(path, features_list=None, concat=False, to_array=True):
    """ Read graph dataset from path and return node features, graph
    features, and headers.

    Parameters
    ----------
    path : str
        Path to the HDF5 file.
    features_list : list
        List of features to read. If empty, all features will be read.
    concat : bool
        If True, the node features of all graphs will be concatenated into a
        single array. Otherwise, the node features will be returned as a list
        of arrays.
    to_array : bool
        If True, the node features will be returned as a numpy array of
        dtype='object'. Otherwise, the node features will be returned as a
        list of arrays. This option is only used when concat is False.

    Returns
    -------
    node_features : dict
        Dictionary of node features. The key is the name of the feature and
        the value is a list of arrays of shape (M, ) where M is the number
        of nodes in all graphs.
    graph_features : dict
        Dictionary of graph features. The key is the name of the feature and
        the value is a list of arrays of shape (N, ) where N is the number
        of graphs.
    headers : dict
        Dictionary of headers.
    """
    if features_list is None:
        features_list = []

    # read dataset from HDF5 file
    with h5py.File(path, 'r') as f:
        # read dataset attributes as headers
        headers = dict(f.attrs)

        # if features_list is empty, read all features
        if len(features_list) == 0:
            features_list = headers['all_features']

        # read node features
        node_features = {}
        for key in headers['node_features']:
            if key in features_list:
                if f.get(key) is None:
                    logger.warning(f"Feature {key} not found in {path}")
                    continue
                if concat:
                    node_features[key] = f[key][:]
                else:
                    node_features[key] = np.split(f[key][:], f['ptr'][:-1])

        # read graph features
        graph_features = {}
        for key in headers['graph_features']:
            if key in features_list:
                if f.get(key) is None:
                    logger.warning(f"Feature {key} not found in {path}")
                    continue
                graph_features[key] = f[key][:]

    # convert node features to numpy array of dtype='object'
    if not concat and to_array:
        node_features = {
            p: np.array(v, dtype='object') for p, v in node_features.items()}
    return node_features, graph_features, headers

def read_datasets(
    root, name, num_datasets=100, init=0, is_directory=True, concat=True):

    if is_directory:
        node_feats, graph_feats = {}, {}

        for i in tqdm(range(init, init + num_datasets)):
            data_path = os.path.join(root, name, "data.{}.hdf5".format(i))
            if not os.path.exists(data_path):
                break
            nodes, graphs, _ = read_graph_dataset(
                data_path, concat=True)

            # append to the dataset
            for k in nodes:
                node_feats[k] = [] if i == init else node_feats[k]
                node_feats[k].append(nodes[k])
            for k in graphs:
                graph_feats[k] = [] if i == init else graph_feats[k]
                graph_feats[k].append(graphs[k])
        # concatenate the datasets
        for k in node_feats:
            node_feats[k] = np.concatenate(node_feats[k])
        for k in graph_feats:
            graph_feats[k] = np.concatenate(graph_feats[k])
    else:
        data_path  = os.path.join(root, name + ".hdf5")
        node_feats, graph_feats, _ = read_graph_dataset(
            data_path, concat=True)

    return node_feats, graph_feats

def prepare_dataloaders(
    node_feats, graph_feats, labels, train_frac=0.8, train_batch_size=32,
    eval_batch_size=32, num_workers=1, norm_dict=None, seed=0,
    norm_version='v2',
):
    """ Prepare the dataloaders for training and validation. """

    if norm_version not in ['v1', 'v2']:
        raise ValueError(
            f"Invalid norm_version {norm_version}. "
            "Supported versions are 'v1' and 'v2'.")

    pl.seed_everything(seed)

    num_graphs = len(graph_feats['num_stars'])
    ptr = np.cumsum(graph_feats['num_stars'])
    ptr = np.insert(ptr, 0, 0)

    graphs = []

    loop = tqdm(range(num_graphs), miniters=num_graphs // 100, desc='Creating dataloader')
    for i in loop:
        pos = node_feats['pos'][ptr[i]:ptr[i+1]]
        vel = node_feats['vel'][ptr[i]:ptr[i+1]]
        vel_error = node_feats['vel_error'][ptr[i]:ptr[i+1]]
        cond = graph_feats['cond'][i]
        flow_labels = [graph_feats[k][i] for k in labels]

        graph = preprocess.create_graph_from_posvel(
            pos, vel, vel_error=vel_error, label=flow_labels, cond=cond)
        graphs.append(graph)

    # split the dataset into train and val
    num_train = int(num_graphs * train_frac)
    np.random.shuffle(graphs)
    train_graphs = graphs[:num_train]
    val_graphs = graphs[num_train:]

    device = train_graphs[0].x.device

    # Normalize input data
    if norm_dict is not None:
        x_loc = torch.tensor(norm_dict['x_loc'], dtype=torch.float32, device=device)
        x_scale = torch.tensor(norm_dict['x_scale'], dtype=torch.float32, device=device)
        theta_loc = torch.tensor(norm_dict['theta_loc'], dtype=torch.float32, device=device)
        theta_scale = torch.tensor(norm_dict['theta_scale'], dtype=torch.float32, device=device)
    else:
        if norm_version == 'v1':
            # old normalization scheme for 2D data
            if norm_dict is None:
                x_train = torch.cat([g.x for g in train_graphs])
                theta_train = torch.cat([g.theta for g in train_graphs])
                x_loc = x_train.mean(dim=0)
                x_scale = x_train.std(dim=0)
                theta_loc = theta_train.mean(dim=0)
                theta_scale = theta_train.std(dim=0)
        elif norm_version == 'v2':
            # new normalization scheme for 3D data
            pos = torch.cat([g.pos for g in train_graphs])
            vel = torch.cat([g.vel for g in train_graphs])
            theta_train = torch.cat([g.theta for g in train_graphs], dim=0)

            # normalize the position and velocity
            rad3d = torch.norm(pos, dim=1).view(-1, 1)
            vel3d = torch.norm(vel, dim=1).view(-1, 1)
            log_rad = torch.log10(rad3d + 1e-6)
            x_loc = torch.cat([log_rad, vel3d], dim=1).mean(dim=0)
            x_scale = torch.cat([log_rad, vel3d], dim=1).std(dim=0)

            # min max normalization for Theta [-1, 1] instead
            theta_min = theta_train.min(dim=0)[0]
            theta_max = theta_train.max(dim=0)[0]
            theta_loc = (theta_max + theta_min) / 2
            theta_scale = (theta_max - theta_min) / 2

        norm_dict = {
            'x_loc': list(x_loc.cpu().numpy()),
            'x_scale': list(x_scale.cpu().numpy()),
            'theta_loc': list(theta_loc.cpu().numpy()),
            'theta_scale': list(theta_scale.cpu().numpy()),
        }

    if norm_version == 'v1':
        for g in train_graphs:
            # g.x = (g.x - x_loc) / x_scale
            g.theta = (g.theta - theta_loc) / theta_scale
        for g in val_graphs:
            # g.x = (g.x - x_loc) / x_scale
            g.theta = (g.theta - theta_loc) / theta_scale
    elif norm_version == 'v2':
        for g in train_graphs:
            g.theta = (g.theta - theta_loc) / theta_scale
        for g in val_graphs:
            g.theta = (g.theta - theta_loc) / theta_scale

    # create data loaders
    train_loader = DataLoader(
        train_graphs, batch_size=train_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(
        val_graphs, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader, norm_dict
