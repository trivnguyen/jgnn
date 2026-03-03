import os
import warnings
import h5py
from tqdm import tqdm

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

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
                    warnings.warn(f"Feature {key} not found in {path}")
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
                    warnings.warn(f"Feature {key} not found in {path}")
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
                print(f"Warning: {data_path} does not exist. Skipping...")
                continue
            nodes, graphs, _ = read_graph_dataset(data_path, concat=concat)

            # append to the dataset
            for k in nodes:
                node_feats[k] = [] if i == init else node_feats[k]
                node_feats[k].append(nodes[k])
            for k in graphs:
                graph_feats[k] = [] if i == init else graph_feats[k]
                graph_feats[k].append(graphs[k])

        if len(node_feats) == 0 or len(graph_feats) == 0:
            raise ValueError(f"No valid datasets found in {root}/{name} with init={init} and num_datasets={num_datasets}.")

        # concatenate the datasets
        for k in node_feats:
            node_feats[k] = np.concatenate(node_feats[k])
        for k in graph_feats:
            graph_feats[k] = np.concatenate(graph_feats[k])
    else:
        data_path  = os.path.join(root, name + ".hdf5")
        node_feats, graph_feats, _ = read_graph_dataset(
            data_path, concat=concat)

    return node_feats, graph_feats


### For inference task ###
def prepare_dataloaders(
    node_feats, graph_feats, labels, train_frac=0.8, train_batch_size=32,
    eval_batch_size=32, num_workers=1, norm_dict=None, seed=0, cond_labels=None
):
    """ Prepare the dataloaders for training and validation. """

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
        flow_labels = [graph_feats[k][i] for k in labels]
        cond = [graph_feats[k][i] for k in cond_labels] if cond_labels is not None else None

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
        if cond_labels is not None:
            cond_loc = torch.tensor(norm_dict['cond_loc'], dtype=torch.float32, device=device)
            cond_scale = torch.tensor(norm_dict['cond_scale'], dtype=torch.float32, device=device)
    else:
        # new normalization scheme for 3D data
        pos_train = torch.cat([g.pos for g in train_graphs])
        vel_train = torch.cat([g.vel for g in train_graphs])
        theta_train = torch.cat([g.theta for g in train_graphs], dim=0)

        # normalize the position and velocity
        rad3d = torch.norm(pos_train, dim=1).view(-1, 1)
        vel3d = torch.norm(vel_train, dim=1).view(-1, 1)
        log_rad = torch.log10(rad3d + 1e-6)
        x_loc = torch.cat([log_rad, vel3d], dim=1).mean(dim=0)
        x_scale = torch.cat([log_rad, vel3d], dim=1).std(dim=0)

        # min max normalization for Theta [-1, 1] instead
        theta_min = theta_train.min(dim=0)[0]
        theta_max = theta_train.max(dim=0)[0]
        theta_loc = (theta_max + theta_min) / 2
        theta_scale = (theta_max - theta_min) / 2

        # min max normalization for cond if needed
        if cond_labels is not None:
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
    if cond_labels is not None:
        norm_dict['cond_loc'] = list(cond_loc.cpu().numpy())
        norm_dict['cond_scale'] = list(cond_scale.cpu().numpy())

    # only normalize theta and cond since x is computed on the fly in the model
    for g in train_graphs:
        g.theta = (g.theta - theta_loc) / theta_scale
        if cond_labels is not None:
            g.cond = (g.cond - cond_loc) / cond_scale
    for g in val_graphs:
        g.theta = (g.theta - theta_loc) / theta_scale
        if cond_labels is not None:
            g.cond = (g.cond - cond_loc) / cond_scale

    # create data loaders
    train_loader = PyGDataLoader(
        train_graphs,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    val_loader = PyGDataLoader(
        val_graphs,
        batch_size=eval_batch_size,
        shuffle=True,  # for the visualization callback, which assumes shuffling
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader, norm_dict

def prepare_test_dataloader(
    node_feats, graph_feats, labels, batch_size=32, num_workers=1, norm_dict=None,
    seed=0, max_graphs=None, cond_labels=None
):
    """ Prepare the dataloaders for testing. """

    pl.seed_everything(seed)

    num_graphs = len(graph_feats['num_stars'])
    if max_graphs is not None:
        num_graphs = min(num_graphs, max_graphs)

    ptr = np.cumsum(graph_feats['num_stars'])
    ptr = np.insert(ptr, 0, 0)

    graphs = []

    loop = tqdm(range(num_graphs), miniters=num_graphs // 100, desc='Creating dataloader')
    for i in loop:
        pos = node_feats['pos'][ptr[i]:ptr[i+1]]
        vel = node_feats['vel'][ptr[i]:ptr[i+1]]
        vel_error = node_feats['vel_error'][ptr[i]:ptr[i+1]]
        flow_labels = [graph_feats[k][i] for k in labels]
        cond = [graph_feats[k][i] for k in cond_labels] if cond_labels is not None else None

        graph = preprocess.create_graph_from_posvel(
            pos, vel, vel_error=vel_error, label=flow_labels, cond=cond)
        graphs.append(graph)

    device = graphs[0].x.device

    # Normalize input data
    if norm_dict is not None:
        x_loc = torch.tensor(norm_dict['x_loc'], dtype=torch.float32, device=device)
        x_scale = torch.tensor(norm_dict['x_scale'], dtype=torch.float32, device=device)
        theta_loc = torch.tensor(norm_dict['theta_loc'], dtype=torch.float32, device=device)
        theta_scale = torch.tensor(norm_dict['theta_scale'], dtype=torch.float32, device=device)
        if cond_labels is not None:
            cond_loc = torch.tensor(norm_dict['cond_loc'], dtype=torch.float32, device=device)
            cond_scale = torch.tensor(norm_dict['cond_scale'], dtype=torch.float32, device=device)
    else:
        # new normalization scheme for 3D data
        pos = torch.cat([g.pos for g in graphs])
        vel = torch.cat([g.vel for g in graphs])
        theta_train = torch.cat([g.theta for g in graphs], dim=0)

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

        # min max normalization for cond if needed
        if cond_labels is not None:
            cond_train = torch.cat([g.cond for g in graphs], dim=0)
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
    if cond_labels is not None:
        norm_dict['cond_loc'] = list(cond_loc.cpu().numpy())
        norm_dict['cond_scale'] = list(cond_scale.cpu().numpy())

    # only normalize theta and cond since x is computed on the fly in the model
    for g in graphs:
        g.theta = (g.theta - theta_loc) / theta_scale
        if cond_labels is not None:
            g.cond = (g.cond - cond_loc) / cond_scale

    # create data loaders
    loader = PyGDataLoader(
        graphs, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False)

    return loader, norm_dict
