
import os
import h5py
from tqdm import tqdm

import numpy as np
import torch
import pytorch_lightning as pl
from tqdm import tqdm
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
                data_path, concat=concat)

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
            data_path, concat=concat)

    return node_feats, graph_feats


### For inference task ###
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
    train_loader = PyGDataLoader(
        train_graphs, batch_size=train_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False)
    val_loader = PyGDataLoader(
        val_graphs, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader, norm_dict

def prepare_test_dataloader(
    node_feats, graph_feats, labels, batch_size=32, num_workers=1, norm_dict=None,
    seed=0, norm_version='v2', max_graphs=None
):
    """ Prepare the dataloaders for training and validation. """

    if norm_version not in ['v1', 'v2']:
        raise ValueError(
            f"Invalid norm_version {norm_version}. "
            "Supported versions are 'v1' and 'v2'.")

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
        cond = graph_feats['cond'][i]
        flow_labels = [graph_feats[k][i] for k in labels]

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
    else:
        if norm_version == 'v1':
            # old normalization scheme for 2D data
            if norm_dict is None:
                x_train = torch.cat([g.x for g in graphs])
                theta_train = torch.cat([g.theta for g in graphs])
                x_loc = x_train.mean(dim=0)
                x_scale = x_train.std(dim=0)
                theta_loc = theta_train.mean(dim=0)
                theta_scale = theta_train.std(dim=0)
        elif norm_version == 'v2':
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

        norm_dict = {
            'x_loc': list(x_loc.cpu().numpy()),
            'x_scale': list(x_scale.cpu().numpy()),
            'theta_loc': list(theta_loc.cpu().numpy()),
            'theta_scale': list(theta_scale.cpu().numpy()),
        }

    if norm_version == 'v1':
        for g in graphs:
            # g.x = (g.x - x_loc) / x_scale
            g.theta = (g.theta - theta_loc) / theta_scale
    elif norm_version == 'v2':
        for g in graphs:
            g.theta = (g.theta - theta_loc) / theta_scale

    # create data loaders
    loader = PyGDataLoader(
        graphs, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False)

    return loader, norm_dict

### For reconstruction task ###
def prepare_dataloaders_recon(
    node_feats, graph_feats, labels, train_frac=0.8, train_batch_size=32,
    eval_batch_size=32, num_subsampling=None, num_workers=1, norm_dict=None,
    seed=0,
):
    """ Prepare the dataloaders for training and validation. """

    pl.seed_everything(seed)

    num_graphs = len(graph_feats['num_stars'])
    ptr = np.cumsum(graph_feats['num_stars'])
    ptr = np.insert(ptr, 0, 0)

    loop = tqdm(range(num_graphs), miniters=num_graphs // 100, desc='Creating dataloader')

    pos_train, vel_true_train, theta_train = [], [], []
    pos_val, vel_true_val, theta_val = [], [], []
    for i in loop:
        pos = node_feats['pos'][ptr[i]:ptr[i+1]]
        vel_true = node_feats['vel_true'][ptr[i]:ptr[i+1]]
        theta = np.array([graph_feats[k][i] for k in labels])

        # subsample the stars if needed
        if num_subsampling is not None and num_subsampling < len(pos):
            idx = np.random.choice(len(pos), num_subsampling, replace=False)
            pos = pos[idx]
            vel_true = vel_true[idx]

        pos = np.log10(np.linalg.norm(pos, axis=1).reshape(-1, 1))
        vel_true = vel_true.reshape(-1, 1)
        theta = np.repeat(theta.reshape(1, -1), len(pos), axis=0)

        # decide whether to put the graph in train or val set
        if np.random.rand() < train_frac:
            pos_train.append(pos)
            vel_true_train.append(vel_true)
            theta_train.append(theta)
        else:
            pos_val.append(pos)
            vel_true_val.append(vel_true)
            theta_val.append(theta)

    pos_train = np.concatenate(pos_train, axis=0)
    vel_true_train = np.concatenate(vel_true_train, axis=0)
    theta_train = np.concatenate(theta_train, axis=0)
    pos_val = np.concatenate(pos_val, axis=0)
    vel_true_val = np.concatenate(vel_true_val, axis=0)
    theta_val = np.concatenate(theta_val, axis=0)

    # Normalize the data
    if norm_dict is not None:
        pos_loc = np.array(norm_dict['pos_loc'])
        pos_scale = np.array(norm_dict['pos_scale'])
        theta_loc = np.array(norm_dict['theta_loc'])
        theta_scale = np.array(norm_dict['theta_scale'])
        vel_true_loc = np.array(norm_dict['vel_true_loc'])
        vel_true_scale = np.array(norm_dict['vel_true_scale'])
    else:
        pos_loc, pos_scale = pos_train.mean(0), pos_train.std(0)
        theta_loc, theta_scale = theta_train.mean(0), theta_train.std(0)
        vel_true_min, vel_true_max = vel_true_train.min(0), vel_true_train.max(0)
        vel_true_loc = (vel_true_min + vel_true_max) / 2
        vel_true_scale = (vel_true_max - vel_true_min) / 2

        norm_dict = {
            'pos_loc': list(pos_loc),
            'pos_scale': list(pos_scale),
            'theta_loc': list(theta_loc),
            'theta_scale': list(theta_scale),
            'vel_true_loc': list(vel_true_loc),
            'vel_true_scale': list(vel_true_scale),
        }

    pos_train = (pos_train - pos_loc) / pos_scale
    vel_true_train = (vel_true_train - vel_true_loc) / vel_true_scale
    theta_train = (theta_train - theta_loc) / theta_scale
    pos_val = (pos_val - pos_loc) / pos_scale
    vel_true_val = (vel_true_val - vel_true_loc) / vel_true_scale
    theta_val = (theta_val - theta_loc) / theta_scale

    # create data loaders
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(pos_train, dtype=torch.float32),
            torch.tensor(vel_true_train, dtype=torch.float32),
            torch.tensor(theta_train, dtype=torch.float32),
        ),
        batch_size=train_batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=False,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(pos_val, dtype=torch.float32),
            torch.tensor(vel_true_val, dtype=torch.float32),
            torch.tensor(theta_val, dtype=torch.float32),
        ),
        batch_size=eval_batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False
    )

    return train_loader, val_loader, norm_dict
