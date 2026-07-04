"""Shared HDF5 I/O utilities for the dataset streams."""

import os
import warnings

import h5py
import numpy as np
from tqdm import tqdm


def read_graph_dataset(path, features_list=None, concat=False, to_array=True):
    """Read a graph dataset from an HDF5 file.

    Parameters
    ----------
    path          : str   path to the HDF5 file
    features_list : list  features to read; reads all if empty / None
    concat        : bool  if True, concatenate all node features into one array
    to_array      : bool  if True (and not concat), wrap node-feature lists in
                          a numpy object array

    Returns
    -------
    node_features  : dict
    graph_features : dict
    headers        : dict
    """
    if features_list is None:
        features_list = []

    with h5py.File(path, 'r') as f:
        headers = dict(f.attrs)

        if len(features_list) == 0:
            features_list = headers['all_features']

        node_features = {}
        for key in headers['node_features']:
            if key in features_list:
                if f.get(key) is None:
                    warnings.warn(f'Feature {key} not found in {path}')
                    continue
                if concat:
                    node_features[key] = f[key][:]
                else:
                    node_features[key] = np.split(f[key][:], f['ptr'][:-1])

        graph_features = {}
        for key in headers['graph_features']:
            if key in features_list:
                if f.get(key) is None:
                    warnings.warn(f'Feature {key} not found in {path}')
                    continue
                graph_features[key] = f[key][:]

    if not concat and to_array:
        node_features = {
            p: np.array(v, dtype='object') for p, v in node_features.items()}

    return node_features, graph_features, headers


def read_datasets(
    root, name, num_datasets=100, init=0, is_directory=True, concat=True, ext='.h5'
):
    """Read and concatenate multiple HDF5 dataset files."""
    if ext[0] != '.':
        ext = '.' + ext

    if is_directory:
        node_feats, graph_feats = {}, {}

        for i in tqdm(range(init, init + num_datasets)):
            data_path = os.path.join(root, name, f'data.{i}{ext}')
            if not os.path.exists(data_path):
                print(f'Warning: {data_path} does not exist. Skipping...')
                continue
            nodes, graphs, _ = read_graph_dataset(data_path, concat=concat)

            for k in nodes:
                node_feats.setdefault(k, []).append(nodes[k])
            for k in graphs:
                graph_feats.setdefault(k, []).append(graphs[k])

        if not node_feats or not graph_feats:
            raise ValueError(
                f'No valid datasets found in {root}/{name} with '
                f'init={init} and num_datasets={num_datasets}.')

        node_feats = {k: np.concatenate(v) for k, v in node_feats.items()}
        graph_feats = {k: np.concatenate(v) for k, v in graph_feats.items()}
    else:
        data_path = os.path.join(root, name + ext)
        node_feats, graph_feats, _ = read_graph_dataset(
            data_path, concat=concat)

    return node_feats, graph_feats
