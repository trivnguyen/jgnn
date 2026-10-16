"""Shared HDF5 I/O utilities for the dataset streams."""

import os
import warnings

import h5py
import numpy as np
from tqdm import tqdm


def _read_key(dset, dtype=None):
    """Read one HDF5 dataset, casting floats to `dtype` on the way in.

    The cast happens inside HDF5 (``dset.astype``), so a float32 read never
    materialises the float64 shard in memory (the files are float64).
    Integer datasets (num_stars, ptr) are left as stored.
    """
    if dtype is not None and np.issubdtype(dset.dtype, np.floating):
        return dset.astype(dtype)[...]
    return dset[...]


def read_graph_dataset(path, features_list=None, concat=False, to_array=True,
                       dtype=None):
    """Read a graph dataset from an HDF5 file.

    Parameters
    ----------
    path          : str   path to the HDF5 file
    features_list : list  features to read; reads all if empty / None
    concat        : bool  if True, concatenate all node features into one array
    to_array      : bool  if True (and not concat), wrap node-feature lists in
                          a numpy object array
    dtype         : str or None  float dtype to cast floating datasets to on
                          read (e.g. 'float32'); None keeps the stored float64

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
                    node_features[key] = _read_key(f[key], dtype)
                else:
                    node_features[key] = np.split(
                        _read_key(f[key], dtype), f['ptr'][:-1])

        graph_features = {}
        for key in headers['graph_features']:
            if key in features_list:
                if f.get(key) is None:
                    warnings.warn(f'Feature {key} not found in {path}')
                    continue
                graph_features[key] = _read_key(f[key], dtype)

    if not concat and to_array:
        node_features = {
            p: np.array(v, dtype='object') for p, v in node_features.items()}

    return node_features, graph_features, headers


def _read_shards_preallocated(paths, dtype):
    """Read concatenated shards straight into preallocated arrays.

    Two passes: sizes first, then each shard is read into its slice of the
    final array (floats cast to `dtype` inside HDF5), so the peak memory is
    the final footprint instead of a per-shard list plus its concatenation.
    Same layout as `read_graph_dataset(concat=True)` followed by
    `np.concatenate` over shards.
    """
    with h5py.File(paths[0], 'r') as f:
        headers = dict(f.attrs)
        node_keys = [k for k in headers['node_features'] if k in f]
        graph_keys = [k for k in headers['graph_features'] if k in f]
        shapes = {k: f[k].shape[1:] for k in node_keys + graph_keys}
        dtypes = {k: (np.dtype(dtype) if dtype is not None and np.issubdtype(
            f[k].dtype, np.floating) else f[k].dtype)
            for k in node_keys + graph_keys}
    sizes = []
    for path in paths:
        with h5py.File(path, 'r') as f:
            sizes.append({k: f[k].shape[0] for k in node_keys + graph_keys})
    totals = {k: sum(sz[k] for sz in sizes) for k in node_keys + graph_keys}
    out = {k: np.empty((totals[k],) + shapes[k], dtype=dtypes[k])
           for k in node_keys + graph_keys}
    offsets = {k: 0 for k in out}
    for path, sz in zip(tqdm(paths), sizes):
        with h5py.File(path, 'r') as f:
            for k in out:
                a, b = offsets[k], offsets[k] + sz[k]
                f[k].read_direct(out[k], dest_sel=np.s_[a:b])
                offsets[k] = b
    node_feats = {k: out[k] for k in node_keys}
    graph_feats = {k: out[k] for k in graph_keys}
    return node_feats, graph_feats


def read_datasets(
    root, name, num_datasets=100, init=0, is_directory=True, concat=True,
    ext='.h5', dtype='float32'
):
    """Read and concatenate multiple HDF5 dataset files.

    `dtype` is the float dtype the floating datasets are cast to on read
    ('float32' by default, halving the resident size; None keeps float64).
    With `is_directory` and `concat` the shards are read into preallocated
    arrays so the peak memory is the final footprint.
    """
    if ext[0] != '.':
        ext = '.' + ext

    if is_directory:
        node_feats, graph_feats = {}, {}
        paths = [os.path.join(root, name, f'data.{i}{ext}')
                 for i in range(init, init + num_datasets)]
        missing = [p for p in paths if not os.path.exists(p)]
        for p in missing:
            print(f'Warning: {p} does not exist. Skipping...')
        paths = [p for p in paths if p not in missing]
        if concat and paths:
            return _read_shards_preallocated(paths, dtype)

        for data_path in tqdm(paths):
            nodes, graphs, _ = read_graph_dataset(data_path, concat=concat,
                                                  dtype=dtype)

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
            data_path, concat=concat, dtype=dtype)

    return node_feats, graph_feats
