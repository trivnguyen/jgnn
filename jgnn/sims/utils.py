
import h5py
import numpy as np

def get_graph(node_features, graph_features, idx):
    """ Get a graph from the dataset given the index """
    nodes = {}
    for k, v in node_features.items():
        nodes[k] = v[idx]
    graph = {}
    for k, v in graph_features.items():
        graph[k] = v[idx]
    return nodes, graph

def random_rotation_matrix():
    # Generate a random quaternion
    q = np.random.randn(4)
    q /= np.linalg.norm(q)  # Normalize the quaternion

    # Convert quaternion to rotation matrix
    q0, q1, q2, q3 = q
    R = np.array([
        [1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q3*q0, 2*q1*q3 + 2*q2*q0],
        [2*q1*q2 + 2*q3*q0, 1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q1*q0],
        [2*q1*q3 - 2*q2*q0, 2*q2*q3 + 2*q1*q0, 1 - 2*q1**2 - 2*q2**2]
    ])
    return R

def project2d(pos, vel, axis=0, use_proper_motions=False):
    """ Project the 3D positions and velocities to 2D.
    Return the 2d positions and line-of-sight velocities.

    Parameters
    ----------
    pos : array_like
        3D positions shape (N, 3)
    vel : array_like
        3D velocities shape (N, 3)
    axis : int or str
        The LOS Axis to project to (0, 1, or 2). If None, apply a random projection.
    use_proper_motions : bool
        Whether to include proper motions in the velocities

    Returns
    -------
    pos_proj : array_like
        2D positions
    vel_proj: array_like
        Line-of-sight velocities
    """
    # if axis is 'random', apply a random projection
    # by randomly rotate the 3D positions and velocities
    if axis is None:
        R = random_rotation_matrix()
        pos = np.dot(pos, R)
        vel = np.dot(vel, R)
        axis = np.random.randint(3)
    # project to 2D
    pos_proj = np.delete(pos, axis, axis=1)

    if use_proper_motions:
        return pos_proj, vel
    else:
        return pos_proj, vel[:, axis]

def parse_graph_features(graph_features, norm_rstar=False):
    """ Parse graph features into training target """

    # create a copy of the graph features
    new_graph_features = graph_features.copy()

    # parse DM parameters
    new_graph_features['dm_log_r_dm'] = np.log10(graph_features['dm_r_dm'])
    new_graph_features['dm_log_rho_0'] = np.log10(graph_features['dm_rho_0'])

    # parse stellar parameters
    if graph_features.get('stellar_r_star') is not None:
        new_graph_features['stellar_log_r_star'] = np.log10(
            graph_features['stellar_r_star'])
    elif graph_features.get('stellar_r_star_r_dm') is not None:
        new_graph_features['stellar_log_r_star'] = (
            np.log10(graph_features['stellar_r_star_r_dm'])
            + new_graph_features['dm_log_r_dm'])
    else:
        raise ValueError('Cannot find stellar radius')

    # parse DF parameters
    if graph_features.get('df_r_a') is not None:
        new_graph_features['df_log_r_a'] = np.log10(graph_features['df_r_a'])
    elif graph_features.get('df_r_a_r_dm') is not None:
        new_graph_features['df_log_r_a'] = (
            np.log10(graph_features['df_r_a_r_dm'])
            + new_graph_features['dm_log_r_dm'])
    elif graph_features.get('df_r_a_r_star') is not None:
        new_graph_features['df_log_r_a'] = (
            np.log10(graph_features['df_r_a_r_star'])
            + new_graph_features['stellar_log_r_star'])
    else:
        raise ValueError('Cannot find DF scale radius')

    # normalize by the stellar radius
    if norm_rstar:
        for k in ['dm_log_r_dm', 'df_log_r_a', ]:
            new_graph_features[k] -= new_graph_features['stellar_log_r_star']
        new_graph_features['dm_log_rho_0'] -= (
            new_graph_features['stellar_log_r_star'] * 3)
    return new_graph_features


### IO functions ###
def write_graph_dataset(
        path, node_features, graph_features, lengths, headers=None):
    """ Write a graph dataset with node features and graph features into HDF5
    file. The node features of all graphs are concatenated into a single
    array. The lengths of each graph is required to split the node features
    into individual graphs.

    Parameters
    ----------
    path : str
        Path to the HDF5 file.
    node_features : dict
        Dictionary of node features. The node features of all graphs are
        concatenated into a single array. The key is the name of the feature and
        the value is a list of arrays of shape (M, ) where M is the number
        of nodes in all graphs.
    graph_features : dict
        Dictionary of graph features. The key is the name of the feature and
        the value is a list of arrays of shape (N, ) where N is the number
        of graphs.
    lengths : list
        List of lengths of each graph. The length of the list is the number of
        graphs. The sum of the lengths must be equal to M, or the total number
        of nodes in all graphs.
    headers : dict
        Dictionary of headers. The key is the name of the header and the value
        is the value of the header.
    Returns
    -------
    None
    """
    # check if total length of node features is equal to the sum of lengths
    assert np.sum(lengths) == len(list(node_features.values())[0])

    # construct pointer to each graph
    ptr = np.cumsum(lengths)

    # define headers
    default_headers ={
        "node_features": list(node_features.keys()),
        "graph_features": list(graph_features.keys()),
    }
    default_headers['all_features'] = (
        default_headers['node_features'] + default_headers['graph_features'])
    if headers is None:
        headers = default_headers
    else:
        headers.update(default_headers)

    # write dataset into HDF5 file
    with h5py.File(path, 'w') as f:
        # write pointers
        f.create_dataset('ptr', data=ptr)

        # write node features
        for key in node_features:
            dset = f.create_dataset(key, data=node_features[key])
            dset.attrs.update({'type': 'node_features'})

        # write tree features
        for key in graph_features:
            dset = f.create_dataset(key, data=graph_features[key])
            dset.attrs.update({'type': 'graph_features'})

        # write headers
        f.attrs.update(headers)

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
