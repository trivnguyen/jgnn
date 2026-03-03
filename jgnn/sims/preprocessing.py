"""Preprocessing, transformations, and geometry operations for simulation data."""

from typing import Dict, Tuple, Optional, List
import numpy as np


def random_rotation_matrix():
    """Generate a random 3D rotation matrix using quaternions."""
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
    """Project the 3D positions and velocities to 2D.
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


### Feature Engineering ###

def parse_graph_features(graph_features, norm_rstar=False):
    """Parse graph features into training target.

    Converts linear parameters to log-space and vice versa, and creates all
    ratio variants (r_star/r_dm, r_a/r_dm, r_a/r_star). Ensures comprehensive
    representation where every quantity has both linear and log versions.

    Parameters
    ----------
    graph_features : dict
        Raw graph features as given by simulator
    norm_rstar : bool
        Whether to normalize by stellar rstar

    Returns
    -------
    new_graph_features : dict
        Parsed graph features with comprehensive parameter representations:
        - DM: r_dm, log_r_dm, rho_0, log_rho_0
        - Stellar: r_star, log_r_star, r_star_r_dm, log_r_star_r_dm
        - DF: r_a, log_r_a, r_a_r_dm, log_r_a_r_dm, r_a_r_star, log_r_a_r_star
    """
    # create a copy of the graph features
    new_graph_features = graph_features.copy()

    # ===== DM parameters =====
    # Ensure both linear and log versions exist
    if graph_features.get('dm_r_dm') is not None:
        new_graph_features['dm_log_r_dm'] = np.log10(graph_features['dm_r_dm'])
    elif graph_features.get('dm_log_r_dm') is not None:
        new_graph_features['dm_r_dm'] = 10 ** graph_features['dm_log_r_dm']
        new_graph_features['dm_log_r_dm'] = graph_features['dm_log_r_dm']
    else:
        raise ValueError('Cannot find dm_r_dm or dm_log_r_dm')

    if graph_features.get('dm_rho_0') is not None:
        new_graph_features['dm_log_rho_0'] = np.log10(graph_features['dm_rho_0'])
    elif graph_features.get('dm_log_rho_0') is not None:
        new_graph_features['dm_rho_0'] = 10 ** graph_features['dm_log_rho_0']
        new_graph_features['dm_log_rho_0'] = graph_features['dm_log_rho_0']
    else:
        raise ValueError('Cannot find dm_rho_0 or dm_log_rho_0')

    # ===== Stellar parameters =====
    # First, ensure we have stellar_r_star (absolute value)
    if graph_features.get('stellar_r_star') is not None:
        stellar_r_star = graph_features['stellar_r_star']
    elif graph_features.get('stellar_r_star_r_dm') is not None:
        stellar_r_star = graph_features['stellar_r_star_r_dm'] * new_graph_features['dm_r_dm']
        new_graph_features['stellar_r_star'] = stellar_r_star
    elif graph_features.get('stellar_log_r_star') is not None:
        stellar_r_star = 10 ** graph_features['stellar_log_r_star']
        new_graph_features['stellar_r_star'] = stellar_r_star
    else:
        raise ValueError('Cannot find stellar radius')

    # Create all stellar variants
    new_graph_features['stellar_log_r_star'] = np.log10(stellar_r_star)
    new_graph_features['stellar_r_star_r_dm'] = stellar_r_star / new_graph_features['dm_r_dm']
    new_graph_features['stellar_log_r_star_r_dm'] = np.log10(
        new_graph_features['stellar_r_star_r_dm'])

    # ===== DF parameters =====
    # First, ensure we have df_r_a (absolute value)
    if graph_features.get('df_r_a') is not None:
        df_r_a = graph_features['df_r_a']
    elif graph_features.get('df_r_a_r_dm') is not None:
        df_r_a = graph_features['df_r_a_r_dm'] * new_graph_features['dm_r_dm']
        new_graph_features['df_r_a'] = df_r_a
    elif graph_features.get('df_r_a_r_star') is not None:
        df_r_a = graph_features['df_r_a_r_star'] * stellar_r_star
        new_graph_features['df_r_a'] = df_r_a
    elif graph_features.get('df_log_r_a') is not None:
        df_r_a = 10 ** graph_features['df_log_r_a']
        new_graph_features['df_r_a'] = df_r_a
    else:
        raise ValueError('Cannot find DF scale radius')

    # Create all DF variants
    new_graph_features['df_log_r_a'] = np.log10(df_r_a)
    new_graph_features['df_r_a_r_dm'] = df_r_a / new_graph_features['dm_r_dm']
    new_graph_features['df_log_r_a_r_dm'] = np.log10(
        new_graph_features['df_r_a_r_dm'])
    new_graph_features['df_r_a_r_star'] = df_r_a / stellar_r_star
    new_graph_features['df_log_r_a_r_star'] = np.log10(
        new_graph_features['df_r_a_r_star'])

    return new_graph_features


def preprocess_simulations(
    node_features: Dict,
    graph_features: Dict,
    vrange: Tuple[float, float] = (0, np.inf),
    vdisp_range: Tuple[float, float] = (0, np.inf),
    r_range: Tuple[float, float] = (0, np.inf),
    r_rstar_range: Tuple[float, float] = (0, np.inf),
    apply_projection: bool = True,
    projection_axis: Optional[int] = None,
    use_proper_motions: bool = False,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Dict, Dict]:
    """Preprocess the raw simulation data into training data. Applies
    velocity cuts, projects to 2D, and selects stars within radius range.

    Parameters
    ----------
    node_features : dict
        Raw node features with keys 'pos' and 'vel', as given by simulator().
    graph_features : dict
        Raw graph features as given by simulator().
    vrange : tuple of float
        Velocity range for star selection.
    vdisp_range : tuple of float
        Velocity dispersion range for star selection.
    r_range : tuple of float
        Radius range in kpc for star selection. If `apply_projection` is True,
        applied on projected radius. Applied in addition to r_rstar_range.
    r_rstar_range : tuple of float
        Radius range in units of stellar rstar for star selection. If `apply_projection`
        is True, applied on projected radius. Applied in addition to r_range.
    apply_projection : bool
        Whether to apply 2D projection.
    projection_axis : int, optional
        Axis to project onto (if `apply_projection` is True). If None, random projection is applied.
        Default is None.
    use_proper_motions : bool
        Whether to include proper motions in the velocities.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool
        Whether to print verbose messages.

    Returns
    -------
    new_node_features : dict
        Processed node features with keys 'pos', 'vel', 'vel_true', 'vel_error'.
    new_graph_features : dict
        Processed graph features with added 'num_stars' and parsed labels.
    """

    np.random.seed(seed)

    num_galaxies  = len(node_features['pos'])
    new_node_features = {
        'pos': [],
        'vel': [],
        'vel_true': [],
        'vel_error': [],
    }
    new_graph_features = {k: [] for k in graph_features.keys()}
    new_graph_features['num_stars'] = []

    for i in range(num_galaxies):
        nodes = {k: v[i] for k, v in node_features.items()}
        graph = {k: v[i] for k, v in graph_features.items()}
        pos = nodes['pos'].astype(np.float32)
        vel = nodes['vel'].astype(np.float32)
        if graph.get('stellar_r_star') is not None:
            stellar_rstar = graph['stellar_r_star']
        elif graph.get('stellar_r_star_r_dm') is not None:
            stellar_rstar = graph['stellar_r_star_r_dm'] * graph['dm_r_dm']
        else:
            raise ValueError('Cannot find stellar radius')

        # apply velocity cut on the 3d velocity
        # large velocity can be due to AGAMA sampling issues
        vel3d = np.linalg.norm(vel, axis=1)
        mask = (vel3d > vrange[0]) & (vel3d < vrange[1])
        if np.sum(mask) < int(len(pos) * 0.5):
            # if half of the stars are outside the velocity range, skip this galaxy
            if verbose:
                print(f'Skipping galaxy {i} due to 3D velocity cut')
            continue
        pos = pos[mask]
        vel = vel[mask]

        # apply velocity dispersion cut on the 3d velocity dispersion
        vdisp = np.std(vel, axis=0)
        vdisp = np.linalg.norm(vdisp)
        if vdisp < vdisp_range[0] or vdisp > vdisp_range[1]:
            if verbose:
                print(f'Skipping galaxy {i} due to velocity dispersion cut')
            continue

        # project onto 2D plane
        if apply_projection:
            pos, vel = project2d(
                pos, vel, axis=projection_axis, use_proper_motions=use_proper_motions)

        # apply radius cut
        min_radius = max(r_rstar_range[0] * stellar_rstar, r_range[0])
        max_radius = min(r_rstar_range[1] * stellar_rstar, r_range[1])
        radius = np.linalg.norm(pos, axis=1)
        mask = (radius > min_radius) & (radius < max_radius)

        if np.sum(mask) < int(len(pos) * 0.5):
            # if half of the stars are outside the radius range, skip this galaxy
            print(f'Skipping galaxy {i} due to radius cut')
            continue

        pos = pos[mask]
        vel = vel[mask]
        radius = radius[mask]

        new_node_features['pos'].append(pos)
        new_node_features['vel'].append(vel)
        new_node_features['vel_true'].append(vel)
        new_node_features['vel_error'].append(np.zeros_like(vel))

        for k in graph.keys():
            new_graph_features[k].append(graph[k])
        new_graph_features['num_stars'].append(len(pos))

    # Finalize node and graph features
    if len(new_node_features['pos']) == 0:
        raise ValueError("All galaxies were filtered out during preprocessing")

    for k in new_node_features.keys():
        new_node_features[k] = np.concatenate(new_node_features[k])
    for k in new_graph_features.keys():
        new_graph_features[k] = np.array(new_graph_features[k])
    new_graph_features = parse_graph_features(new_graph_features)

    return new_node_features, new_graph_features


### Parameter Conversion ###

def samples_to_simulation_params(
    samples: np.ndarray,
    labels: List[str],
    dm_type: str,
    stellar_type: str,
    df_type: str,
    dm_params_default: Optional[Dict] = None,
    stellar_params_default: Optional[Dict] = None,
    df_params_default: Optional[Dict] = None,
) -> List[Dict]:
    """Convert parameter samples to simulation parameter format.

    Handles log-space parameters: labels containing 'log_' will be converted
    from log10 space to linear space (e.g., 'dm_log_r_dm' -> 'dm_r_dm').

    Parameters
    ----------
    samples : np.ndarray, shape (num_samples, num_params)
        Parameter samples
    labels : list of str
        Parameter names (may include 'log_' prefix for log-space parameters)
    dm_type : str
        Dark matter potential type
    stellar_type : str
        Stellar density profile type
    df_type : str
        Distribution function type
    dm_params_default : dict, optional
        Default values for DM parameters not in labels
    stellar_params_default : dict, optional
        Default values for stellar parameters not in labels
    df_params_default : dict, optional
        Default values for DF parameters not in labels

    Returns
    -------
    params_list : list of dict
        List of parameter dictionaries ready for simulator
    """
    num_samples = len(samples)
    params_list = []

    # Set defaults to empty dicts if not provided
    dm_params_default = dm_params_default or {}
    stellar_params_default = stellar_params_default or {}
    df_params_default = df_params_default or {}

    for i in range(num_samples):
        # Create parameter dictionaries
        dm_params = {}
        stellar_params = {}
        df_params = {}

        # Fill in sampled parameters
        for j, label in enumerate(labels):
            value = samples[i, j]

            # Handle log-space parameters
            # If label contains 'log_', convert from log10 to linear
            if 'log_' in label:
                # Remove 'log_' from the label
                linear_label = label.replace('log_', '')
                linear_value = 10 ** value
            else:
                linear_label = label
                linear_value = value

            # Parse label to determine which dict it belongs to
            if linear_label.startswith('dm_'):
                param_name = linear_label[3:]  # Remove 'dm_' prefix
                dm_params[param_name] = linear_value
            elif linear_label.startswith('stellar_'):
                param_name = linear_label[8:]  # Remove 'stellar_' prefix
                stellar_params[param_name] = linear_value
            elif linear_label.startswith('df_'):
                param_name = linear_label[3:]  # Remove 'df_' prefix
                df_params[param_name] = linear_value
            else:
                # Try to infer from parameter name
                param_name = linear_label
                if param_name in ['rho_0', 'r_dm', 'alpha', 'beta', 'gamma', 'q', 'p']:
                    dm_params[param_name] = linear_value
                elif param_name in ['r_star', 'r_star_r_dm']:
                    stellar_params[param_name] = linear_value
                elif param_name in ['r_a', 'r_a_r_dm', 'r_a_r_star', 'beta0']:
                    df_params[param_name] = linear_value

        # Fill in default values from config if not already set
        for key, val in dm_params_default.items():
            if key not in dm_params:
                dm_params[key] = val

        for key, val in stellar_params_default.items():
            if key not in stellar_params:
                stellar_params[key] = val

        for key, val in df_params_default.items():
            if key not in df_params:
                df_params[key] = val

        # Construct full parameter dict
        params = {
            'dm_type': dm_type,
            'stellar_type': stellar_type,
            'df_type': df_type,
            'dm_params': dm_params,
            'stellar_params': stellar_params,
            'df_params': df_params,
        }

        params_list.append(params)

    return params_list
