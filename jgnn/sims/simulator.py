"""
Core simulation functionality for generating dwarf galaxy stellar kinematics.
"""
from typing import Dict, Tuple, Optional

import warnings
import numpy as np
import astropy.units as u

# Optional agama import with graceful error handling
_AGAMA_AVAILABLE = False
_AGAMA_IMPORT_ERROR = None

try:
    import agama
    # Set agama units to Msun, kpc, km/s
    agama.setUnits(mass=1 * u.Msun, length=1 * u.kpc, velocity=1 * u.km / u.s)
    _AGAMA_AVAILABLE = True
except ImportError as e:
    _AGAMA_IMPORT_ERROR = str(e)
    warnings.warn(
        f"AGAMA could not be imported. Simulation functionality will be unavailable. "
        f"Original error: {_AGAMA_IMPORT_ERROR}"
    )
    agama = None

from . import utils

# agama sometimes fails to sample (not sure why), so we allow multiple attempts
# Maximum iterations for sampling before giving up
N_MAX_ITER = 1000


def _check_agama():
    """Check if agama is available and raise an error if not."""
    if not _AGAMA_AVAILABLE:
        raise ImportError(
            f"AGAMA is required for simulations but could not be imported. "
            f"Original error: {_AGAMA_IMPORT_ERROR}\n"
            f"Please install AGAMA to use the sims module. "
            f"See https://github.com/GalacticDynamics-Oxford/Agama for installation instructions."
        )


def _parse_parameters(
    dm_params: Dict, stellar_params: Dict, df_params: Dict) -> Tuple[Dict, Dict, Dict]:
    """
    Parse simplified parameter names into AGAMA-compatible format.

    This handles conversion of user-friendly parameter names to AGAMA conventions
    and resolves relative parameter specifications (e.g., r_star_r_dm).

    Parameters
    ----------
    dm_params : dict
        DM potential parameters. Should include 'rho_0' (density norm) and
        'r_dm' (scale radius), and optionally 'q' and 'p' (axis ratios)
    stellar_params : dict
        Stellar density parameters. Can specify 'r_star' directly or
        'r_star_r_dm' as ratio. Similar for 'q' and 'p'
    df_params : dict
        Distribution function parameters. Can specify 'r_a' directly or
        as ratios 'r_a_r_dm' or 'r_a_r_star'

    Returns
    -------
    dm_params_agama : dict
        AGAMA-formatted DM parameters
    stellar_params_agama : dict
        AGAMA-formatted stellar parameters
    df_params_agama : dict
        AGAMA-formatted DF parameters
    """
    # Make copies to avoid modifying input dicts
    dm_params = dm_params.copy()
    stellar_params = stellar_params.copy()
    df_params = df_params.copy()

    # Parse DM parameters: rename to AGAMA conventions
    dm_params['densityNorm'] = dm_params.pop('rho_0')
    dm_params['scaleRadius'] = dm_params.pop('r_dm')
    dm_params['axisRatioY'] = dm_params.pop('q', 1)
    dm_params['axisRatioZ'] = dm_params.pop('p', 1)

    # Parse stellar parameters: handle both direct and relative specifications
    if stellar_params.get('r_star') is not None:
        stellar_params['scaleRadius'] = stellar_params.pop('r_star')
    elif stellar_params.get('r_star_r_dm') is not None:
        stellar_params['scaleRadius'] = (
            stellar_params.pop('r_star_r_dm') * dm_params['scaleRadius'])

    if stellar_params.get('q') is not None:
        stellar_params['axisRatioY'] = stellar_params.pop('q')
    elif stellar_params.get('q_star_q_dm') is not None:
        stellar_params['axisRatioY'] = (
            stellar_params.pop('q_star_q_dm') * dm_params['axisRatioY'])

    if stellar_params.get('p') is not None:
        stellar_params['axisRatioZ'] = stellar_params.pop('p')
    elif stellar_params.get('p_star_p_dm') is not None:
        stellar_params['axisRatioZ'] = (
            stellar_params.pop('p_star_p_dm') * dm_params['axisRatioZ'])

    # Parse DF parameters: anisotropy radius
    if df_params.get('r_a') is not None:
        df_params['r_a'] = df_params.pop('r_a')
    elif df_params.get('r_a_r_dm') is not None:
        df_params['r_a'] = (
            df_params.pop('r_a_r_dm') * dm_params['scaleRadius'])
    elif df_params.get('r_a_r_star') is not None:
        df_params['r_a'] = (
            df_params.pop('r_a_r_star') * stellar_params['scaleRadius'])

    return dm_params, stellar_params, df_params


def create_galaxy_model(
    dm_type: str, stellar_type: str, df_type: str,
    dm_params: Dict, stellar_params: Dict, df_params: Dict
):
    """
    Create an AGAMA galaxy model from parsed parameters.

    Parameters
    ----------
    dm_type : str
        Dark matter potential type (e.g., 'NFW', 'Dehnen')
    stellar_type : str
        Stellar density profile type (e.g., 'Plummer', 'Dehnen')
    df_type : str
        Distribution function type (e.g., 'QuasiSpherical')
    dm_params : dict
        AGAMA-formatted DM parameters
    stellar_params : dict
        AGAMA-formatted stellar parameters
    df_params : dict
        AGAMA-formatted DF parameters

    Returns
    -------
    agama.GalaxyModel
        Configured galaxy model ready for sampling
    """
    _check_agama()

    # Construct AGAMA objects
    dm_potential = agama.Potential(type=dm_type, **dm_params)
    stellar_density = agama.Potential(
        type=stellar_type, mass=1, **stellar_params)
    df = agama.DistributionFunction(
        type=df_type, potential=dm_potential, density=stellar_density,
        **df_params)

    # Create galaxy model
    galaxy_model = agama.GalaxyModel(dm_potential, df)

    return galaxy_model


def run_simulations(
    params: Dict, num_stars: int, max_iter: int = N_MAX_ITER) -> Tuple[Dict, Dict]:
    """
    Simulate stellar kinematics for a single dwarf galaxy.

    This is the main interface for generating mock galaxy data. It takes
    physical parameters describing the galaxy's dark matter halo, stellar
    distribution, and velocity anisotropy, and returns sampled stellar
    positions and velocities.

    Parameters
    ----------
    params : dict
        Galaxy parameters with keys:
        - 'dm_type' : str - Dark matter potential type
        - 'stellar_type' : str - Stellar density profile type
        - 'df_type' : str - Distribution function type
        - 'dm_params' : dict - DM parameters (rho_0, r_dm, q, p)
        - 'stellar_params' : dict - Stellar parameters (r_star or r_star_r_dm, etc)
        - 'df_params' : dict - DF parameters (r_a or ratios)
    num_stars : int
        Number of stellar particles to sample
    max_iter : int, optional
        Maximum number of sampling attempts before giving up (default: 1000)

    Returns
    -------
    node_features : dict
        Dictionary with keys:
        - 'pos' : ndarray, shape (num_stars, 3) - 3D positions in kpc
        - 'vel' : ndarray, shape (num_stars, 3) - 3D velocities in km/s
    graph_features : dict
        Dictionary containing all input parameters with prefixed names
        (dm_*, stellar_*, df_*)

    Raises
    ------
    ImportError
        If AGAMA is not available
    RuntimeError
        If sampling fails after max_iter attempts

    Examples
    --------
    >>> params = {
    ...     'dm_type': 'NFW',
    ...     'stellar_type': 'Plummer',
    ...     'df_type': 'QuasiSpherical',
    ...     'dm_params': {'rho_0': 0.01, 'r_dm': 1.0},
    ...     'stellar_params': {'r_star': 0.3},
    ...     'df_params': {'r_a': 0.5}
    ... }
    >>> node_features, graph_features = simulator(params, num_stars=1000)
    """
    _check_agama()

    # Extract parameter groups
    dm_type = params['dm_type']
    stellar_type = params['stellar_type']
    df_type = params['df_type']
    dm_params = params['dm_params']
    stellar_params = params['stellar_params']
    df_params = params['df_params']

    # Parse parameters to AGAMA format
    dm_params_agama, stellar_params_agama, df_params_agama = _parse_parameters(
        dm_params, stellar_params, df_params
    )

    # Attempt to sample with retries (AGAMA can occasionally fail)
    success = False
    for iteration in range(max_iter):
        try:
            galaxy_model = create_galaxy_model(
                dm_type, stellar_type, df_type,
                dm_params_agama, stellar_params_agama, df_params_agama
            )
            posvel, _ = galaxy_model.sample(num_stars)
            success = True
            break
        except Exception as e:
            if iteration == max_iter - 1:
                # Last iteration, re-raise the error
                raise RuntimeError(
                    f"Failed to sample galaxy after {max_iter} attempts. "
                    f"Last error: {str(e)}"
                ) from e
            # Otherwise continue to next iteration
            continue
        finally:
            # Clean up to avoid potential memory leaks in AGAMA
            if 'galaxy_model' in locals():
                del galaxy_model

    # Package node features
    node_features = {
        'pos': posvel[:, :3].astype(np.float32),
        'vel': posvel[:, 3:].astype(np.float32)
    }

    # Package graph features with prefixes to avoid name collisions
    graph_features = {}
    for k, v in dm_params.items():
        graph_features[f'dm_{k}'] = v
    for k, v in stellar_params.items():
        graph_features[f'stellar_{k}'] = v
    for k, v in df_params.items():
        graph_features[f'df_{k}'] = v

    return node_features, graph_features

def preprocess(
    node_features: Dict,
    graph_features: Dict,
    vrange: Tuple[float, float] = (0, np.inf),
    vdisp_range: Tuple[float, float] = (0, np.inf),
    r_range: Tuple[float, float] = (0, np.inf),
    r_rstar_range: Tuple[float, float] = (0, np.inf),
    apply_projection: bool = True,
    projection_axis: Optional[int] = None,
    use_proper_motions: bool = False,
    norm_rstar: bool = False,
    seed: Optional[int] = None
) -> Tuple[Dict, Dict]:
    """ Preprocess the raw simulation data into training data. Applies
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
    norm_rstar : bool
        Whether to normalize positions by stellar rstar.
    seed : int, optional
        Random seed for reproducibility.

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
    new_graph_features['cond'] = []
    new_graph_features['num_stars'] = []

    for i in range(num_galaxies):
        nodes, graph = utils.get_graph(node_features, graph_features, i)
        pos = nodes['pos'].astype(np.float32)
        vel = nodes['vel'].astype(np.float32)
        stellar_rstar = graph['stellar_r_star_r_dm'] * graph['dm_r_dm']

        # apply velocity cut on the 3d velocity
        # large velocity can be due to AGAMA sampling issues
        vel3d = np.linalg.norm(vel, axis=1)
        mask = (vel3d > vrange[0]) & (vel3d < vrange[1])
        if np.sum(mask) < len(pos) * 0.5:
            # if half of the stars are outside the velocity range, skip this galaxy
            print(f'Skipping galaxy {i} due to velocity cut')
            continue
        pos = pos[mask]
        vel = vel[mask]

        # apply velocity dispersion cut on the 3d velocity dispersion
        vdisp = np.std(vel, axis=0)
        vdisp = np.linalg.norm(vdisp)
        if vdisp < vdisp_range[0] or vdisp > vdisp_range[1]:
            print(f'Skipping galaxy {i} due to velocity dispersion cut')
            continue

        # project onto 2D plane
        if apply_projection:
            pos, vel = utils.project2d(
                pos, vel, axis=projection_axis, use_proper_motions=use_proper_motions)

        # apply radius cut
        min_radius = max(r_rstar_range[0] * stellar_rstar, r_range[0])
        max_radius = min(r_rstar_range[1] * stellar_rstar, r_range[1])
        radius = np.linalg.norm(pos, axis=1)
        mask = (radius > min_radius) & (radius < max_radius)

        if np.sum(mask) < len(pos) * 0.5:
            # if half of the stars are outside the radius range, skip this galaxy
            print(f'Skipping galaxy {i} due to radius cut')
            continue

        pos = pos[mask]
        vel = vel[mask]
        radius = radius[mask]

        if norm_rstar:
            pos = pos / stellar_rstar

        new_node_features['pos'].append(pos)
        new_node_features['vel'].append(vel)
        new_node_features['vel_true'].append(vel)
        new_node_features['vel_error'].append(np.zeros_like(vel))

        for k in graph.keys():
            new_graph_features[k].append(graph[k])
        new_graph_features['num_stars'].append(len(pos))
        new_graph_features['cond'].append(np.log10(stellar_rstar))

    # Finalize node and graph features
    if len(new_node_features['pos']) == 0:
        raise ValueError("All galaxies were filtered out during preprocessing")

    for k in new_node_features.keys():
        new_node_features[k] = np.concatenate(new_node_features[k])
    for k in new_graph_features.keys():
        new_graph_features[k] = np.array(new_graph_features[k])
    new_graph_features = utils.parse_graph_features(
        new_graph_features, norm_rstar=norm_rstar)

    return new_node_features, new_graph_features
