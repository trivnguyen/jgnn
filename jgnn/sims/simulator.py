"""Core simulation functionality for generating dwarf galaxy stellar kinematics."""

from typing import Dict, Tuple, List, Optional
import warnings
import numpy as np
import astropy.units as u
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import cpu_count

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
    """Parse simplified parameter names into AGAMA-compatible format.

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
    """Create an AGAMA galaxy model from parsed parameters.

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


def run_simulation(
    params: Dict, num_stars: int, max_iter: int = N_MAX_ITER) -> Tuple[Dict, Dict]:
    """Simulate stellar kinematics for a single dwarf galaxy.

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
    >>> node_features, graph_features = run_simulation(params, num_stars=1000)
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


def run_simulation_batch(
    params_list: List[Dict],
    num_stars_list: List[int],
    max_iter: int = N_MAX_ITER,
    n_jobs: Optional[int] = None,
    use_multiprocessing: bool = False
) -> Tuple[Dict, Dict]:
    """Run simulations for a batch of galaxies.

    Parameters
    ----------
    params_list : list of dict
        List of galaxy parameter dictionaries (see run_simulation)
    num_stars_list : list of int
        List of number of stars to sample for each galaxy
    max_iter : int, optional
        Maximum number of sampling attempts per galaxy (default: 1000)
    n_jobs : int, optional
        Number of parallel workers to use. If None, uses all available CPUs.
        Set to 1 to disable parallelism. Default: None (use all CPUs)
    use_multiprocessing : bool, optional
        Whether to run simulations in parallel. If False, runs sequentially.
        Default: True

    Returns
    -------
    node_features : dict
        Dictionary with keys:
        - 'pos' : list of ndarray - 3D positions for each galaxy
        - 'vel' : list of ndarray - 3D velocities for each galaxy
    graph_features : dict
        Dictionary containing all input parameters with prefixed names
        (dm_*, stellar_*, df_*), each as an ndarray of shape (num_galaxies,)

    Raises
    ------
    RuntimeError
        If all simulations fail, an error is raised.

    Notes
    -----
    Parallelism uses a thread pool, not separate processes: AGAMA releases
    the GIL during sampling, so threads parallelize correctly within a single
    process. A process pool would fork AGAMA's internal RNG state into every
    worker, and (since AGAMA objects aren't passed across the pool boundary
    here anyway) would only add pickling overhead for no benefit.
    """

    num_galaxies = len(params_list)

    # Determine number of workers
    if n_jobs is None:
        n_workers = cpu_count() or 1
    else:
        n_workers = min(n_jobs, num_galaxies)

    # Disable parallelism if requested or if only 1 job
    if not use_multiprocessing or n_workers == 1:
        n_workers = 1
        use_parallel = False
    else:
        use_parallel = True

    print(f"[Simulations] Running {num_galaxies} simulations with {n_workers} worker(s)...")

    all_pos = []
    all_vel = []
    graph_feat_lists = {key: [] for key in []}
    successful_sims = []

    if use_parallel:
        results = {}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(run_simulation, params_list[i], num_stars_list[i], max_iter): i
                for i in range(num_galaxies)
            }
            for future in tqdm(
                as_completed(futures), total=num_galaxies, desc="Simulating galaxies"
            ):
                i = futures[future]
                try:
                    results[i] = future.result()
                except Exception as e:
                    warnings.warn(f"Simulation {i} failed: {str(e)}")
                    results[i] = (None, None)

        # Process results in original order so node/graph features stay
        # aligned with params_list regardless of thread completion order
        for i in range(num_galaxies):
            node_feat, graph_feat = results[i]
            if node_feat is not None and graph_feat is not None:
                all_pos.append(node_feat['pos'])
                all_vel.append(node_feat['vel'])

                # Initialize graph feature lists on first success
                if len(graph_feat_lists) == 0:
                    graph_feat_lists = {key: [] for key in graph_feat.keys()}

                # Append graph features
                for key in graph_feat.keys():
                    graph_feat_lists[key].append(graph_feat[key])

                successful_sims.append(i)
    else:
        # Sequential mode (original implementation)
        for i in tqdm(range(num_galaxies), desc="Simulating galaxies"):
            try:
                node_feat, graph_feat = run_simulation(
                    params_list[i],
                    num_stars_list[i],
                    max_iter=max_iter
                )

                all_pos.append(node_feat['pos'])
                all_vel.append(node_feat['vel'])

                # Initialize graph feature lists on first success
                if len(graph_feat_lists) == 0:
                    graph_feat_lists = {key: [] for key in graph_feat.keys()}

                # Append graph features
                for key in graph_feat.keys():
                    graph_feat_lists[key].append(graph_feat[key])

                successful_sims.append(i)

            except Exception as e:
                warnings.warn(
                    f"Simulation {i} failed after {max_iter} attempts. Error: {str(e)}"
                )
                continue

    # Combine results
    if len(successful_sims) == 0:
        raise RuntimeError("All simulations failed")

    print(f"[Simulations] Completed {len(successful_sims)}/{num_galaxies} simulations successfully")

    # Return node features as lists (not concatenated) for preprocessing
    node_features = {
        'pos': all_pos,
        'vel': all_vel,
    }

    graph_features = {
        key: np.array(values) for key, values in graph_feat_lists.items()
    }

    return node_features, graph_features
