
import torch
from torch_geometric import transforms as T
from torch_geometric.loader import DataLoader as PyGDataLoader

from .basic import GetNodeFeatures, Normalize
from .graph import ALL_GRAPHS
from .projection import RandomProjection
from .selection_function import RandomSelectionStrategy
from .uncertainty import UncertaintySampler


def build_transformation(
    apply_graph: bool = True,
    apply_projection: bool = False,
    apply_selection: bool = False,
    apply_uncertainty: bool = False,
    recompute_node_features: bool = True,
    graph_name: str = 'KNN',
    graph_args: dict = None,
    projection_args: dict = None,
    selection_args: dict = None,
    uncertainty_args = None,
    norm_dict = None,
    use_log_features: bool = True
):
    """
    Build a transformation pipeline for graph data.

    `uncertainty_args` accepts either a single dict (one uncertainty
    transform applied to a single feature) or a list of dicts to chain
    multiple `UncertaintySampler` transforms, each with its own
    `distribution_type`, `feature_idx`, and parameters. This is useful when
    `apply_projection` is configured with `use_proper_motions=True`, so that
    the line-of-sight velocity and the two proper-motion components can
    each be assigned a different uncertainty distribution.

    `recompute_node_features` controls whether `GetNodeFeatures` (re)builds
    `x` from `pos`/`vel`. It is independent of `apply_projection` and
    `apply_selection` — those two control whether the raw phase-space is
    projected/subselected, while this controls whether `x` gets rebuilt from
    the (possibly projected/selected) `pos`/`vel` afterwards. Set it to False
    for pipelines applied to real observations, where there is no raw 3-D
    phase-space to project or select from and `x` is already provided on
    the graph.
    """

    transforms = []
    transforms.append(T.ToDevice(device=torch.device("cpu")))  # not gpu-supported yet

    # Apply random projection and/or selection function
    if apply_projection:
        transforms.append(RandomProjection(**projection_args))
    if apply_selection:
        if selection_args is None:
            raise ValueError('`selection_args` must be provided when `apply_selection` is True.')
        transforms.append(RandomSelectionStrategy(**selection_args))
    if recompute_node_features:
        transforms.append(GetNodeFeatures(log=use_log_features))

    # Apply uncertainty sampling
    if apply_uncertainty:
        if uncertainty_args is None:
            raise ValueError('`uncertainty_args` must be provided when `apply_uncertainty` is True.')
        # allow a single dict or a list of dicts, one per feature (e.g. one
        # per velocity coordinate), each with its own uncertainty distribution
        if isinstance(uncertainty_args, dict):
            uncertainty_args = [uncertainty_args]
        for args in uncertainty_args:
            transforms.append(UncertaintySampler(**args))

    # Normalizing node features
    if norm_dict is not None:
        # print(f"Applying normalization with provided norm_dict: {norm_dict}")
        transforms.append(Normalize(norm_dict['x_loc'], norm_dict['x_scale']))

    # Apply graph transformation, connect edges based on the specified graph type
    # set to False for no graph construction (e.g., for Transformer models)
    if apply_graph:
        if graph_name.lower() not in ALL_GRAPHS:
            raise ValueError(f"Unknown graph name: {graph_name}. Supported graphs: {list(ALL_GRAPHS.keys())}")
        transforms.append(ALL_GRAPHS[graph_name.lower()](**graph_args))

    transforms = T.Compose(transforms)
    return transforms


def compute_norm_dict(
    graphs, batch_size: int = 256, num_max_graphs=None, **pre_transform_kwargs):
    """Compute `x` normalization stats from the real pre-transform pipeline.

    Runs `graphs` through the same pipeline `build_transformation` would
    build from `pre_transform_kwargs` (graph construction and `Normalize`
    are skipped, since only `x` is needed here), then measures the
    per-feature mean/std of the resulting `x`. This is more accurate than
    hand-approximating normalization stats from raw `pos`/`vel`, since the
    pipeline may include projection, selection, and uncertainty transforms
    that change `x`'s shape and distribution.

    Args:
        graphs: List of PyG `Data` objects with `pos`/`vel` (and optionally
            `vel_error`) set, but no `x`.
        batch_size: Batch size used while streaming `graphs` through the
            pipeline.
        **pre_transform_kwargs: Same kwargs as `build_transformation` (e.g.
            `apply_projection`, `apply_uncertainty`, ...), minus `norm_dict`.

    Returns:
        Tuple of `(x_loc, x_scale)` tensors.
    """
    kwargs = dict(pre_transform_kwargs)
    kwargs['apply_graph'] = False
    kwargs['norm_dict'] = None
    pipeline = build_transformation(**kwargs)

    loader = PyGDataLoader(graphs, batch_size=batch_size, shuffle=False)
    num_graphs_tot = 0
    transformed_graphs = []

    for batch in loader:
        num_graphs_tot += batch.num_graphs
        if num_max_graphs is not None and num_graphs_tot >= num_max_graphs:
            break
        transformed_graphs.append(pipeline(batch))

    x_all = torch.cat([g.x for g in transformed_graphs], dim=0)

    return x_all.mean(dim=0), x_all.std(dim=0)
