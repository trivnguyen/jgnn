
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from networkx.drawing.nx_pydot import graphviz_layout

def convert_to_nx(data):
    # convert PyG graph to a networkx graph manually
    # since the to_networkx method is bugged
    G = nx.Graph()
    G.add_nodes_from(range(len(data.x)))
    G.add_edges_from(data.edge_index.T.numpy())
    for i in range(len(data.x)):
        G.nodes[i]['x'] = data.x[i].numpy()
    return G

def create_nx_graph(halo_id, halo_desc_id, halo_props=None):
    """ Create a directed graph of the halo merger tree.

    Parameters
    ----------
    halo_id : array_like
        Array of halo IDs.
    halo_desc_id : array_like
        Array of halo descendant IDs.
    halo_props : dict or None, optional
        Array of halo properties. If provided, the properties will be added as
        node attributes.

    Returns
    -------
    G : networkx.DiGraph
        A directed graph of the halo merger tree.
    """
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes using indices
    for idx in range(len(halo_id)):
        if halo_props is not None:
            prop_dict = {key: halo_props[key][idx] for key in halo_props.keys()}
            G.add_node(idx, **prop_dict)
        G.add_node(idx)

    # Add edges based on desc_id
    for idx, desc_id in enumerate(halo_desc_id):
        if desc_id != -1:
            parent_idx = np.where(halo_id==desc_id)[0][0] # Find the index of the parent ID
            G.add_edge(parent_idx, idx) # Use indices for edges
    return G

def plot_graph(G, fig_args=None, draw_args=None):
    if isinstance(G, Data):
        G = convert_to_nx(G)

    pos = graphviz_layout(G, prog='dot')
    fig_args = fig_args or {}
    draw_args = draw_args or {}

    fig, ax = plt.subplots(**fig_args)

    default_draw_args = dict(
        with_labels=False,
        node_size=20, node_color="black",
        font_size=10, font_color="black"
    )
    default_draw_args.update(draw_args)
    nx.draw(G, pos, ax=ax, **default_draw_args)
    return fig, ax
