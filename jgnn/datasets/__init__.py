"""jgnn.datasets

Shared HDF5 I/O utilities plus two dataset streams:

  cartesian  — 3-D Cartesian phase-space (sample_galaxies.py output)
               pos=(x,y,z), vel=(vx,vy,vz); `x` is built later by the
               pre_transform pipeline (see jgnn.transforms.GetNodeFeatures)

  icrs       — sky-plane ICRS observables (sample_galaxies_target.py output)
               pos=(ra,dec), vel=(vlos,), x=[log10(R_proj), vlos]
"""

from . import cartesian
from . import icrs
from .io import read_graph_dataset, read_datasets

__all__ = [
    'cartesian',
    'icrs',
    'read_graph_dataset',
    'read_datasets',
]
