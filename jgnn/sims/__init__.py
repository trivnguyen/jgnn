"""
Simulation module for generating dwarf galaxy mock data.

This module provides a simulator for generating stellar kinematics of dwarf
galaxies using the Jeans equations via the AGAMA library. The main interface
is the `run_simulation()` function which takes galaxy parameters and returns
node and graph features suitable for sequential SNPE.
"""

from .simulator import (
    run_simulation,
    run_simulation_batch,
)
from .preprocessing import (
    samples_to_simulation_params,
    preprocess,
)
from .io import (
    write_graph_dataset,
    read_graph_dataset,
    load_observation,
)

__all__ = [
    'run_simulation',
    'run_simulation_batch',
    'samples_to_simulation_params',
    'preprocess',
    'write_graph_dataset',
    'read_graph_dataset',
    'load_observation',
]
