"""
Simulation module for generating dwarf galaxy mock data.

This module provides a simulator for generating stellar kinematics of dwarf
galaxies using the Jeans equations via the AGAMA library. The main interface
is the `simulator()` function which takes galaxy parameters and returns
node and graph features suitable for sequential SNPE.
"""

from .simulator import run_simulations

__all__ = ['run_simulation']
