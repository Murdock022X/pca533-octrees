#!/usr/bin/env python
from dist_helpers import *
from run_batches import run_batches
import numpy as np

run_batches(
    generators=[spherical_initial],
    n_particles=[1e4, 1e5, 1e6, 1e7, 1e8],
    scale_factors=[0.001],
    rotations=None,
    out_of_bounds='truncate',
)

# run_batches(
#     generators=[
#         uniform_initial, normal_initial, rectangular_initial,
#         pancake_initial, pancake_initial_tilted, spherical_initial,
#         filament_z_initial, filament_yz_initial, filament_xyz_initial,
#     ],
#     n_particles=[1e4, 1e5, 1e6, 1e7, 1e8],
#     scale_factors=[0.001, 0.01, 0.1, .5],
#     rotations=None,
#     out_of_bounds='truncate',
# )
