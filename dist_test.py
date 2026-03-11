#!/usr/bin/env python
from dist_helpers import (
    uniform_initial, normal_initial, rectangular_initial, pancake_initial,
    generate_dists,
)
import numpy as np

dist = generate_dists(
    generators=[uniform_initial, normal_initial, rectangular_initial, pancake_initial],
    n_particles=[1e4, 1e5, 1e6],
    scale_factors=[0.001, 0.01, 0.1, .5],
    # rotations=[(0, 0, 0), (np.pi / 4, 0, 0), (0, np.pi / 6, np.pi / 3)],
    rotations=None,
    out_of_bounds='truncate',
)

print(f"\n{'='*60}")
print(f"Generated {len(dist)} distributions:\n")
for i, (name, (x, y, z), _) in enumerate(dist, 1):
    print(f"  {i:3d}. {name}  ({len(x)} particles)")
