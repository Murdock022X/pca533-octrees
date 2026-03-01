# pca533-octrees
Parallel octree construction methods, project for CS 533 @ UIUC - Parallel Computer Architecture.

## Export + Plot Domain/Focus Octrees

The CPU and GPU paths export octree snapshots to HDF5:

`outputs/domain_octree_<group>_<stage>_rank<rank>.h5`

Each file contains:

- `domain_box` (xmin, xmax, ymin, ymax, zmin, zmax)
- `global_octree/*` datasets
- `focus_octree/*` datasets

### 1) Run octree build (GPU)

```bash
mpirun -n 1 ./build/src/pca --gpu <dataset.h5> <group_name>
```

### 2) Plot with matplotlib

```bash
python3 scripts/plot_domain_octree.py outputs/domain_octree_<group_name>_initial_rank0.h5 -o outputs/octree_<group_name>.png --leaves-only --tree focus
```

Slice examples:

```bash
python3 scripts/plot_domain_octree.py outputs/domain_octree_<group_name>_initial_rank0.h5 -o outputs/octree_slice_x.png --slice-axis x --slice-pos 0.5 --leaves-only --tree focus
python3 scripts/plot_domain_octree.py outputs/domain_octree_<group_name>_initial_rank0.h5 -o outputs/octree_slice_y.png --slice-axis y --slice-pos 0.5 --leaves-only --tree focus
python3 scripts/plot_domain_octree.py outputs/domain_octree_<group_name>_initial_rank0.h5 -o outputs/octree_slice_z.png --slice-axis z --slice-pos 0.5 --leaves-only --tree focus
```

Leaf SFC order example:

```bash
python3 scripts/plot_domain_octree.py outputs/domain_octree_<group_name>_initial_rank0.h5 -o outputs/octree_leaf_sfc.png --tree focus --leaf-sfc-order
```

Optional filters:

- `--level <n>`: plot only one octree level.
- `--max-nodes <N>`: cap rendered nodes for very large trees.
- `--slice-axis {x,y,z}` + `--slice-pos <value>`: keep only cells intersecting the slice.
- `--tree {focus,global}`: choose which HDF5 octree group to plot.
- `--leaf-sfc-order`: force leaves-only view and color/order leaves by SFC traversal.
