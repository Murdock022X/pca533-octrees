# pca533-octrees
Parallel octree construction methods, project for CS 533 @ UIUC - Parallel Computer Architecture.

## Export + Plot Domain Octree

The GPU path now exports the domain octree to CSV at:

`outputs/domain_octree_<group>_rank<rank>.csv`

### 1) Run octree build (GPU)

```bash
mpirun -n 1 ./build/src/pca --gpu <dataset.h5> <group_name>
```

### 2) Plot with matplotlib

```bash
python3 scripts/plot_domain_octree.py outputs/domain_octree_<group_name>_rank0.csv -o outputs/octree_<group_name>.png --leaves-only
```

Slice examples:

```bash
python3 scripts/plot_domain_octree.py outputs/domain_octree_<group_name>_rank0.csv -o outputs/octree_slice_x.png --slice-axis x --slice-pos 0.5 --leaves-only
python3 scripts/plot_domain_octree.py outputs/domain_octree_<group_name>_rank0.csv -o outputs/octree_slice_y.png --slice-axis y --slice-pos 0.5 --leaves-only
python3 scripts/plot_domain_octree.py outputs/domain_octree_<group_name>_rank0.csv -o outputs/octree_slice_z.png --slice-axis z --slice-pos 0.5 --leaves-only
```

Optional filters:

- `--level <n>`: plot only one octree level.
- `--max-nodes <N>`: cap rendered nodes for very large trees.
- `--slice-axis {x,y,z}` + `--slice-pos <value>`: keep only cells intersecting the slice.
