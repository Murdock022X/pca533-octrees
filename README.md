# pca533-octrees
Parallel octree construction methods, project for CS 533 @ UIUC - Parallel Computer Architecture.

## Export + Plot Domain/Focus Octrees

The CPU and GPU paths export octree snapshots to HDF5:

`outputs/domain_octree_<group>_<stage>_rank<rank>.h5`

Each file contains:

- `domain_box` (xmin, xmax, ymin, ymax, zmin, zmax)
- `global_octree/*` datasets
- `focus_octree/*` datasets

### 0) Clone repository and dependencies

```bash
git clone --recursive https://github.com/Murdock022X/pca533-octrees.git
```

### 1) Run octree build (GPU)

```bash
cmake -S . -B build -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_CUDA_HOST_COMPILER=mpicxx -DCMAKE_CUDA_FLAGS="-I/usr/lib/x86_64-linux-gnu/openmpi/include" -DCMAKE_CXX_FLAGS="-I/usr/include/hdf5/serial" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGPU_DIRECT=ON -DCSTONE_WITH_HIP=OFF

cmake --build ./build -j

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

## Delta (NCSA): run an existing build

If the repo is **already built on Delta** with the same environment you launch under, you do **not** need to reconfigure or rebuild just because you opened a GPU allocation or SSH’d to a compute node (your `build/` directory is usually on shared filesystem).

1. Request GPUs, e.g. `salloc --account=bgnm-delta-gpu --partition=gpuA40x4-interactive --nodes=1 --gpus-per-node=1 --tasks=1 --tasks-per-node=16 --cpus-per-task=1 --mem=20g` (adjust account/partition as needed).
2. **SSH to the allocated compute node** when your site’s workflow requires it, then load the module again if needed.
3. `module load nvhpc-hpcx-cuda12/25.3`
4. From the repo root: `mpirun -n 1 ./build/src/pca --gpu <dataset.h5> <group_name>`
5. **Note the allocation job ID** (while the job exists): `echo $SLURM_JOB_ID`, or list your jobs with `squeue -u $USER`.
6. **End the reservation:** `exit` out of SSH and any shells tied to the allocation; if the job still appears in `squeue`, cancel it with `scancel <jobid>` (from the login node is fine). You can also press Ctrl+C in the terminal where `salloc` is holding the allocation, depending on how that session was started.

Re-run CMake only after **code changes** or after switching to a **different** compiler/MPI/CUDA module than the build used. Configure with `-DCSTONE_WITH_HIP=OFF` on NVIDIA nodes; see [Delta docs](https://docs.ncsa.illinois.edu/systems/delta/) for current Slurm and module names.
