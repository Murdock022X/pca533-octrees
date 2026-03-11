#!/usr/bin/env python
# Batch runner: lazily generate distributions per-batch, profile with nsys, plot octree output, clean up between batches.

import glob as globmod
import re
import shutil
import subprocess
import sys
from pathlib import Path

import h5py as h5
import numpy as np

from dist_helpers import (
    _dist_name, _format_particle_count, _format_scale, _format_rotation,
    _rotation_matrix, perturb,
)

BUILD_DIR = Path("build")
PCA_BINARY = BUILD_DIR / "src" / "pca"
NSYS_BINARY = Path("/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/bin/nsys")
PARTICLES_H5 = Path("particles.h5")
OUTPUTS_DIR = Path("outputs")
PLOT_SCRIPT = Path("scripts/plot_domain_octree.py")


# Derive hierarchical folder path from a distribution name
def _name_to_folder(name):
    match = re.match(r'^(.+?)_(s[^_]+)_(n.+)$', name)
    if match:
        dist_part, scale_part, n_part = match.groups()
        return OUTPUTS_DIR / dist_part / scale_part / n_part
    return OUTPUTS_DIR / name


# Plan all configs (names + metadata) without generating any particle data
def _plan(generators, n_particles, scale_factors, rotations, out_of_bounds):
    if rotations is None:
        rotations = [(0, 0, 0)]
    has_nontrivial_rot = any(any(r != 0 for r in rot) for rot in rotations)
    configs = []
    for gen in generators:
        dname = _dist_name(gen)
        for scale in scale_factors:
            scale_str = _format_scale(scale)
            for rot in rotations:
                is_rotated = any(r != 0 for r in rot)
                rot_tag = ""
                if has_nontrivial_rot and is_rotated:
                    rot_tag = _format_rotation(rot)
                for n in n_particles:
                    n_str = _format_particle_count(n)
                    if rot_tag:
                        name = f"{dname}_{rot_tag}_{scale_str}_n{n_str}"
                    else:
                        name = f"{dname}_{scale_str}_n{n_str}"
                    configs.append({
                        "name": name, "gen": gen, "n": int(n),
                        "scale": scale, "rot": rot,
                        "is_rotated": is_rotated,
                        "out_of_bounds": out_of_bounds,
                    })
    return configs


# Generate one distribution and write it into an open h5 file
def _generate_one(f, cfg):
    name = cfg["name"]
    print(f"  Generating {name} ...", end=" ", flush=True)
    x, y, z = cfg["gen"](cfg["n"])
    if cfg["is_rotated"]:
        coords = _rotation_matrix(*cfg["rot"]) @ np.stack([x, y, z])
        x, y, z = coords[0], coords[1], coords[2]
        if cfg["out_of_bounds"] == "truncate":
            x, y, z = np.clip(x, -1, 1), np.clip(y, -1, 1), np.clip(z, -1, 1)
        elif cfg["out_of_bounds"] == "remove":
            mask = (np.abs(x) <= 1) & (np.abs(y) <= 1) & (np.abs(z) <= 1)
            x, y, z = x[mask], y[mask], z[mask]
    px, py, pz = perturb(len(x), cfg["scale"])
    g = f.create_group(name)
    g.create_dataset("ix", data=x)
    g.create_dataset("iy", data=y)
    g.create_dataset("iz", data=z)
    g.create_dataset("px", data=px)
    g.create_dataset("py", data=py)
    g.create_dataset("pz", data=pz)
    print("done", flush=True)


# Run pca under nsys (single GPU, single node)
def _run_pca(name, nsys_output):
    cmd = [
        "mpiexec", "-np", "1",
        str(NSYS_BINARY), "profile",
        "-o", str(nsys_output),
        "--trace=cuda,nvtx",
        "--nvtx-capture", "Initial",
        str(PCA_BINARY), "--gpu", "--save",
        str(PARTICLES_H5), name,
    ]
    print(f"  $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    return result.returncode


# Move nsys-rep files into the output folder
def _collect_nsys_rep(nsys_output, output_folder):
    pattern = f"{nsys_output}*.nsys-rep"
    for rep in sorted(globmod.glob(pattern)):
        dest = output_folder / Path(rep).name
        print(f"  Moving {rep} -> {dest}", flush=True)
        shutil.move(rep, str(dest))


# Generate octree plots from any h5 files the pca binary wrote to outputs/
def _generate_plots(name, output_folder):
    pattern = str(OUTPUTS_DIR / f"domain_{name}_*rank*.h5")
    h5_files = sorted(globmod.glob(pattern))
    if not h5_files:
        print(f"  No octree h5 files found for {name}, skipping plots")
        return
    for h5_file in h5_files:
        h5_path = Path(h5_file)
        stem = h5_path.stem
        for tree in ("global",):
            png_name = f"{stem}_{tree}.png"
            png_path = output_folder / png_name
            cmd = [
                sys.executable, str(PLOT_SCRIPT),
                str(h5_path),
                "--tree", tree,
                "-o", str(png_path),
                "--title", f"{name} ({tree})",
            ]
            print(f"  Plotting {png_name} ...", end=" ", flush=True)
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                print("done", flush=True)
            else:
                print(f"FAILED (rc={r.returncode})", flush=True)
                if r.stderr:
                    print(f"    {r.stderr.strip()}")
        h5_path.unlink(missing_ok=True)
    leftover = sorted(globmod.glob(str(OUTPUTS_DIR / "domain_*.h5")))
    for f in leftover:
        Path(f).unlink(missing_ok=True)


# Lazily generate distributions in batches, profile, plot, clean up
def run_batches(generators, n_particles, scale_factors, rotations=None,
                out_of_bounds="truncate", batch_size=5):
    if not PCA_BINARY.exists():
        print(f"ERROR: {PCA_BINARY} not found. Build the project first.")
        sys.exit(1)

    configs = _plan(generators, n_particles, scale_factors, rotations, out_of_bounds)
    total = len(configs)
    n_batches = (total + batch_size - 1) // batch_size
    print(f"Total configurations: {total}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {n_batches}")

    for batch_start in range(0, total, batch_size):
        batch = configs[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1

        print(f"\n{'='*70}")
        print(f"  BATCH {batch_num}/{n_batches}  ({len(batch)} distributions)")
        print(f"{'='*70}")

        print(f"\nWriting {len(batch)} distributions to {PARTICLES_H5} ...")
        with h5.File(str(PARTICLES_H5), "w") as f:
            for cfg in batch:
                _generate_one(f, cfg)

        for cfg in batch:
            name = cfg["name"]
            folder = _name_to_folder(name)
            folder.mkdir(parents=True, exist_ok=True)

            print(f"\n--- {name} ---")
            nsys_output = Path(name)
            rc = _run_pca(name, nsys_output)
            if rc != 0:
                print(f"  WARNING: pca exited with code {rc}")
            _generate_plots(name, folder)
            _collect_nsys_rep(nsys_output, folder)

        print(f"\nCleaning up {PARTICLES_H5} ...")
        PARTICLES_H5.unlink(missing_ok=True)

    print(f"\n{'='*70}")
    print("All batches complete!")
    print(f"{'='*70}")
