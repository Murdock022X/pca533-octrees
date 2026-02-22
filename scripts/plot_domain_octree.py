#!/usr/bin/env python3
import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a domain octree CSV exported by src/runner.cu")
    parser.add_argument("csv", type=pathlib.Path, help="Path to octree CSV file")
    parser.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path("octree_plot.png"))
    parser.add_argument("--level", type=int, default=None, help="Only plot nodes at this level")
    parser.add_argument("--leaves-only", action="store_true", help="Only plot leaf nodes")
    parser.add_argument("--slice-axis", choices=["x", "y", "z"], default=None, help="Slice axis")
    parser.add_argument("--slice-pos", type=float, default=None, help="Slice position on the selected axis")
    parser.add_argument("--max-nodes", type=int, default=25000, help="Maximum nodes to draw")
    args = parser.parse_args()
    if (args.slice_axis is None) != (args.slice_pos is None):
        parser.error("--slice-axis and --slice-pos must be provided together")
    return args


def load_csv(path: pathlib.Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        raise RuntimeError(f"No rows found in {path}")
    if data.ndim == 0:
        data = data.reshape(1)
    return data


def filter_rows(
    data: np.ndarray,
    level: int | None,
    leaves_only: bool,
    max_nodes: int,
    slice_axis: str | None,
    slice_pos: float | None,
) -> np.ndarray:
    mask = np.ones(data.shape[0], dtype=bool)
    if level is not None:
        mask &= data["level"] == level
    if leaves_only:
        mask &= data["is_leaf"] == 1
    if slice_axis is not None and slice_pos is not None:
        half = data[f"s{slice_axis}"]
        center = data[f"c{slice_axis}"]
        mask &= (slice_pos >= center - half) & (slice_pos <= center + half)

    filtered = data[mask]
    if filtered.shape[0] > max_nodes:
        rng = np.random.default_rng(0)
        idx = rng.choice(filtered.shape[0], size=max_nodes, replace=False)
        filtered = filtered[idx]
    return filtered


def plot_projection(data: np.ndarray, output: pathlib.Path, title: str, axis_u: str, axis_v: str) -> None:
    patches = []
    colors = []
    for row in data:
        u0 = row[f"c{axis_u}"] - row[f"s{axis_u}"]
        v0 = row[f"c{axis_v}"] - row[f"s{axis_v}"]
        wu = 2.0 * row[f"s{axis_u}"]
        hv = 2.0 * row[f"s{axis_v}"]
        patches.append(Rectangle((u0, v0), wu, hv))
        colors.append(row["level"])

    if not patches:
        raise RuntimeError("No nodes remain after filtering; nothing to plot")

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    coll = PatchCollection(patches, cmap="viridis", alpha=0.30, linewidths=0.20, edgecolor="black")
    coll.set_array(np.asarray(colors))
    ax.add_collection(coll)

    umin = np.min(data[f"c{axis_u}"] - data[f"s{axis_u}"])
    umax = np.max(data[f"c{axis_u}"] + data[f"s{axis_u}"])
    vmin = np.min(data[f"c{axis_v}"] - data[f"s{axis_v}"])
    vmax = np.max(data[f"c{axis_v}"] + data[f"s{axis_v}"])
    ax.set_xlim(float(umin), float(umax))
    ax.set_ylim(float(vmin), float(vmax))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(axis_u)
    ax.set_ylabel(axis_v)
    ax.set_title(title)
    fig.colorbar(coll, ax=ax, label="octree level")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data = load_csv(args.csv)
    data = filter_rows(data, args.level, args.leaves_only, args.max_nodes, args.slice_axis, args.slice_pos)

    if args.slice_axis == "x":
        axis_u, axis_v = "y", "z"
        title = f"Domain Octree slice x={args.slice_pos:g} ({args.csv.name})"
    elif args.slice_axis == "y":
        axis_u, axis_v = "x", "z"
        title = f"Domain Octree slice y={args.slice_pos:g} ({args.csv.name})"
    elif args.slice_axis == "z":
        axis_u, axis_v = "x", "y"
        title = f"Domain Octree slice z={args.slice_pos:g} ({args.csv.name})"
    else:
        axis_u, axis_v = "x", "y"
        title = f"Domain Octree XY Projection ({args.csv.name})"

    plot_projection(data, args.output, title, axis_u, axis_v)
    print(f"Saved plot: {args.output}")
    print(f"Plotted nodes: {data.shape[0]}")


if __name__ == "__main__":
    main()
