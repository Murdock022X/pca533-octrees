#!/usr/bin/env python3
import argparse
import pathlib

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a domain octree (HDF5 or CSV) exported by src/runner.cu")
    parser.add_argument("input", type=pathlib.Path, help="Path to octree .h5 or .csv file")
    parser.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path("octree_plot.png"))
    parser.add_argument("--title", type=str, help="Title for the plot")
    parser.add_argument("--tree", choices=["focus", "global"], default="focus", help="Octree group to plot for HDF5 input")
    parser.add_argument("--level", type=int, default=None, help="Only plot nodes at this level")
    parser.add_argument("--leaves-only", action="store_true", help="Only plot leaf nodes")
    parser.add_argument(
        "--leaf-sfc-order",
        action="store_true",
        help="Plot leaves in SFC order (implies --leaves-only)",
    )
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


def load_hdf5(path: pathlib.Path, tree: str) -> np.ndarray:
    group_name = "focus_octree" if tree == "focus" else "global_octree"
    with h5py.File(path, "r") as f:
        if group_name not in f:
            raise RuntimeError(f"Group '{group_name}' not found in {path}")
        g = f[group_name]
        n = g["level"].shape[0]
        dtype = [
            ("level", np.uint32),
            ("is_leaf", np.uint32),
            ("cx", np.float64),
            ("cy", np.float64),
            ("cz", np.float64),
            ("sx", np.float64),
            ("sy", np.float64),
            ("sz", np.float64),
        ]
        for name, field_type in [
            ("start_key", np.uint64),
            ("prefixes", np.uint64),
            ("internal_to_leaf", np.int32),
        ]:
            if name in g:
                dtype.append((name, field_type))
        data = np.empty(n, dtype=dtype)
        data["level"] = g["level"][:]
        data["is_leaf"] = g["is_leaf"][:]
        data["cx"] = g["cx"][:]
        data["cy"] = g["cy"][:]
        data["cz"] = g["cz"][:]
        data["sx"] = g["sx"][:]
        data["sy"] = g["sy"][:]
        data["sz"] = g["sz"][:]
        for name in ["start_key", "prefixes", "internal_to_leaf"]:
            if name in data.dtype.names:
                data[name] = g[name][:]
    return data


def load_data(path: pathlib.Path, tree: str) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return load_hdf5(path, tree)
    return load_csv(path)


def filter_rows(
    data: np.ndarray,
    level: int | None,
    leaves_only: bool,
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
    return filtered


def random_subsample(data: np.ndarray, max_nodes: int) -> np.ndarray:
    if data.shape[0] <= max_nodes:
        return data
    rng = np.random.default_rng(0)
    idx = rng.choice(data.shape[0], size=max_nodes, replace=False)
    return data[idx]


def ordered_subsample(data: np.ndarray, max_nodes: int) -> np.ndarray:
    if data.shape[0] <= max_nodes:
        return data
    idx = np.linspace(0, data.shape[0] - 1, num=max_nodes, dtype=int)
    return data[np.unique(idx)]


def leaf_sfc_order(data: np.ndarray) -> np.ndarray:
    names = data.dtype.names or ()
    if "internal_to_leaf" in names:
        return np.argsort(data["internal_to_leaf"], kind="stable")
    if "start_key" in names:
        return np.argsort(data["start_key"], kind="stable")
    if "prefixes" in names:
        return np.argsort(data["prefixes"], kind="stable")
    raise RuntimeError(
        "Cannot determine leaf SFC order: expected one of "
        "'internal_to_leaf', 'start_key', or 'prefixes' in input data"
    )


def plot_projection(
    data: np.ndarray,
    output: pathlib.Path,
    title: str,
    axis_u: str,
    axis_v: str,
) -> None:
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
    cax = ax.inset_axes([1.02, 0.0, 0.035, 1.0], transform=ax.transAxes)
    fig.colorbar(coll, cax=cax, label="octree level")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_leaf_sfc_order(
    data: np.ndarray,
    output: pathlib.Path,
    title: str,
    axis_u: str,
    axis_v: str,
) -> None:
    patches = []
    for row in data:
        u0 = row[f"c{axis_u}"] - row[f"s{axis_u}"]
        v0 = row[f"c{axis_v}"] - row[f"s{axis_v}"]
        wu = 2.0 * row[f"s{axis_u}"]
        hv = 2.0 * row[f"s{axis_v}"]
        patches.append(Rectangle((u0, v0), wu, hv))

    if not patches:
        raise RuntimeError("No nodes remain after filtering; nothing to plot")

    order_rank = np.arange(data.shape[0], dtype=np.float64)
    u_center = data[f"c{axis_u}"]
    v_center = data[f"c{axis_v}"]

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    coll = PatchCollection(patches, cmap="viridis", alpha=0.38, linewidths=0.20, edgecolor="black")
    coll.set_array(order_rank)
    ax.add_collection(coll)
    ax.plot(u_center, v_center, "-", lw=0.8, color="red", alpha=0.8)

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
    cax = ax.inset_axes([1.02, 0.0, 0.035, 1.0], transform=ax.transAxes)
    fig.colorbar(coll, cax=cax, label="Ordering in SFC Curve")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    leaves_only = args.leaves_only or args.leaf_sfc_order
    data = load_data(args.input, args.tree)
    data = filter_rows(data, args.level, leaves_only, args.slice_axis, args.slice_pos)

    if args.slice_axis == "x":
        axis_u, axis_v = "y", "z"
    elif args.slice_axis == "y":
        axis_u, axis_v = "x", "z"
    elif args.slice_axis == "z":
        axis_u, axis_v = "x", "y"
    else:
        axis_u, axis_v = "x", "y"

    if args.leaf_sfc_order:
        data = data[leaf_sfc_order(data)]
        data = ordered_subsample(data, args.max_nodes)
        plot_leaf_sfc_order(data, args.output, args.title, axis_u, axis_v)
    else:
        data = random_subsample(data, args.max_nodes)
        plot_projection(data, args.output, args.title, axis_u, axis_v)

    print(f"Saved plot: {args.output}")
    print(f"Plotted nodes: {data.shape[0]}")


if __name__ == "__main__":
    main()
