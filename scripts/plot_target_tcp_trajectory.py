#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pika-g1")

import matplotlib
import mpl_toolkits

# Ubuntu's system mpl_toolkits can precede a user-installed Matplotlib and make
# mplot3d ABI-incompatible. Keep the toolkit paired with Matplotlib itself.
paired_toolkits = Path(matplotlib.__file__).resolve().parent.parent / "mpl_toolkits"
if paired_toolkits.is_dir():
    mpl_toolkits.__path__[:] = [str(paired_toolkits)]

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def set_equal_3d_axes(ax: plt.Axes, xyz: np.ndarray) -> None:
    lower = xyz.min(axis=0)
    upper = xyz.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = max(0.5 * float(np.max(upper - lower)), 0.01)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot only the target TCP trajectory from a retarget CSV.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--elev", type=float, default=24.0)
    parser.add_argument("--azim", type=float, default=-58.0)
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    columns = [f"{args.side}_target_{axis}" for axis in "xyz"]
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"missing target columns: {missing}")

    xyz = frame[columns].to_numpy(dtype=np.float64)
    if len(xyz) < 2:
        raise ValueError("at least two trajectory points are required")

    segments = np.stack([xyz[:-1], xyz[1:]], axis=1)
    progress = np.arange(len(segments), dtype=np.float64)
    output = args.output or args.csv.with_name(f"{args.csv.stem}_{args.side}_target_3d.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(9, 8), constrained_layout=True)
    ax = figure.add_subplot(111, projection="3d")
    line = Line3DCollection(segments, cmap="viridis", linewidth=2.5)
    line.set_array(progress)
    ax.add_collection3d(line)
    ax.scatter(*xyz[0], color="#16a34a", s=70, label=f"start ({int(frame.iloc[0]['dataset_index'])})")
    ax.scatter(*xyz[-1], color="#dc2626", s=70, label=f"end ({int(frame.iloc[-1]['dataset_index'])})")

    set_equal_3d_axes(ax, xyz)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(f"{args.side.capitalize()} target TCP trajectory ({len(xyz)} steps)")
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.legend(loc="upper left")
    colorbar = figure.colorbar(line, ax=ax, shrink=0.7, pad=0.08)
    colorbar.set_label("Trajectory step")
    ax.grid(True, alpha=0.35)
    figure.savefig(output, dpi=180)
    plt.close(figure)

    delta = np.diff(xyz, axis=0)
    step_distance = np.linalg.norm(delta, axis=1)
    print(f"plot: {output}")
    print(f"points: {len(xyz)}")
    print(f"start_xyz: {xyz[0]}")
    print(f"end_xyz: {xyz[-1]}")
    print(f"bounds_min: {xyz.min(axis=0)}")
    print(f"bounds_max: {xyz.max(axis=0)}")
    print(f"path_length_m: {step_distance.sum():.6f}")
    print(f"max_step_m: {step_distance.max():.6f}")


if __name__ == "__main__":
    main()
