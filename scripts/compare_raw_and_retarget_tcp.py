#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pika-g1")

import matplotlib
import mpl_toolkits

paired_toolkits = Path(matplotlib.__file__).resolve().parent.parent / "mpl_toolkits"
if paired_toolkits.is_dir():
    mpl_toolkits.__path__[:] = [str(paired_toolkits)]

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as pads
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def rotation6d_to_matrix(values: np.ndarray) -> np.ndarray:
    col0 = values[:3].astype(np.float64)
    col1 = values[3:6].astype(np.float64)
    col0 /= max(np.linalg.norm(col0), 1e-12)
    col1 -= col0 * float(col0 @ col1)
    col1 /= max(np.linalg.norm(col1), 1e-12)
    return np.stack((col0, col1, np.cross(col0, col1)), axis=1)


def integrate_raw_actions(actions: np.ndarray) -> np.ndarray:
    position = np.zeros(3, dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)
    points = []
    for action in actions:
        position = position + rotation @ action[:3]
        points.append(position.copy())
        rotation = rotation @ rotation6d_to_matrix(action[3:9])
    return np.asarray(points)


def similarity_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    covariance = source.T @ target
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    rotated = source @ rotation.T
    scale = float(np.sum(rotated * target) / np.sum(rotated * rotated))
    return scale * rotated, rotation, scale


def colored_line(ax: plt.Axes, xyz: np.ndarray, color_map: str = "viridis") -> None:
    segments = np.stack((xyz[:-1], xyz[1:]), axis=1)
    line = Line3DCollection(segments, cmap=color_map, linewidth=2.2)
    line.set_array(np.arange(len(segments), dtype=np.float64))
    ax.add_collection3d(line)


def equal_axes(ax: plt.Axes, trajectories: list[np.ndarray]) -> None:
    xyz = np.concatenate(trajectories, axis=0)
    lower = xyz.min(axis=0)
    upper = xyz.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = max(0.5 * float(np.max(upper - lower)), 0.01)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.view_init(elev=24, azim=-58)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare raw PIKA and retargeted target TCP trajectories.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--retarget-csv", required=True, type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=56)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    retarget = pd.read_csv(args.retarget_csv)
    count = len(retarget)
    parquet = pads.dataset(args.dataset_root / "data", format="parquet")
    table = parquet.to_table(
        columns=["action", "episode_index", "frame_index"],
        filter=(pads.field("episode_index") == args.episode)
        & (pads.field("frame_index") >= args.start_index)
        & (pads.field("frame_index") < args.start_index + count),
    ).sort_by("frame_index")
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float64)
    if len(actions) != count:
        raise ValueError(f"action/retarget length mismatch: {len(actions)} != {count}")

    raw = integrate_raw_actions(actions)
    target = retarget[["right_target_x", "right_target_y", "right_target_z"]].to_numpy(dtype=np.float64)
    raw -= raw[0]
    target -= target[0]
    raw_aligned, alignment_rotation, alignment_scale = similarity_align(raw, target)
    residual = np.linalg.norm(raw_aligned - target, axis=1)

    output = args.output or args.retarget_csv.with_name(f"{args.retarget_csv.stem}_raw_comparison.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(19, 6.5), constrained_layout=True)

    raw_ax = figure.add_subplot(131, projection="3d")
    colored_line(raw_ax, raw)
    raw_ax.set_title("Before retargeting\nPIKA native frame")
    equal_axes(raw_ax, [raw])

    target_ax = figure.add_subplot(132, projection="3d")
    colored_line(target_ax, target)
    target_ax.set_title("After retargeting\nG1 target displacement")
    equal_axes(target_ax, [target])

    overlay_ax = figure.add_subplot(133, projection="3d")
    overlay_ax.plot(*raw_aligned.T, color="#2563eb", linewidth=2.5, label="raw, similarity aligned")
    overlay_ax.plot(*target.T, color="#dc2626", linewidth=1.4, linestyle="--", label="retarget target")
    overlay_ax.set_title(
        f"Shape comparison (scale {alignment_scale:.3f})\n"
        f"RMS residual {np.sqrt(np.mean(residual**2)) * 1000:.4f} mm"
    )
    equal_axes(overlay_ax, [raw_aligned, target])
    overlay_ax.legend(loc="upper left", fontsize=9)

    figure.savefig(output, dpi=180)
    plt.close(figure)

    print(f"plot: {output}")
    print(f"points: {count}")
    print(f"rms_shape_residual_m: {np.sqrt(np.mean(residual**2)):.9f}")
    print(f"max_shape_residual_m: {residual.max():.9f}")
    print(f"raw_to_target_scale: {alignment_scale:.9f}")
    print("raw_to_target_rotation:")
    print(alignment_rotation)


if __name__ == "__main__":
    main()
