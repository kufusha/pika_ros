#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pika-g1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.dataset as pads


def rotation6d_to_matrix(values: np.ndarray) -> np.ndarray:
    x_axis = values[:3].astype(np.float64)
    y_axis = values[3:6].astype(np.float64)
    x_axis /= max(np.linalg.norm(x_axis), 1e-12)
    y_axis -= x_axis * float(x_axis @ y_axis)
    y_axis /= max(np.linalg.norm(y_axis), 1e-12)
    return np.stack((x_axis, y_axis, np.cross(x_axis, y_axis)), axis=1)


def matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    pitch = np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))
    roll = np.arctan2(rotation[2, 1], rotation[2, 2])
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    return np.array([roll, pitch, yaw], dtype=np.float64)


def integrate_actions(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    position = np.zeros(3, dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)
    positions = [position.copy()]
    rotations = [rotation.copy()]
    for action in actions:
        position = position + rotation @ action[:3]
        rotation = rotation @ rotation6d_to_matrix(action[3:9])
        positions.append(position.copy())
        rotations.append(rotation.copy())
    rpy = np.unwrap(np.stack([matrix_to_rpy(value) for value in rotations]), axis=0)
    return np.stack(positions), rpy


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the raw PIKA pose before G1 frame conversion.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--episode", type=int, default=10)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    dataset = pads.dataset(args.dataset_root / "data", format="parquet")
    table = dataset.to_table(
        columns=["action", "episode_index", "frame_index"],
        filter=pads.field("episode_index") == args.episode,
    ).sort_by("frame_index")
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 10:
        raise ValueError(f"expected 10D single-PIKA actions, got {actions.shape}")

    xyz, rpy = integrate_actions(actions)
    seconds = np.arange(len(xyz), dtype=np.float64) / args.fps
    args.output.parent.mkdir(parents=True, exist_ok=True)

    colors = ("#dc2626", "#16a34a", "#2563eb")
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    for index, (label, color) in enumerate(zip(("X", "Y", "Z"), colors, strict=True)):
        axes[0].plot(seconds, xyz[:, index], color=color, linewidth=2.0, label=label)
    axes[0].scatter([0], [0], color="#111827", s=45, zorder=5, label="initial pose")
    axes[0].set_ylabel("Position displacement [m]")
    axes[0].set_title(f"Episode {args.episode}: raw PIKA pose before G1 frame conversion")
    axes[0].legend(ncol=4, loc="upper left")

    for index, (label, color) in enumerate(zip(("roll", "pitch", "yaw"), colors, strict=True)):
        axes[1].plot(seconds, rpy[:, index], color=color, linewidth=2.0, label=label)
    axes[1].scatter([0], [0], color="#111827", s=45, zorder=5, label="initial pose")
    axes[1].set_xlabel(
        f"Time [s] at {args.fps:g} fps ({len(xyz)} samples including initial pose)"
    )
    axes[1].set_ylabel("Orientation displacement [rad]")
    axes[1].legend(ncol=4, loc="upper left")

    for axis in axes:
        axis.axhline(0.0, color="#6b7280", linewidth=0.8)
        axis.grid(True, alpha=0.3)

    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(f"plot: {args.output}")
    print(f"actions: {len(actions)}")
    print(f"initial_xyz_rpy: {np.zeros(6)}")
    print(f"final_xyz: {xyz[-1]}")
    print(f"final_rpy: {rpy[-1]}")


if __name__ == "__main__":
    main()
