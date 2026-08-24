#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import compare_raw_and_retarget_tcp as compare


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot raw and retargeted TCP trajectories for several episodes.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--csv-root", required=True, type=Path)
    parser.add_argument("--episodes", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    parquet = compare.pads.dataset(args.dataset_root / "data", format="parquet")
    figure = compare.plt.figure(figsize=(15, 5.2 * len(args.episodes)), constrained_layout=True)

    for row, episode in enumerate(args.episodes):
        csv_path = (
            args.csv_root
            / f"episode{episode}"
            / f"episode{episode}_index0_g1_pika_retarget_teacher_pose_mujocoik.csv"
        )
        retarget = compare.pd.read_csv(csv_path)
        count = len(retarget)
        table = parquet.to_table(
            columns=["action", "episode_index", "frame_index"],
            filter=compare.pads.field("episode_index") == episode,
        ).sort_by("frame_index")
        actions = compare.np.asarray(table.column("action").to_pylist(), dtype=compare.np.float64)
        raw = compare.integrate_raw_actions(actions)
        target = retarget[["right_target_x", "right_target_y", "right_target_z"]].to_numpy(
            dtype=compare.np.float64
        )
        raw -= raw[0]
        target -= target[0]

        raw_length = float(compare.np.linalg.norm(compare.np.diff(raw, axis=0), axis=1).sum())
        target_length = float(compare.np.linalg.norm(compare.np.diff(target, axis=0), axis=1).sum())

        raw_ax = figure.add_subplot(len(args.episodes), 2, 2 * row + 1, projection="3d")
        compare.colored_line(raw_ax, raw)
        raw_ax.set_title(f"Episode {episode}: before retarget ({count} steps)\npath {raw_length:.3f} m")
        compare.equal_axes(raw_ax, [raw])

        target_ax = figure.add_subplot(len(args.episodes), 2, 2 * row + 2, projection="3d")
        compare.colored_line(target_ax, target)
        target_ax.set_title(f"Episode {episode}: after retarget\npath {target_length:.3f} m")
        compare.equal_axes(target_ax, [target])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    compare.plt.close(figure)
    print(f"plot: {args.output}")


if __name__ == "__main__":
    main()
