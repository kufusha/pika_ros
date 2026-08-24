#!/usr/bin/env python3
"""Search a G1 right-arm start pose that can reproduce a full PIKA pose trajectory."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.unitree_g1_pika.g1_pika_kinematics import G1PikaArmIK


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from g1_pika_mujoco_retarget import action_pose  # noqa: E402


@dataclass
class CandidateResult:
    xyz_offset: np.ndarray
    yaw_offset_rad: float
    initial_q: np.ndarray
    q_path: np.ndarray
    position_errors: np.ndarray
    rotation_errors: np.ndarray
    failed_step: int | None
    minimum_joint_margin_rad: float

    @property
    def valid_steps(self) -> int:
        return len(self.position_errors) if self.failed_step is None else self.failed_step

    @property
    def passed(self) -> bool:
        return self.failed_step is None

    def score(self) -> tuple[float, ...]:
        return (
            0.0 if self.passed else 1.0,
            -float(self.valid_steps),
            -self.minimum_joint_margin_rad,
            float(np.max(self.position_errors, initial=0.0)),
            float(np.max(self.rotation_errors, initial=0.0)),
        )

    @property
    def maximum_joint_step_rad(self) -> float:
        if self.q_path.size == 0:
            return 0.0
        full_path = np.vstack([self.initial_q, self.q_path])
        return float(np.max(np.abs(np.diff(full_path, axis=0)), initial=0.0))


def parse_vec(text: str, length: int) -> np.ndarray:
    values = np.fromstring(text.replace(",", " "), sep=" ", dtype=np.float64)
    if values.shape != (length,):
        raise argparse.ArgumentTypeError(f"expected {length} values, got {values.size}: {text!r}")
    return values


def rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rotation_distance(pin, target: np.ndarray, actual: np.ndarray) -> float:
    return float(np.linalg.norm(pin.log3(actual.T @ target)))


class RightArmSolver:
    def __init__(self, ik: G1PikaArmIK, left_q: np.ndarray):
        self.ik = ik
        self.pin = ik._pin
        self.model = ik.reduced_robot.model
        self.data = self.model.createData()
        self.left_q = left_q.copy()
        self.right_slice = slice(7, 14)
        self.lower = self.model.lowerPositionLimit[self.right_slice] + 1e-5
        self.upper = self.model.upperPositionLimit[self.right_slice] - 1e-5

    def fk(self, right_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.concatenate([self.left_q, right_q])
        self.pin.framesForwardKinematics(self.model, self.data, q)
        placement = self.data.oMf[self.ik.R_hand_id]
        return np.asarray(placement.translation).copy(), np.asarray(placement.rotation).copy()

    def solve(
        self,
        target_xyz: np.ndarray,
        target_rot: np.ndarray,
        seed: np.ndarray,
        *,
        position_scale: float,
        rotation_scale: float,
        regularization: float,
        max_nfev: int,
    ) -> tuple[np.ndarray, float, float]:
        seed = np.clip(seed, self.lower, self.upper)

        def residual(right_q: np.ndarray) -> np.ndarray:
            actual_xyz, actual_rot = self.fk(right_q)
            pose_error = np.concatenate(
                [
                    (actual_xyz - target_xyz) / position_scale,
                    self.pin.log3(actual_rot.T @ target_rot) / rotation_scale,
                ]
            )
            if regularization <= 0.0:
                return pose_error
            return np.concatenate([pose_error, regularization * (right_q - seed)])

        result = least_squares(
            residual,
            seed,
            bounds=(self.lower, self.upper),
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=max_nfev,
        )
        q = result.x
        actual_xyz, actual_rot = self.fk(q)
        return q, float(np.linalg.norm(actual_xyz - target_xyz)), rotation_distance(
            self.pin, target_rot, actual_rot
        )


def load_relative_trajectory(
    dataset: LeRobotDataset,
    start_index: int,
    steps: int,
    align_rotation: np.ndarray,
    motion_scale: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    relative_xyz = np.zeros(3, dtype=np.float64)
    relative_rot = np.eye(3, dtype=np.float64)
    trajectory: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(start_index, min(start_index + steps, len(dataset))):
        _, _, raw_xyz, raw_rot = action_pose(
            dataset[index]["action"], use_row_pose6d=False, single_arm="right"
        )
        delta_xyz = relative_rot @ align_rotation @ (raw_xyz * motion_scale)
        aligned_delta_rot = align_rotation @ raw_rot @ align_rotation.T
        relative_xyz = relative_xyz + delta_xyz
        relative_rot = relative_rot @ aligned_delta_rot
        trajectory.append((relative_xyz.copy(), relative_rot.copy()))
    return trajectory


def evaluate_candidate(
    solver: RightArmSolver,
    relative_trajectory: list[tuple[np.ndarray, np.ndarray]],
    initial_xyz: np.ndarray,
    initial_rot: np.ndarray,
    seed_q: np.ndarray,
    args: argparse.Namespace,
) -> CandidateResult:
    q_path: list[np.ndarray] = []
    position_errors: list[float] = []
    rotation_errors: list[float] = []
    q, initial_position_error, initial_rotation_error = solver.solve(
        initial_xyz,
        initial_rot,
        seed_q,
        position_scale=args.position_tolerance_m,
        rotation_scale=args.rotation_tolerance_rad,
        regularization=args.regularization,
        max_nfev=args.max_nfev,
    )
    initial_q = q.copy()
    failed_step = None

    if (
        initial_position_error > args.position_tolerance_m
        or initial_rotation_error > args.rotation_tolerance_rad
    ):
        return CandidateResult(
            xyz_offset=np.zeros(3),
            yaw_offset_rad=0.0,
            initial_q=initial_q,
            q_path=np.empty((0, 7)),
            position_errors=np.asarray([initial_position_error]),
            rotation_errors=np.asarray([initial_rotation_error]),
            failed_step=0,
            minimum_joint_margin_rad=float(
                np.min(np.minimum(initial_q - solver.lower, solver.upper - initial_q))
            ),
        )

    for step, (relative_xyz, relative_rot) in enumerate(relative_trajectory):
        target_xyz = initial_xyz + initial_rot @ relative_xyz
        target_rot = initial_rot @ relative_rot
        q, position_error, rotation_error = solver.solve(
            target_xyz,
            target_rot,
            q,
            position_scale=args.position_tolerance_m,
            rotation_scale=args.rotation_tolerance_rad,
            regularization=args.regularization,
            max_nfev=args.max_nfev,
        )
        q_path.append(q.copy())
        position_errors.append(position_error)
        rotation_errors.append(rotation_error)
        if position_error > args.position_tolerance_m or rotation_error > args.rotation_tolerance_rad:
            failed_step = step
            break

    q_path_array = np.asarray(q_path)
    evaluated_path = np.vstack([initial_q, q_path_array]) if q_path else initial_q[None, :]
    minimum_joint_margin = float(
        np.min(np.minimum(evaluated_path - solver.lower, solver.upper - evaluated_path))
    )
    return CandidateResult(
        xyz_offset=np.zeros(3),
        yaw_offset_rad=0.0,
        initial_q=initial_q,
        q_path=q_path_array,
        position_errors=np.asarray(position_errors),
        rotation_errors=np.asarray(rotation_errors),
        failed_step=failed_step,
        minimum_joint_margin_rad=minimum_joint_margin,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--repo-id", default="data")
    parser.add_argument("--episode", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--left-q", type=lambda value: parse_vec(value, 7), required=True)
    parser.add_argument("--reference-right-q", type=lambda value: parse_vec(value, 7), required=True)
    parser.add_argument("--motion-scale-xyz", type=lambda value: parse_vec(value, 3), default=np.ones(3))
    parser.add_argument("--x-offsets", type=lambda value: parse_vec(value, len(value.split())), default=np.array([-0.08, 0.0, 0.08]))
    parser.add_argument("--y-offsets", type=lambda value: parse_vec(value, len(value.split())), default=np.array([-0.08, 0.0, 0.08]))
    parser.add_argument("--z-offsets", type=lambda value: parse_vec(value, len(value.split())), default=np.array([-0.08, 0.0, 0.08]))
    parser.add_argument("--yaw-offsets-deg", type=lambda value: parse_vec(value, len(value.split())), default=np.array([-30.0, 0.0, 30.0]))
    parser.add_argument(
        "--tool-roll-offset-deg",
        type=float,
        default=0.0,
        help="Rotate the initial TCP about its local +Z forward axis before searching.",
    )
    parser.add_argument("--position-tolerance-m", type=float, default=0.005)
    parser.add_argument("--rotation-tolerance-rad", type=float, default=0.05)
    parser.add_argument("--regularization", type=float, default=0.002)
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ik = G1PikaArmIK()
    solver = RightArmSolver(ik, args.left_q)
    reference_xyz, reference_rot = solver.fk(args.reference_right_q)
    reference_rot = reference_rot @ rot_z(np.deg2rad(args.tool_roll_offset_deg))
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        episodes=[args.episode],
        return_uint8=True,
        video_backend=args.video_backend,
    )
    trajectory = load_relative_trajectory(
        dataset,
        args.start_index,
        min(args.steps, len(dataset) - args.start_index),
        ik.tracker_to_tcp_rotation("right"),
        args.motion_scale_xyz,
    )

    candidates: list[CandidateResult] = []
    offsets = [
        np.array([x, y, z], dtype=np.float64)
        for x in args.x_offsets
        for y in args.y_offsets
        for z in args.z_offsets
    ]
    yaw_offsets = np.deg2rad(args.yaw_offsets_deg)
    total = len(offsets) * len(yaw_offsets)
    for candidate_index, offset in enumerate(offsets):
        for yaw_offset in yaw_offsets:
            initial_xyz = reference_xyz + offset
            initial_rot = rot_z(float(yaw_offset)) @ reference_rot
            result = evaluate_candidate(
                solver,
                trajectory,
                initial_xyz,
                initial_rot,
                args.reference_right_q,
                args,
            )
            result.xyz_offset = offset.copy()
            result.yaw_offset_rad = float(yaw_offset)
            candidates.append(result)
            status = "PASS" if result.passed else f"FAIL@{result.failed_step}"
            print(
                f"[{len(candidates):03d}/{total:03d}] {status} offset={offset} "
                f"yaw={np.rad2deg(yaw_offset):.1f}deg "
                f"max_pos={np.max(result.position_errors, initial=0.0):.4f}m "
                f"max_rot={np.rad2deg(np.max(result.rotation_errors, initial=0.0)):.2f}deg "
                f"joint_margin={np.rad2deg(result.minimum_joint_margin_rad):.1f}deg"
            )

    candidates.sort(key=CandidateResult.score)
    rows = []
    for rank, candidate in enumerate(candidates, start=1):
        rows.append(
            {
                "rank": rank,
                "passed": int(candidate.passed),
                "valid_steps": candidate.valid_steps,
                "failed_step": "" if candidate.failed_step is None else candidate.failed_step,
                "offset_x": candidate.xyz_offset[0],
                "offset_y": candidate.xyz_offset[1],
                "offset_z": candidate.xyz_offset[2],
                "yaw_offset_deg": np.rad2deg(candidate.yaw_offset_rad),
                "max_position_error_m": np.max(candidate.position_errors, initial=0.0),
                "max_rotation_error_rad": np.max(candidate.rotation_errors, initial=0.0),
                "minimum_joint_margin_rad": candidate.minimum_joint_margin_rad,
                "maximum_joint_step_rad": candidate.maximum_joint_step_rad,
                "initial_right_q": " ".join(f"{value:.8f}" for value in candidate.initial_q),
            }
        )
    csv_path = args.output_dir / "start_pose_search.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best = candidates[0]
    best_payload = {
        "passed": best.passed,
        "steps": len(trajectory),
        "valid_steps": best.valid_steps,
        "failed_step": best.failed_step,
        "reference_xyz": reference_xyz.tolist(),
        "xyz_offset": best.xyz_offset.tolist(),
        "start_xyz": (reference_xyz + best.xyz_offset).tolist(),
        "yaw_offset_deg": float(np.rad2deg(best.yaw_offset_rad)),
        "tool_roll_offset_deg": args.tool_roll_offset_deg,
        "initial_right_q": best.initial_q.tolist(),
        "max_position_error_m": float(np.max(best.position_errors, initial=0.0)),
        "max_rotation_error_rad": float(np.max(best.rotation_errors, initial=0.0)),
        "minimum_joint_margin_rad": best.minimum_joint_margin_rad,
        "maximum_joint_step_rad": best.maximum_joint_step_rad,
    }
    json_path = args.output_dir / "best_start_pose.json"
    json_path.write_text(json.dumps(best_payload, indent=2) + "\n")
    if best.q_path.size:
        np.savetxt(args.output_dir / "best_joint_path.csv", best.q_path, delimiter=",")
    print(json.dumps(best_payload, indent=2))
    print(f"results: {csv_path}")


if __name__ == "__main__":
    main()
