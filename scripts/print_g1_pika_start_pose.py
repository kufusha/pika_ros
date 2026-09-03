#!/usr/bin/env python3
"""Print and validate the canonical G1+PIKA URDF-zero start pose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lerobot.robots.unitree_g1_pika.g1_pika_kinematics import G1PikaArmIK


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config/g1_pika_retarget_start_pose.json"


def rotation_error_rad(actual: np.ndarray, expected: np.ndarray) -> float:
    cosine = np.clip((np.trace(actual.T @ expected) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    left_q = np.asarray(config["left_arm_q_g1_rad"], dtype=np.float64)
    right_q = np.asarray(config["right_arm_q_g1_rad"], dtype=np.float64)
    expected_position = np.asarray(
        config["right_pika_tcp"]["position_pelvis_m"], dtype=np.float64
    )
    expected_rotation = np.asarray(
        config["right_pika_tcp"]["rotation_matrix"], dtype=np.float64
    )

    ik = G1PikaArmIK()
    q_pin = np.concatenate([left_q, right_q])[ik._arm_reorder_g1_to_pin]
    data = ik.reduced_robot.model.createData()
    ik._pin.framesForwardKinematics(ik.reduced_robot.model, data, q_pin)
    ik._pin.updateFramePlacements(ik.reduced_robot.model, data)
    actual = data.oMf[ik.R_hand_id].homogeneous
    position_error = float(np.linalg.norm(actual[:3, 3] - expected_position))
    orientation_error = rotation_error_rad(actual[:3, :3], expected_rotation)

    position_tolerance = float(config["fk_validation"]["position_tolerance_m"])
    rotation_tolerance = float(config["fk_validation"]["rotation_tolerance_rad"])
    if position_error > position_tolerance or orientation_error > rotation_tolerance:
        raise RuntimeError(
            "canonical start pose does not match the current G1+PIKA URDF: "
            f"position_error={position_error:.9f}m, rotation_error={orientation_error:.9f}rad"
        )

    world_position = expected_position.copy()
    world_position[2] += float(config["mujoco_world_pelvis_z_m"])
    print(f"config: {args.config.resolve()}")
    print(f"name: {config['name']}")
    print("left_arm_q_g1_rad:", np.array2string(left_q, precision=8))
    print("right_arm_q_g1_rad:", np.array2string(right_q, precision=8))
    print("right_tcp_position_pelvis_m:", np.array2string(actual[:3, 3], precision=9))
    print("right_tcp_position_mujoco_world_m:", np.array2string(world_position, precision=9))
    print(
        "right_tcp_quaternion_xyzw:",
        np.array2string(np.asarray(config["right_pika_tcp"]["quaternion_xyzw"]), precision=9),
    )
    print("right_tcp_rotation_matrix:\n", np.array2string(actual[:3, :3], precision=9))
    print(
        "right_tcp_rpy_xyz_rad_diagnostic_only:",
        np.array2string(
            np.asarray(config["right_pika_tcp"]["rpy_xyz_rad_diagnostic_only"]), precision=9
        ),
    )
    print(f"FK validation: OK (position={position_error:.3e}m rotation={orientation_error:.3e}rad)")


if __name__ == "__main__":
    main()
