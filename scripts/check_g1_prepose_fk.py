#!/usr/bin/env python3
"""Print current G1 arm FK for comparing real prepose with retarget work-start."""

from __future__ import annotations

import argparse

import numpy as np

from lerobot.robots.unitree_g1_pika.g1_pika_kinematics import G1PikaArmIK
from lerobot.robots.unitree_g1_pika.config_unitree_g1_pika import UnitreeG1PikaConfig
from lerobot.robots.unitree_g1_pika.unitree_g1_pika import UnitreeG1Pika


ARM_PAIRS = [
    ("left_shoulder_pitch", "kLeftShoulderPitch.q"),
    ("left_shoulder_roll", "kLeftShoulderRoll.q"),
    ("left_shoulder_yaw", "kLeftShoulderYaw.q"),
    ("left_elbow", "kLeftElbow.q"),
    ("left_wrist_roll", "kLeftWristRoll.q"),
    ("left_wrist_pitch", "kLeftWristPitch.q"),
    ("left_wrist_yaw", "kLeftWristYaw.q"),
    ("right_shoulder_pitch", "kRightShoulderPitch.q"),
    ("right_shoulder_roll", "kRightShoulderRoll.q"),
    ("right_shoulder_yaw", "kRightShoulderYaw.q"),
    ("right_elbow", "kRightElbow.q"),
    ("right_wrist_roll", "kRightWristRoll.q"),
    ("right_wrist_pitch", "kRightWristPitch.q"),
    ("right_wrist_yaw", "kRightWristYaw.q"),
]


def parse_vec3(text: str) -> np.ndarray:
    values = [float(x) for x in text.replace(",", " ").split()]
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f"expected 3 floats, got {text!r}")
    return np.asarray(values, dtype=np.float64)


def transform_matrix(xyz: np.ndarray, rot: np.ndarray) -> np.ndarray:
    tf = np.eye(4)
    tf[:3, :3] = rot
    tf[:3, 3] = xyz
    return tf


def invert_transform(tf: np.ndarray) -> np.ndarray:
    inv = np.eye(4)
    inv[:3, :3] = tf[:3, :3].T
    inv[:3, 3] = -inv[:3, :3] @ tf[:3, 3]
    return inv


def neutral_tcp_tf(ik: G1PikaArmIK, frame_id: int) -> np.ndarray:
    pin = ik._pin
    data = ik.reduced_robot.model.createData()
    q_neutral = np.zeros(ik.reduced_robot.model.nq)
    pin.forwardKinematics(ik.reduced_robot.model, data, q_neutral)
    pin.updateFramePlacements(ik.reduced_robot.model, data)
    return data.oMf[frame_id].homogeneous


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-ip", default="192.168.123.164")
    parser.add_argument("--right-work-start", type=parse_vec3, default=parse_vec3("0.15 -0.20 0.90"))
    parser.add_argument("--left-work-start", type=parse_vec3, default=parse_vec3("0.35 0.20 0.85"))
    parser.add_argument("--mujoco-neutral-left-pos", type=parse_vec3, default=parse_vec3("0.3798 0.1486 0.8882"))
    parser.add_argument("--mujoco-neutral-right-pos", type=parse_vec3, default=parse_vec3("0.3798 -0.1486 0.8882"))
    args = parser.parse_args()

    robot = UnitreeG1Pika(
        UnitreeG1PikaConfig(
            is_simulation=False,
            robot_ip=args.robot_ip,
            gripper_mode="remote",
            use_left_gripper=False,
            use_right_gripper=True,
        )
    )
    robot.connect(calibrate=False)
    try:
        obs = robot.get_observation()
    finally:
        robot.disconnect()

    ik = G1PikaArmIK()
    q_g1 = np.array([obs[key] for _, key in ARM_PAIRS], dtype=np.float64)
    q_pin = q_g1[ik._arm_reorder_g1_to_pin]

    pin = ik._pin
    data = ik.reduced_robot.model.createData()
    pin.forwardKinematics(ik.reduced_robot.model, data, q_pin)
    pin.updateFramePlacements(ik.reduced_robot.model, data)
    left = data.oMf[ik.L_hand_id]
    right = data.oMf[ik.R_hand_id]

    unitree_neutral_left_tf = neutral_tcp_tf(ik, ik.L_hand_id)
    unitree_neutral_right_tf = neutral_tcp_tf(ik, ik.R_hand_id)
    mujoco_neutral_left_tf = transform_matrix(args.mujoco_neutral_left_pos, np.eye(3))
    mujoco_neutral_right_tf = transform_matrix(args.mujoco_neutral_right_pos, np.eye(3))
    unitree_to_mujoco_left_tf = mujoco_neutral_left_tf @ invert_transform(unitree_neutral_left_tf)
    unitree_to_mujoco_right_tf = mujoco_neutral_right_tf @ invert_transform(unitree_neutral_right_tf)
    left_mujoco_tf = unitree_to_mujoco_left_tf @ left.homogeneous
    right_mujoco_tf = unitree_to_mujoco_right_tf @ right.homogeneous

    print("OK: current G1 prepose FK")
    print("left arm q_g1 :", np.round(q_g1[:7], 4))
    print("right arm q_g1:", np.round(q_g1[7:], 4))
    print("left PIKA TCP FK pos unitree :", np.round(left.translation, 4))
    print("right PIKA TCP FK pos unitree:", np.round(right.translation, 4))
    print("left PIKA TCP FK pos mujoco-equivalent :", np.round(left_mujoco_tf[:3, 3], 4))
    print("right PIKA TCP FK pos mujoco-equivalent:", np.round(right_mujoco_tf[:3, 3], 4))
    print("left FK - left-work-start  :", np.round(left_mujoco_tf[:3, 3] - args.left_work_start, 4))
    print("right FK - right-work-start:", np.round(right_mujoco_tf[:3, 3] - args.right_work_start, 4))
    print("left PIKA TCP FK rot unitree:")
    print(np.round(left.rotation, 4))
    print("right PIKA TCP FK rot unitree:")
    print(np.round(right.rotation, 4))
    print("script left-work-start mujoco :", np.round(args.left_work_start, 4))
    print("script right-work-start mujoco:", np.round(args.right_work_start, 4))


if __name__ == "__main__":
    main()
