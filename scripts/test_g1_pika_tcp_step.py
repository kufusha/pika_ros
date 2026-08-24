#!/usr/bin/env python3
"""Test one small Cartesian TCP step on the real G1 right arm."""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.g1_utils import G1_29_JointArmIndex
from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1
from lerobot.robots.unitree_g1_pika.g1_pika_kinematics import G1PikaArmIK


RIGHT_ARM = tuple(joint for joint in G1_29_JointArmIndex if joint.name.startswith("kRight"))


def parse_vec3(value: str) -> np.ndarray:
    result = np.fromstring(value.replace(",", " "), sep=" ", dtype=np.float64)
    if result.shape != (3,):
        raise argparse.ArgumentTypeError(f"expected 3 values, got {result.shape}: {value!r}")
    return result


def arm_q_from_observation(observation: dict[str, object]) -> np.ndarray:
    return np.asarray(
        [float(observation[f"{joint.name}.q"]) for joint in G1_29_JointArmIndex],
        dtype=np.float64,
    )


def frame_transform(ik: G1PikaArmIK, q_pin: np.ndarray) -> np.ndarray:
    data = ik.reduced_robot.model.createData()
    ik._pin.framesForwardKinematics(ik.reduced_robot.model, data, q_pin)
    placement = data.oMf[ik.R_hand_id]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(placement.rotation)
    transform[:3, 3] = np.asarray(placement.translation)
    return transform


def rotation_error_rad(ik: G1PikaArmIK, target: np.ndarray, actual: np.ndarray) -> float:
    return float(np.linalg.norm(ik._pin.log3(actual[:3, :3].T @ target[:3, :3])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="192.168.123.164")
    parser.add_argument("--delta", type=parse_vec3, default=parse_vec3("0 0 0.005"))
    parser.add_argument(
        "--frame",
        choices=["base", "tcp"],
        default="base",
        help="Coordinate frame in which --delta is expressed.",
    )
    parser.add_argument("--real", action="store_true", help="Actually send joint commands.")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--settle-sec", type=float, default=3.0)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.002)
    parser.add_argument("--max-joint-move-rad", type=float, default=0.10)
    parser.add_argument("--position-tolerance-m", type=float, default=0.003)
    parser.add_argument("--rotation-tolerance-rad", type=float, default=0.03)
    parser.add_argument("--tracking-error-limit-rad", type=float, default=0.12)
    parser.add_argument(
        "--gravity-compensation",
        action="store_true",
        help="Apply right-arm gravity feedforward from the G1+PIKA model.",
    )
    parser.add_argument(
        "--closed-loop-hold",
        action="store_true",
        help="Apply bounded joint-error correction while holding the TCP target.",
    )
    parser.add_argument("--outer-loop-gain", type=float, default=0.05)
    parser.add_argument("--outer-correction-step-rad", type=float, default=0.0005)
    parser.add_argument("--max-command-correction-rad", type=float, default=0.04)
    args = parser.parse_args()

    if args.fps <= 0.0 or args.duration <= 0.0 or args.settle_sec <= 0.0:
        raise ValueError("--fps, --duration, and --settle-sec must be positive")
    if np.linalg.norm(args.delta) > 0.01:
        raise ValueError("TCP step is limited to 10 mm for this test")

    robot = UnitreeG1(UnitreeG1Config(is_simulation=False, robot_ip=args.robot_ip))
    robot.connect(calibrate=False)
    last_action = None
    commanded = False
    try:
        observation = robot.get_observation()
        hold_action = {key: float(value) for key, value in observation.items() if key.endswith(".q")}
        q_g1 = arm_q_from_observation(observation)
        ik = G1PikaArmIK()
        q_pin = q_g1[ik._arm_reorder_g1_to_pin]
        start_tcp = frame_transform(ik, q_pin)
        target_tcp = start_tcp.copy()
        if args.frame == "base":
            target_tcp[:3, 3] += args.delta
        else:
            target_tcp[:3, 3] += start_tcp[:3, :3] @ args.delta

        solved_pin, solver_position_error, solver_rotation_error = ik.solve_right_ik(
            target_tcp,
            q_pin,
            position_scale=args.position_tolerance_m,
            rotation_scale=args.rotation_tolerance_rad,
            regularization=0.002,
        )
        solved_g1 = solved_pin[ik._arm_reorder_pin_to_g1]
        right_start = q_g1[7:]
        right_target = solved_g1[7:]
        joint_delta = right_target - right_start
        max_joint_move = float(np.max(np.abs(joint_delta)))

        print("\n=== G1 PIKA TCP STEP CHECK ===")
        print("mode                    :", "REAL" if args.real else "DRY RUN")
        print("gravity compensation    :", args.gravity_compensation)
        print("closed-loop hold        :", args.closed_loop_hold)
        print("delta frame             :", args.frame)
        print("requested delta [m]     :", np.round(args.delta, 6))
        print("start TCP xyz [m]       :", np.round(start_tcp[:3, 3], 6))
        print("target TCP xyz [m]      :", np.round(target_tcp[:3, 3], 6))
        print("start right q [rad]     :", np.round(right_start, 6))
        print("target right q [rad]    :", np.round(right_target, 6))
        print("joint delta [rad]       :", np.round(joint_delta, 6))
        print("max joint move [rad]    :", f"{max_joint_move:.6f}")
        print("offline TCP pos err [m] :", f"{solver_position_error:.9f}")
        print("offline TCP rot err [rad]:", f"{solver_rotation_error:.9f}")

        if solver_position_error > args.position_tolerance_m:
            raise RuntimeError("IK position error exceeds tolerance")
        if solver_rotation_error > args.rotation_tolerance_rad:
            raise RuntimeError("IK rotation error exceeds tolerance")
        if max_joint_move > args.max_joint_move_rad:
            raise RuntimeError(
                f"IK joint move {max_joint_move:.4f}rad exceeds limit {args.max_joint_move_rad:.4f}rad"
            )
        if not args.real:
            print("DRY RUN complete: no LowCmd was sent")
            return

        input(
            "Press Enter to command this TCP step. Keep the arm clear; Ctrl-C cancels and makes it passive: "
        )
        period = 1.0 / args.fps
        right_kp = np.zeros(29, dtype=np.float32)
        right_kd = np.zeros(29, dtype=np.float32)
        for joint in RIGHT_ARM:
            right_kp[joint.value] = robot.kp[joint.value]
            right_kd[joint.value] = robot.kd[joint.value]

        def publish_right_arm(action: dict[str, float]) -> None:
            tau = np.zeros(29, dtype=np.float32)
            if args.gravity_compensation:
                arm_q = np.asarray(
                    [float(action[f"{joint.name}.q"]) for joint in G1_29_JointArmIndex],
                    dtype=np.float64,
                )
                arm_tau = ik.solve_tau(arm_q)
                for local_index, joint in enumerate(G1_29_JointArmIndex):
                    if joint.name.startswith("kRight"):
                        tau[joint.value] = arm_tau[local_index]
            robot.publish_lowcmd(action, kp=right_kp, kd=right_kd, tau=tau)

        print(f"ENGAGE: holding the current command for {args.settle_sec:.1f}s before recapturing TCP")
        for _ in range(max(1, int(args.settle_sec * args.fps))):
            publish_right_arm(hold_action)
            last_action = hold_action
            commanded = True
            time.sleep(period)

        settled_observation = robot.get_observation()
        settled_g1 = arm_q_from_observation(settled_observation)
        settled_right = settled_g1[7:]
        settled_pin = settled_g1[ik._arm_reorder_g1_to_pin]
        settled_tcp = frame_transform(ik, settled_pin)
        engage_q_shift = settled_right - right_start
        engage_tcp_shift = settled_tcp[:3, 3] - start_tcp[:3, 3]

        target_tcp = settled_tcp.copy()
        if args.frame == "base":
            target_tcp[:3, 3] += args.delta
        else:
            target_tcp[:3, 3] += settled_tcp[:3, :3] @ args.delta
        solved_pin, solver_position_error, solver_rotation_error = ik.solve_right_ik(
            target_tcp,
            settled_pin,
            position_scale=args.position_tolerance_m,
            rotation_scale=args.rotation_tolerance_rad,
            regularization=0.002,
        )
        solved_g1 = solved_pin[ik._arm_reorder_pin_to_g1]
        expected_right_target = solved_g1[7:]
        joint_delta = expected_right_target - settled_right
        max_joint_move = float(np.max(np.abs(joint_delta)))
        if solver_position_error > args.position_tolerance_m:
            raise RuntimeError("settled-baseline IK position error exceeds tolerance")
        if solver_rotation_error > args.rotation_tolerance_rad:
            raise RuntimeError("settled-baseline IK rotation error exceeds tolerance")
        if max_joint_move > args.max_joint_move_rad:
            raise RuntimeError(
                f"settled-baseline IK joint move {max_joint_move:.4f}rad exceeds "
                f"limit {args.max_joint_move_rad:.4f}rad"
            )

        # Preserve the measured closed-loop bias by applying only the IK delta
        # to the command that established the settled baseline.
        right_command_start = right_start.copy()
        right_target = right_command_start + joint_delta
        print("\n=== SETTLED REAL BASELINE ===")
        print("engage q shift [rad]     :", np.round(engage_q_shift, 6))
        print("engage TCP shift [m]     :", np.round(engage_tcp_shift, 6))
        print("settled TCP xyz [m]      :", np.round(settled_tcp[:3, 3], 6))
        print("target TCP xyz [m]       :", np.round(target_tcp[:3, 3], 6))
        print("IK joint delta [rad]     :", np.round(joint_delta, 6))
        print("command target q [rad]   :", np.round(right_target, 6))
        print("offline TCP pos err [m]  :", f"{solver_position_error:.9f}")
        print("offline TCP rot err [rad]:", f"{solver_rotation_error:.9f}")

        minimum_steps = math.ceil(max_joint_move / args.max_joint_step_rad)
        steps = max(1, int(args.duration * args.fps), minimum_steps)
        target_action = dict(hold_action)
        for joint, value in zip(RIGHT_ARM, right_target, strict=True):
            target_action[f"{joint.name}.q"] = float(value)

        for step in range(1, steps + 1):
            alpha = step / steps
            action = dict(hold_action)
            current_command = right_command_start * (1.0 - alpha) + right_target * alpha
            for joint, value in zip(RIGHT_ARM, current_command, strict=True):
                action[f"{joint.name}.q"] = float(value)
            publish_right_arm(action)
            last_action = action
            commanded = True

            if step % max(1, int(args.fps / 5.0)) == 0 or step == steps:
                observed_q = arm_q_from_observation(robot.get_observation())[7:]
                expected_observed_q = settled_right * (1.0 - alpha) + expected_right_target * alpha
                tracking_error = float(np.max(np.abs(observed_q - expected_observed_q)))
                print(
                    f"MOVE {step:04d}/{steps} alpha={alpha:.3f} "
                    f"observed_q_error={tracking_error:.4f}rad"
                )
                if tracking_error > args.tracking_error_limit_rad:
                    raise RuntimeError(
                        f"joint tracking error {tracking_error:.4f}rad exceeds "
                        f"{args.tracking_error_limit_rad:.4f}rad"
                    )
            time.sleep(period)

        print("TCP step complete. Holding the final command; press Ctrl-C to make the robot passive.")
        next_report = 0.0
        command_correction = np.zeros(7, dtype=np.float64)
        while True:
            corrected_action = dict(target_action)
            corrected_command = right_target + command_correction
            for joint, value in zip(RIGHT_ARM, corrected_command, strict=True):
                corrected_action[f"{joint.name}.q"] = float(value)
            publish_right_arm(corrected_action)
            last_action = corrected_action

            observed = robot.get_observation()
            observed_g1 = arm_q_from_observation(observed)
            right_q_error = expected_right_target - observed_g1[7:]
            if args.closed_loop_hold:
                correction_step = np.clip(
                    args.outer_loop_gain * right_q_error,
                    -args.outer_correction_step_rad,
                    args.outer_correction_step_rad,
                )
                command_correction = np.clip(
                    command_correction + correction_step,
                    -args.max_command_correction_rad,
                    args.max_command_correction_rad,
                )
                total_command_move = float(
                    np.max(np.abs(right_target + command_correction - right_command_start))
                )
                if total_command_move > args.max_joint_move_rad:
                    raise RuntimeError(
                        f"corrected command move {total_command_move:.4f}rad exceeds "
                        f"limit {args.max_joint_move_rad:.4f}rad"
                    )

            now = time.monotonic()
            if now >= next_report:
                observed_pin = observed_g1[ik._arm_reorder_g1_to_pin]
                observed_tcp = frame_transform(ik, observed_pin)
                position_error = float(np.linalg.norm(observed_tcp[:3, 3] - target_tcp[:3, 3]))
                orientation_error = rotation_error_rad(ik, target_tcp, observed_tcp)
                q_error = float(np.max(np.abs(observed_g1[7:] - expected_right_target)))
                print(
                    f"HOLD tcp_position_error={position_error:.5f}m "
                    f"tcp_rotation_error={orientation_error:.5f}rad "
                    f"q_error={q_error:.4f}rad"
                )
                print("     observed_dxyz=", np.round(observed_tcp[:3, 3] - settled_tcp[:3, 3], 6))
                print("     right_q_error=", np.round(observed_g1[7:] - expected_right_target, 5))
                print("     command_correction=", np.round(command_correction, 5))
                next_report = now + 1.0
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nOperator stop: disconnecting and making G1 passive")
    finally:
        if commanded and last_action is not None:
            publish_right_arm(last_action)
            time.sleep(0.1)
        if not commanded:
            # UnitreeG1.disconnect() normally publishes a zero-gain LowCmd.
            # Suppress that command when this process only observed state.
            robot.config.is_simulation = True
        robot.disconnect()


if __name__ == "__main__":
    main()
