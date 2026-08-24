#!/usr/bin/env python3
"""Visualize LeRobot PIKA dataset hand positions and orientations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


PIKA_MESH_DIR = Path("src/pika_ros/src/PikaAnyArm/piper/piper_ros/piper_description/meshes")
DEFAULT_PIKA_TCP_OFFSET = np.array([0.18, 0.0, 0.0], dtype=np.float64)


def scene_xml(
    mesh_dir: Path,
    pika_pos: str,
    pika_euler: str,
    link7_pos: str,
    link8_pos: str,
    link7_euler: str,
    link8_euler: str,
) -> str:
    mesh_dir = mesh_dir.resolve()
    return f"""
<mujoco model="pika_dataset_pose_viewer">
  <compiler angle="radian"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="2048"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.86 0.86 0.86" rgb2="0.68 0.70 0.70"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.04"/>
    <material name="x_axis" rgba="1.00 0.08 0.05 1.00"/>
    <material name="y_axis" rgba="0.05 0.75 0.10 1.00"/>
    <material name="z_axis" rgba="0.05 0.25 1.00 1.00"/>
    <material name="left_center" rgba="0.00 0.85 1.00 1.00"/>
    <material name="right_center" rgba="1.00 0.55 0.00 1.00"/>
    <material name="left_base" rgba="0.00 0.40 0.85 1.00"/>
    <material name="right_base" rgba="0.85 0.28 0.00 1.00"/>
    <mesh name="pika_gripper_base_mesh" file="{mesh_dir / "gripper_base.STL"}"/>
    <mesh name="pika_link7_mesh" file="{mesh_dir / "link7.STL"}"/>
    <mesh name="pika_link8_mesh" file="{mesh_dir / "link8.STL"}"/>
  </asset>
  <worldbody>
    <light name="key" pos="0 -2.5 2.2" dir="0 0.8 -1" diffuse="0.9 0.9 0.9"/>
    <light name="fill" pos="-1.5 1.2 1.4" dir="0.7 -0.5 -1" diffuse="0.35 0.35 0.35"/>
    <geom name="floor" type="plane" pos="0 0 0" size="1.0 1.0 0.02" material="grid"/>
    <geom name="box" type="box" pos="0.30 0.05 0.08" size="0.22 0.16 0.08" rgba="0.55 0.36 0.20 0.45"/>
    <geom name="bottle" type="cylinder" pos="0.30 0.05 0.24" size="0.04 0.15" rgba="0.05 0.35 0.85 0.55"/>

    <body name="left_pose" mocap="true" pos="-0.15 0.10 0.25">
      <geom type="sphere" size="0.018" material="left_center"/>
      <geom type="capsule" fromto="0 0 0 0.08 0 0" size="0.006" material="x_axis"/>
      <geom type="capsule" fromto="0 0 0 0 0.08 0" size="0.006" material="y_axis"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.08" size="0.006" material="z_axis"/>
      <body name="left_pika_gripper" pos="{pika_pos}" euler="{pika_euler}">
        <geom type="mesh" mesh="pika_gripper_base_mesh" rgba="0.76 0.79 0.86 0.90" contype="0" conaffinity="0"/>
        <body name="left_pika_link7" pos="{link7_pos}" euler="{link7_euler}">
          <joint name="left_pika_joint7" type="slide" axis="0 0 1" range="0 0.035" limited="true"/>
          <geom type="mesh" mesh="pika_link7_mesh" rgba="0.76 0.79 0.86 0.90" contype="0" conaffinity="0"/>
        </body>
        <body name="left_pika_link8" pos="{link8_pos}" euler="{link8_euler}">
          <joint name="left_pika_joint8" type="slide" axis="0 0 1" range="0 0.035" limited="true"/>
          <geom type="mesh" mesh="pika_link8_mesh" rgba="0.76 0.79 0.86 0.90" contype="0" conaffinity="0"/>
        </body>
      </body>
    </body>
    <body name="right_pose" mocap="true" pos="0.15 -0.10 0.25">
      <geom type="sphere" size="0.018" material="right_center"/>
      <geom type="capsule" fromto="0 0 0 0.08 0 0" size="0.006" material="x_axis"/>
      <geom type="capsule" fromto="0 0 0 0 0.08 0" size="0.006" material="y_axis"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.08" size="0.006" material="z_axis"/>
      <body name="right_pika_gripper" pos="{pika_pos}" euler="{pika_euler}">
        <geom type="mesh" mesh="pika_gripper_base_mesh" rgba="0.76 0.79 0.86 0.90" contype="0" conaffinity="0"/>
        <body name="right_pika_link7" pos="{link7_pos}" euler="{link7_euler}">
          <joint name="right_pika_joint7" type="slide" axis="0 0 1" range="0 0.035" limited="true"/>
          <geom type="mesh" mesh="pika_link7_mesh" rgba="0.76 0.79 0.86 0.90" contype="0" conaffinity="0"/>
        </body>
        <body name="right_pika_link8" pos="{link8_pos}" euler="{link8_euler}">
          <joint name="right_pika_joint8" type="slide" axis="0 0 1" range="0 0.035" limited="true"/>
          <geom type="mesh" mesh="pika_link8_mesh" rgba="0.76 0.79 0.86 0.90" contype="0" conaffinity="0"/>
        </body>
      </body>
    </body>
    <body name="left_base_link" mocap="true" pos="-0.25 0.10 0.02">
      <geom type="sphere" size="0.024" material="left_base"/>
      <geom type="capsule" fromto="0 0 0 0.12 0 0" size="0.008" material="x_axis"/>
      <geom type="capsule" fromto="0 0 0 0 0.12 0" size="0.008" material="y_axis"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.12" size="0.008" material="z_axis"/>
    </body>
    <body name="right_base_link" mocap="true" pos="0.25 -0.10 0.02">
      <geom type="sphere" size="0.024" material="right_base"/>
      <geom type="capsule" fromto="0 0 0 0.12 0 0" size="0.008" material="x_axis"/>
      <geom type="capsule" fromto="0 0 0 0 0.12 0" size="0.008" material="y_axis"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.12" size="0.008" material="z_axis"/>
    </body>
  </worldbody>
</mujoco>
"""


def parse_vec3(value: str) -> np.ndarray:
    parts = [float(x) for x in value.replace(",", " ").split()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected xyz, e.g. '0 0 0.35'")
    return np.array(parts, dtype=np.float64)


def parse_pose6(value: str) -> np.ndarray:
    parts = [float(x) for x in value.replace(",", " ").split()]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("expected xyzrpy, e.g. '0 0 0 0 0 0'")
    return np.array(parts, dtype=np.float64)


def vec_to_mjcf(value: np.ndarray) -> str:
    return " ".join(f"{x:.8g}" for x in value)


def euler_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def apply_tracker_to_tcp(
    tracker_xyz: np.ndarray,
    tracker_rot: np.ndarray,
    tcp_pos_in_tracker: np.ndarray,
    tcp_rot_in_tracker: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tcp_xyz = tracker_xyz + tracker_rot @ tcp_pos_in_tracker
    tcp_rot = tracker_rot @ tcp_rot_in_tracker
    return tcp_xyz, tcp_rot


def rotation6d_to_matrix(rot6d: np.ndarray, use_row_pose6d: bool = False) -> np.ndarray:
    values = rot6d.astype(np.float64)
    if use_row_pose6d:
        raw = values.reshape(3, 2)
        col0 = raw[:, 0]
        col1 = raw[:, 1]
    else:
        col0 = values[:3]
        col1 = values[3:6]
    n0 = np.linalg.norm(col0)
    col0 = np.array([1.0, 0.0, 0.0]) if n0 < 1e-8 else col0 / n0
    col1 = col1 - col0 * float(col0 @ col1)
    n1 = np.linalg.norm(col1)
    if n1 < 1e-8:
        col1 = np.array([0.0, 1.0, 0.0])
        col1 = col1 - col0 * float(col0 @ col1)
        col1 = col1 / np.linalg.norm(col1)
    else:
        col1 = col1 / n1
    col2 = np.cross(col0, col1)
    return np.stack([col0, col1, col2], axis=1)


def rotmat_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(rot)))
        if idx == 0:
            s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            qw = (rot[2, 1] - rot[1, 2]) / s
            qx = 0.25 * s
            qy = (rot[0, 1] + rot[1, 0]) / s
            qz = (rot[0, 2] + rot[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            qw = (rot[0, 2] - rot[2, 0]) / s
            qx = (rot[0, 1] + rot[1, 0]) / s
            qy = 0.25 * s
            qz = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            qw = (rot[1, 0] - rot[0, 1]) / s
            qx = (rot[0, 2] + rot[2, 0]) / s
            qy = (rot[1, 2] + rot[2, 1]) / s
            qz = 0.25 * s
    quat = np.array([qw, qx, qy, qz], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def pose_from_vector(
    values: np.ndarray,
    use_row_pose6d: bool,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float]:
    left_xyz = values[0:3].copy()
    left_rot = rotation6d_to_matrix(values[3:9], use_row_pose6d=use_row_pose6d)
    left_gripper = float(values[9])
    right_xyz = values[10:13].copy()
    right_rot = rotation6d_to_matrix(values[13:19], use_row_pose6d=use_row_pose6d)
    right_gripper = float(values[19])
    return left_xyz, left_rot, left_gripper, right_xyz, right_rot, right_gripper


def tensor_values(item: dict, field: str) -> np.ndarray:
    value = item[field]
    if not torch.is_tensor(value):
        raise TypeError(f"{field} is not a tensor")
    values = value.detach().cpu().numpy().astype(np.float64)
    if values.shape[0] < 20:
        raise ValueError(f"{field} has shape {values.shape}, expected at least 20 values")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--repo-id", default="data")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--field", choices=["action", "observation.state"], default="action")
    parser.add_argument("--use-row-pose6d", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("videos/dataset_pose_viewer"))
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--center", type=parse_vec3, default=parse_vec3("0,0,0.35"))
    parser.add_argument("--left-tracker-to-tcp-pos", type=parse_vec3, default=parse_vec3("0 0 0"))
    parser.add_argument("--right-tracker-to-tcp-pos", type=parse_vec3, default=parse_vec3("0 0 0"))
    parser.add_argument("--left-tracker-to-tcp-euler", type=parse_vec3, default=parse_vec3("0 0 0"))
    parser.add_argument("--right-tracker-to-tcp-euler", type=parse_vec3, default=parse_vec3("0 0 0"))
    parser.add_argument("--left-base-link-in-world", type=parse_pose6, default=parse_pose6("0 0 0 0 0 0"))
    parser.add_argument("--right-base-link-in-world", type=parse_pose6, default=parse_pose6("0 0 0 0 0 0"))
    parser.add_argument("--mesh-dir", type=Path, default=PIKA_MESH_DIR)
    parser.add_argument("--pika-pos", type=parse_vec3, default=None)
    parser.add_argument("--pika-euler", type=parse_vec3, default=parse_vec3("0 1.54 0"))
    parser.add_argument("--pika-tcp-offset", type=parse_vec3, default=DEFAULT_PIKA_TCP_OFFSET)
    parser.add_argument("--pika-link7-pos", default="0 0 0.1358")
    parser.add_argument("--pika-link8-pos", default="0 0 0.1358")
    parser.add_argument("--pika-link7-euler", default="1.57 0 0")
    parser.add_argument("--pika-link8-euler", default="-1.54 0 -3.14")
    parser.add_argument("--gripper-open-width", type=float, default=0.098)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        episodes=[args.episode],
        return_uint8=True,
        video_backend=args.video_backend,
    )

    pika_pos = args.pika_pos if args.pika_pos is not None else -args.pika_tcp_offset
    model = mujoco.MjModel.from_xml_string(
        scene_xml(
            args.mesh_dir,
            vec_to_mjcf(pika_pos),
            vec_to_mjcf(args.pika_euler),
            args.pika_link7_pos,
            args.pika_link8_pos,
            args.pika_link7_euler,
            args.pika_link8_euler,
        )
    )
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=720, width=1280)
    camera = mujoco.MjvCamera()
    camera.azimuth = 135
    camera.elevation = -25
    camera.distance = 1.05
    camera.lookat[:] = args.center + np.array([0.08, 0.02, 0.02])

    mocap_ids = {
        "left": int(model.body_mocapid[model.body("left_pose").id]),
        "right": int(model.body_mocapid[model.body("right_pose").id]),
        "left_base": int(model.body_mocapid[model.body("left_base_link").id]),
        "right_base": int(model.body_mocapid[model.body("right_base_link").id]),
    }
    left_base_xyz = args.left_base_link_in_world[:3]
    right_base_xyz = args.right_base_link_in_world[:3]
    left_base_rot = euler_to_matrix(*args.left_base_link_in_world[3:6])
    right_base_rot = euler_to_matrix(*args.right_base_link_in_world[3:6])
    data.mocap_pos[mocap_ids["left_base"]] = left_base_xyz * args.scale + args.center
    data.mocap_quat[mocap_ids["left_base"]] = rotmat_to_quat_wxyz(left_base_rot)
    data.mocap_pos[mocap_ids["right_base"]] = right_base_xyz * args.scale + args.center
    data.mocap_quat[mocap_ids["right_base"]] = rotmat_to_quat_wxyz(right_base_rot)

    frames: list[np.ndarray] = []
    rows: list[dict[str, float | int]] = []
    steps = min(args.steps, len(dataset))
    left_tracker_to_tcp_rot = euler_to_matrix(*args.left_tracker_to_tcp_euler)
    right_tracker_to_tcp_rot = euler_to_matrix(*args.right_tracker_to_tcp_euler)
    gripper_qpos = {
        side: (
            int(model.jnt_qposadr[model.joint(f"{side}_pika_joint7").id]),
            int(model.jnt_qposadr[model.joint(f"{side}_pika_joint8").id]),
        )
        for side in ("left", "right")
    }

    for step in range(steps):
        values = tensor_values(dataset[step], args.field)
        left_xyz, left_rot, left_gripper, right_xyz, right_rot, right_gripper = pose_from_vector(
            values, use_row_pose6d=args.use_row_pose6d
        )
        left_xyz, left_rot = apply_tracker_to_tcp(
            left_xyz, left_rot, args.left_tracker_to_tcp_pos, left_tracker_to_tcp_rot
        )
        right_xyz, right_rot = apply_tracker_to_tcp(
            right_xyz, right_rot, args.right_tracker_to_tcp_pos, right_tracker_to_tcp_rot
        )
        left_draw = left_xyz * args.scale + args.center
        right_draw = right_xyz * args.scale + args.center

        data.mocap_pos[mocap_ids["left"]] = left_draw
        data.mocap_quat[mocap_ids["left"]] = rotmat_to_quat_wxyz(left_rot)
        data.mocap_pos[mocap_ids["right"]] = right_draw
        data.mocap_quat[mocap_ids["right"]] = rotmat_to_quat_wxyz(right_rot)
        left_open = float(np.clip(left_gripper / args.gripper_open_width, 0.0, 1.0) * 0.035)
        right_open = float(np.clip(right_gripper / args.gripper_open_width, 0.0, 1.0) * 0.035)
        data.qpos[list(gripper_qpos["left"])] = left_open
        data.qpos[list(gripper_qpos["right"])] = right_open
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render())

        row = {
            "step": step,
            "left_x": float(left_xyz[0]),
            "left_y": float(left_xyz[1]),
            "left_z": float(left_xyz[2]),
            "left_gripper": left_gripper,
            "right_x": float(right_xyz[0]),
            "right_y": float(right_xyz[1]),
            "right_z": float(right_xyz[2]),
            "right_gripper": right_gripper,
        }
        for prefix, rot in (("left", left_rot), ("right", right_rot)):
            for r in range(3):
                for c in range(3):
                    row[f"{prefix}_r{r}{c}"] = float(rot[r, c])
        rows.append(row)

    safe_field = args.field.replace(".", "_")
    stem = f"episode{args.episode}_{safe_field}_pose"
    video_path = args.output_dir / f"{stem}.mp4"
    csv_path = args.output_dir / f"{stem}.csv"
    imageio.mimsave(video_path, frames, fps=args.fps)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"video: {video_path}")
    print(f"csv: {csv_path}")
    print(f"steps: {steps}")
    print(f"left_xyz_min: {np.min([[r['left_x'], r['left_y'], r['left_z']] for r in rows], axis=0)}")
    print(f"left_xyz_max: {np.max([[r['left_x'], r['left_y'], r['left_z']] for r in rows], axis=0)}")
    print(f"right_xyz_min: {np.min([[r['right_x'], r['right_y'], r['right_z']] for r in rows], axis=0)}")
    print(f"right_xyz_max: {np.max([[r['right_x'], r['right_y'], r['right_z']] for r in rows], axis=0)}")


if __name__ == "__main__":
    main()
