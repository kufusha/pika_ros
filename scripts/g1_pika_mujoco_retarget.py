#!/usr/bin/env python3
"""Retarget LeRobot PIKA hand targets to Unitree G1 arms with MuJoCo IK."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_policy
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: F401 - registers "act"
from lerobot.policies.pi0.configuration_pi0 import PI0Config  # noqa: F401 - registers "pi0"


PIKA_ROS_DIR = Path(__file__).resolve().parents[1]
PIKA_MESH_DIR = Path(
    os.environ.get(
        "PIKA_MESH_DIR",
        PIKA_ROS_DIR / "src/PikaAnyArm/piper/piper_ros/piper_description/meshes",
    )
).expanduser()
TCP_OFFSET = np.array([0.18, 0.0, 0.0], dtype=np.float64)


def load_start_pose_config(path: Path) -> dict:
    resolved = path.expanduser().absolute()
    if not resolved.is_file():
        raise FileNotFoundError(f"retarget start pose config was not found: {resolved}")
    config = json.loads(resolved.read_text())
    for key in ("name", "left_arm_q_g1_rad", "right_arm_q_g1_rad", "right_pika_tcp"):
        if key not in config:
            raise ValueError(f"missing {key!r} in start pose config: {resolved}")
    config["_path"] = resolved
    return config


def validate_start_pose_fk(unitree_ik, config: dict, q_g1: np.ndarray) -> tuple[float, float]:
    expected_position = np.asarray(config["right_pika_tcp"]["position_pelvis_m"], dtype=np.float64)
    expected_rotation = np.asarray(config["right_pika_tcp"]["rotation_matrix"], dtype=np.float64)
    q_pin = q_g1[unitree_ik._arm_reorder_g1_to_pin]
    pin_data = unitree_ik.reduced_robot.model.createData()
    unitree_ik._pin.framesForwardKinematics(unitree_ik.reduced_robot.model, pin_data, q_pin)
    unitree_ik._pin.updateFramePlacements(unitree_ik.reduced_robot.model, pin_data)
    actual = pin_data.oMf[unitree_ik.R_hand_id].homogeneous
    position_error = float(np.linalg.norm(actual[:3, 3] - expected_position))
    rotation_error_rad = float(np.linalg.norm(rotation_error(expected_rotation, actual[:3, :3])))
    tolerances = config.get("fk_validation", {})
    position_tolerance = float(tolerances.get("position_tolerance_m", 1e-5))
    rotation_tolerance = float(tolerances.get("rotation_tolerance_rad", 1e-5))
    if position_error > position_tolerance or rotation_error_rad > rotation_tolerance:
        raise RuntimeError(
            "retarget start pose does not match the current G1+PIKA URDF: "
            f"position_error={position_error:.9f}m, "
            f"rotation_error={rotation_error_rad:.9f}rad, config={config['_path']}"
        )
    return position_error, rotation_error_rad


def resolve_g1_xml(explicit_path: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path.expanduser())
    if env_path := os.environ.get("G1_MUJOCO_XML"):
        candidates.append(Path(env_path).expanduser())

    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    candidates.extend(
        sorted(
            (cache_root / "hub/models--lerobot--unitree-g1-mujoco/snapshots").glob(
                "*/assets/g1_29dof_no_hand.xml"
            )
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            # Keep the snapshot path intact. Hugging Face assets are symlinks to
            # blobs, while mesh paths are relative to the snapshot assets dir.
            return candidate.absolute()

    searched = "\n  ".join(str(path) for path in candidates) or "(no candidates)"
    raise FileNotFoundError(
        "g1_29dof_no_hand.xml was not found. Pass --g1-xml or set G1_MUJOCO_XML.\n"
        f"Searched:\n  {searched}"
    )

LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
LEFT_ACTION_KEYS = [
    "kLeftShoulderPitch.q",
    "kLeftShoulderRoll.q",
    "kLeftShoulderYaw.q",
    "kLeftElbow.q",
    "kLeftWristRoll.q",
    "kLeftWristPitch.q",
    "kLeftWristYaw.q",
]
RIGHT_ACTION_KEYS = [
    "kRightShoulderPitch.q",
    "kRightShoulderRoll.q",
    "kRightShoulderYaw.q",
    "kRightElbow.q",
    "kRightWristRoll.q",
    "kRightWristPitch.q",
    "kRightWristYaw.q",
]

MARKER_ASSETS = """
    <material name="retarget_target_left" rgba="0.05 0.25 1.00 0.45"/>
    <material name="retarget_target_right" rgba="1.00 0.15 0.05 0.45"/>
    <material name="retarget_tcp_left" rgba="0.05 0.85 1.00 1.00"/>
    <material name="retarget_tcp_right" rgba="1.00 0.60 0.05 1.00"/>
    <texture name="retarget_floor_grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.80 0.82 0.82" rgb2="0.62 0.65 0.65"/>
    <material name="retarget_floor_mat" texture="retarget_floor_grid" texrepeat="8 8" reflectance="0.04"/>
"""

MARKER_WORLD = """
    <light name="retarget_key" pos="1.5 -3.0 3.2" dir="-0.4 0.8 -1" diffuse="0.9 0.9 0.9"/>
    <light name="retarget_fill" pos="-2.0 1.5 2.2" dir="0.8 -0.5 -1" diffuse="0.35 0.35 0.35"/>
    <camera name="retarget_camera" pos="1.75 -2.25 1.25" xyaxes="0.78 0.62 0 -0.28 0.35 0.89"/>

    <camera name="retarget_front_camera" pos="1.3 0 1.0" xyaxes="0 1 0 0 0 1"/>
    <camera name="retarget_left_camera" pos="0 1.3 1.0" xyaxes="-1 0 0 0 0 1"/>
    <camera name="retarget_oblique_camera" pos="1.6 -2.2 1.55" xyaxes="0.78 0.62 0 -0.28 0.35 0.89"/>
    <camera name="retarget_back_camera" pos="-1.3 0 1.0" xyaxes="0 -1 0 0 0 1"/>

    <geom name="retarget_floor" type="plane" pos="0 0 0" size="2.0 2.0 0.02" material="retarget_floor_mat"/>

    <body name="left_target_marker" mocap="true" pos="0 0 1">
      <geom type="sphere" size="0.024" material="retarget_target_left"/>
    </body>
    <body name="right_target_marker" mocap="true" pos="0 0 1">
      <geom type="sphere" size="0.024" material="retarget_target_right"/>
    </body>
    <body name="left_tcp_marker" mocap="true" pos="0 0 1">
      <geom type="sphere" size="0.030" material="retarget_tcp_left"/>
    </body>
    <body name="right_tcp_marker" mocap="true" pos="0 0 1">
      <geom type="sphere" size="0.030" material="retarget_tcp_right"/>
    </body>
"""


def pika_mesh_assets(mesh_dir: Path) -> str:
    mesh_dir = mesh_dir.resolve()
    return f"""
    <mesh name="pika_gripper_base_mesh" file="{mesh_dir / "gripper_base.STL"}"/>
    <mesh name="pika_link7_mesh" file="{mesh_dir / "link7.STL"}"/>
    <mesh name="pika_link8_mesh" file="{mesh_dir / "link8.STL"}"/>
"""


def pika_gripper_body(
    side: str,
    mount_pos: str,
    mount_frame_euler: str,
    mount_euler: str,
    link7_pos: str,
    link8_pos: str,
    link7_euler: str,
    link8_euler: str,
) -> str:
    return f"""
                          <body name="{side}_pika_mount" pos="{mount_pos}" euler="{mount_frame_euler}">
                            <body name="{side}_pika_gripper" euler="{mount_euler}">
                              <inertial pos="0 0 0.032" mass="0.206" diaginertia="0.00012 0.00012 0.00012"/>
                              <geom type="mesh" mesh="pika_gripper_base_mesh" rgba="0.76 0.79 0.86 1" contype="0" conaffinity="0"/>
                              <body name="{side}_pika_link7" pos="{link7_pos}" euler="{link7_euler}">
                                <joint name="{side}_pika_joint7" type="slide" axis="0 0 1" range="0 0.035" limited="true" damping="0.02"/>
                                <geom type="mesh" mesh="pika_link7_mesh" rgba="0.76 0.79 0.86 1" contype="0" conaffinity="0"/>
                              </body>
                              <body name="{side}_pika_link8" pos="{link8_pos}" euler="{link8_euler}">
                                <joint name="{side}_pika_joint8" type="slide" axis="0 0 1" range="0 0.035" limited="true" damping="0.02"/>
                                <geom type="mesh" mesh="pika_link8_mesh" rgba="0.76 0.79 0.86 1" contype="0" conaffinity="0"/>
                              </body>
                            </body>
                          </body>
                          <body name="{side}_pika_tcp_mount" euler="{mount_frame_euler}">
                            <body name="{side}_pika_tcp" pos="0.18 0 0" euler="{mount_euler}"/>
                          </body>
"""


def add_trajectory_to_scene(
    scene: mujoco.MjvScene,
    points: list[np.ndarray],
    rgba: np.ndarray,
    width: float,
) -> None:
    for start, end in zip(points, points[1:], strict=False):
        if scene.ngeom >= len(scene.geoms):
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).reshape(-1),
            rgba,
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, width, start, end)
        scene.ngeom += 1


def add_trajectory_legend(frame: np.ndarray, origin: tuple[int, int] = (0, 0)) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(frame, (x + 18, y + 17), (x + 315, y + 91), (15, 15, 15), thickness=-1)
    cv2.line(frame, (x + 34, y + 42), (x + 86, y + 42), (255, 40, 20), thickness=4)
    cv2.putText(frame, "TARGET TCP", (x + 101, y + 49), font, 0.63, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (x + 34, y + 72), (x + 86, y + 72), (20, 220, 255), thickness=4)
    cv2.putText(frame, "IK TCP", (x + 101, y + 79), font, 0.63, (255, 255, 255), 2, cv2.LINE_AA)


def render_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    layout: str,
    target_history: list[np.ndarray] | None = None,
    tcp_history: list[np.ndarray] | None = None,
) -> np.ndarray:
    def render_view(camera: str) -> np.ndarray:
        renderer.update_scene(data, camera=camera)
        if target_history is not None and tcp_history is not None:
            add_trajectory_to_scene(
                renderer.scene,
                target_history,
                np.array([1.0, 0.16, 0.08, 0.95], dtype=np.float32),
                4.0,
            )
            add_trajectory_to_scene(
                renderer.scene,
                tcp_history,
                np.array([0.08, 0.86, 1.0, 1.0], dtype=np.float32),
                3.0,
            )
        frame = renderer.render().copy()
        return frame

    if layout == "single":
        frame = render_view("retarget_camera")
        if target_history is not None and tcp_history is not None:
            add_trajectory_legend(frame)
        return frame

    views = []
    for camera in (
        "retarget_front_camera",
        "retarget_left_camera",
        "retarget_oblique_camera",
        "retarget_back_camera",
    ):
        views.append(render_view(camera))
    frame = np.vstack([np.hstack(views[:2]), np.hstack(views[2:])])
    if target_history is not None and tcp_history is not None:
        view_height, view_width = views[0].shape[:2]
        for origin in ((0, 0), (view_width, 0), (0, view_height), (view_width, view_height)):
            add_trajectory_legend(frame, origin)
    return frame


def prepare_scene_xml(
    g1_xml: Path,
    output_dir: Path,
    pika_pos: str,
    left_pika_mount_euler: str,
    right_pika_mount_euler: str,
    pika_euler: str,
    right_pika_euler: str,
    link7_pos: str,
    link8_pos: str,
    link7_euler: str,
    link8_euler: str,
    camera_pos: str,
    camera_xyaxes: str,
) -> Path:
    xml = g1_xml.read_text()
    meshes_dir = g1_xml.parent / "meshes"
    xml = xml.replace('meshdir="meshes"', f'meshdir="{meshes_dir}"')
    xml = xml.replace(
        "  <default>",
        '  <visual>\n'
        '    <global offwidth="1280" offheight="720"/>\n'
        '    <quality shadowsize="2048"/>\n'
        '  </visual>\n\n'
        "  <default>",
        1,
    )
    xml = xml.replace("</asset>", f"{MARKER_ASSETS}\n{pika_mesh_assets(PIKA_MESH_DIR)}\n  </asset>")
    marker_world = MARKER_WORLD.replace(
        '<camera name="retarget_camera" pos="1.75 -2.25 1.25" xyaxes="0.78 0.62 0 -0.28 0.35 0.89"/>',
        f'<camera name="retarget_camera" pos="{camera_pos}" xyaxes="{camera_xyaxes}"/>',
    )
    xml = xml.replace("<worldbody>", f"<worldbody>\n{marker_world}", 1)
    xml = xml.replace(
        """                          <body name="left_rubber_hand" >
                            <inertial pos="0.0001 0.0001 0.0001" mass="0.0001" diaginertia="0 0 0" />
                            <geom pos="0.0415 0.003 0" quat="1 0 0 0" type="mesh"  density="0" rgba="0.7 0.7 0.7 1"
                              mesh="left_rubber_hand" />
                          </body>""",
        pika_gripper_body(
            "left", pika_pos, left_pika_mount_euler, pika_euler,
            link7_pos, link8_pos, link7_euler, link8_euler,
        ),
    )
    xml = xml.replace(
        """                          <body name="right_rubber_hand" >
                            <inertial pos="0.0001 0.0001 0.0001" mass="0.0001" diaginertia="0 0 0" />
                            <geom pos="0.0415 -0.003 0" quat="1 0 0 0" type="mesh"  density="0" rgba="0.7 0.7 0.7 1"
                              mesh="right_rubber_hand" />
                          </body>""",
        pika_gripper_body(
            "right", pika_pos, right_pika_mount_euler, right_pika_euler,
            link7_pos, link8_pos, link7_euler, link8_euler,
        ),
    )

    scene_dir = output_dir / "g1_pika_scene"
    scene_dir.mkdir(parents=True, exist_ok=True)
    scene_xml = scene_dir / "g1_pika_retarget.xml"
    scene_xml.write_text(xml)
    return scene_xml


def prepare_batch(item: dict, camera_keys: list[str]) -> dict[str, Any]:
    batch = {key: value.unsqueeze(0) for key, value in item.items() if torch.is_tensor(value)}
    if isinstance(item.get("task"), str):
        batch["task"] = item["task"]
    for cam_key in camera_keys:
        if cam_key in batch and batch[cam_key].dtype == torch.uint8:
            batch[cam_key] = batch[cam_key].float() / 255.0
    return batch


def parse_vec3(value: str) -> np.ndarray:
    parts = [float(x) for x in value.replace(",", " ").split()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected 3 numbers, e.g. '0 0 0.08'")
    return np.array(parts, dtype=np.float64)


def parse_vec7(value: str) -> np.ndarray:
    parts = [float(x) for x in value.replace(",", " ").split()]
    if len(parts) != 7:
        raise argparse.ArgumentTypeError("expected 7 numbers")
    return np.array(parts, dtype=np.float64)


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


def action_xyz(action: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    values = action.detach().cpu().numpy().astype(np.float64)
    return values[0:3].copy(), values[10:13].copy()


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
    if n0 < 1e-8:
        col0 = np.array([1.0, 0.0, 0.0])
    else:
        col0 = col0 / n0

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


def action_pose(
    action: torch.Tensor,
    use_row_pose6d: bool,
    single_arm: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = action.detach().cpu().numpy().astype(np.float64)
    if values.shape[0] == 10:
        xyz = values[0:3].copy()
        rot = rotation6d_to_matrix(values[3:9], use_row_pose6d=use_row_pose6d)
        zero_xyz = np.zeros(3, dtype=np.float64)
        identity_rot = np.eye(3, dtype=np.float64)
        if single_arm == "left":
            return xyz, rot, zero_xyz, identity_rot
        return zero_xyz, identity_rot, xyz, rot

    left_xyz = values[0:3].copy()
    right_xyz = values[10:13].copy()
    left_rot = rotation6d_to_matrix(values[3:9], use_row_pose6d=use_row_pose6d)
    right_rot = rotation6d_to_matrix(values[13:19], use_row_pose6d=use_row_pose6d)
    return left_xyz, left_rot, right_xyz, right_rot


def action_grippers(action: torch.Tensor, open_width: float, open_angle: float, single_arm: str) -> tuple[float, float]:
    values = action.detach().cpu().numpy().astype(np.float64)
    if values.shape[0] == 10:
        value = float(np.clip(values[9] / open_width, 0.0, 1.0) * open_angle)
        if single_arm == "left":
            return value, 0.0
        return 0.0, value

    left = float(np.clip(values[9] / open_width, 0.0, 1.0) * open_angle)
    right = float(np.clip(values[19] / open_width, 0.0, 1.0) * open_angle)
    return left, right


def action_gripper_widths(action: torch.Tensor, open_width: float, single_arm: str) -> tuple[float, float]:
    values = action.detach().cpu().numpy().astype(np.float64)
    if values.shape[0] == 10:
        value = float(np.clip(values[9], 0.0, open_width))
        if single_arm == "left":
            return value, 0.0
        return 0.0, value

    left = float(np.clip(values[9], 0.0, open_width))
    right = float(np.clip(values[19], 0.0, open_width))
    return left, right


def set_visual_grippers(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left_width: float,
    right_width: float,
    open_width: float,
) -> None:
    left_q = np.clip(left_width / open_width, 0.0, 1.0) * 0.035
    right_q = np.clip(right_width / open_width, 0.0, 1.0) * 0.035
    for side, value in (("left", left_q), ("right", right_q)):
        joint7 = model.joint(f"{side}_pika_joint7").id
        joint8 = model.joint(f"{side}_pika_joint8").id
        data.qpos[int(model.jnt_qposadr[joint7])] = value
        data.qpos[int(model.jnt_qposadr[joint8])] = value


def tcp_position(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    body_id = model.body(body_name).id
    if body_name.endswith("_pika_tcp"):
        return data.xpos[body_id].copy()
    rotation = data.xmat[body_id].reshape(3, 3)
    return data.xpos[body_id] + rotation @ TCP_OFFSET


def body_rotation(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    body_id = model.body(body_name).id
    return data.xmat[body_id].reshape(3, 3).copy()


def rotation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    error_mat = target @ current.T
    cos_theta = np.clip((np.trace(error_mat) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    vee = np.array(
        [
            error_mat[2, 1] - error_mat[1, 2],
            error_mat[0, 2] - error_mat[2, 0],
            error_mat[1, 0] - error_mat[0, 1],
        ],
        dtype=np.float64,
    )
    if theta < 1e-6:
        return 0.5 * vee
    return theta / (2.0 * np.sin(theta)) * vee


def orientation_error_weights(mode: str, weight: float, axis_weight: np.ndarray | None) -> np.ndarray:
    if axis_weight is not None:
        arm_weight = weight * axis_weight.astype(np.float64)
        return np.concatenate([arm_weight, arm_weight])
    if mode == "full":
        arm_weight = np.array([weight, weight, weight], dtype=np.float64)
    elif mode == "tool-z":
        # Keep the tool Z-axis direction, but leave rotation about that Z-axis loose.
        arm_weight = np.array([weight, weight, 0.0], dtype=np.float64)
    else:
        arm_weight = np.zeros(3, dtype=np.float64)
    return np.concatenate([arm_weight, arm_weight])


def transform_matrix(translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = rotation
    tf[:3, 3] = translation
    return tf


def invert_transform(tf: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = tf[:3, :3].T
    inv[:3, 3] = -inv[:3, :3] @ tf[:3, 3]
    return inv


def joint_addresses(model: mujoco.MjModel, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    qpos = []
    dofs = []
    for name in joint_names:
        joint_id = model.joint(name).id
        qpos.append(int(model.jnt_qposadr[joint_id]))
        dofs.append(int(model.jnt_dofadr[joint_id]))
    return np.array(qpos, dtype=np.int32), np.array(dofs, dtype=np.int32)


def clip_joint_ranges(model: mujoco.MjModel, data: mujoco.MjData, joint_names: list[str]) -> None:
    for name in joint_names:
        joint_id = model.joint(name).id
        if not model.jnt_limited[joint_id]:
            continue
        qposadr = int(model.jnt_qposadr[joint_id])
        low, high = model.jnt_range[joint_id]
        data.qpos[qposadr] = np.clip(data.qpos[qposadr], low, high)


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left_target: np.ndarray,
    right_target: np.ndarray,
    left_rot_target: np.ndarray | None,
    right_rot_target: np.ndarray | None,
    qpos_ids: np.ndarray,
    dof_ids: np.ndarray,
    iterations: int,
    damping: float,
    gain: float,
    max_delta: float,
    rotation_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    left_body = model.body("left_wrist_yaw_link").id
    right_body = model.body("right_wrist_yaw_link").id
    all_arm_joints = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS

    for _ in range(iterations):
        mujoco.mj_forward(model, data)
        left_tcp = tcp_position(model, data, "left_wrist_yaw_link")
        right_tcp = tcp_position(model, data, "right_wrist_yaw_link")
        errors = [left_target - left_tcp, right_target - right_tcp]
        use_orientation = (
            left_rot_target is not None
            and right_rot_target is not None
            and bool(np.any(rotation_weight > 0.0))
        )
        if use_orientation:
            left_rot = body_rotation(model, data, "left_wrist_yaw_link")
            right_rot = body_rotation(model, data, "right_wrist_yaw_link")
            errors.extend(
                [
                    rotation_weight[:3] * rotation_error(left_rot_target, left_rot),
                    rotation_weight[3:] * rotation_error(right_rot_target, right_rot),
                ]
            )
        error = np.concatenate(errors)
        if np.linalg.norm(error) < 1e-4:
            break

        jacp_l = np.zeros((3, model.nv), dtype=np.float64)
        jacp_r = np.zeros((3, model.nv), dtype=np.float64)
        jacr_l = np.zeros((3, model.nv), dtype=np.float64)
        jacr_r = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jac(model, data, jacp_l, jacr_l, left_tcp, left_body)
        mujoco.mj_jac(model, data, jacp_r, jacr_r, right_tcp, right_body)

        jac_rows = [jacp_l[:, dof_ids], jacp_r[:, dof_ids]]
        if use_orientation:
            jac_rows.extend(
                [
                    rotation_weight[:3, None] * jacr_l[:, dof_ids],
                    rotation_weight[3:, None] * jacr_r[:, dof_ids],
                ]
            )
        jac = np.vstack(jac_rows)
        lhs = jac @ jac.T + (damping * damping) * np.eye(jac.shape[0])
        dq = gain * jac.T @ np.linalg.solve(lhs, error)
        dq = np.clip(dq, -max_delta, max_delta)

        data.qpos[qpos_ids] += dq
        clip_joint_ranges(model, data, all_arm_joints)

    mujoco.mj_forward(model, data)
    left_tcp = tcp_position(model, data, "left_wrist_yaw_link")
    right_tcp = tcp_position(model, data, "right_wrist_yaw_link")
    left_err = float(np.linalg.norm(left_target - left_tcp))
    right_err = float(np.linalg.norm(right_target - right_tcp))
    left_rot_err = 0.0
    right_rot_err = 0.0
    if left_rot_target is not None and right_rot_target is not None:
        left_rot_err = float(np.linalg.norm(rotation_error(left_rot_target, body_rotation(model, data, "left_wrist_yaw_link"))))
        right_rot_err = float(np.linalg.norm(rotation_error(right_rot_target, body_rotation(model, data, "right_wrist_yaw_link"))))
    return left_tcp, right_tcp, left_err, right_err, left_rot_err, right_rot_err


def make_processors(cfg, dataset_stats: dict):
    if cfg.type == "act":
        from lerobot.policies.act.processor_act import make_act_pre_post_processors

        return make_act_pre_post_processors(cfg, dataset_stats=dataset_stats)
    if cfg.type == "pi0":
        from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors

        return make_pi0_pre_post_processors(cfg, dataset_stats=dataset_stats)
    raise ValueError(f"unsupported policy type for MuJoCo rollout: {cfg.type}")


def load_policy(args: argparse.Namespace, meta_dataset: LeRobotDataset):
    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    cfg.device = args.device
    cfg.pretrained_path = args.policy_path
    cfg.n_action_steps = args.policy_action_steps
    if hasattr(cfg, "pretrained_backbone_weights"):
        cfg.pretrained_backbone_weights = None
    policy = make_policy(cfg, ds_meta=meta_dataset.meta)
    policy.eval()
    policy.reset()
    preprocessor, postprocessor = make_processors(cfg, meta_dataset.meta.stats)
    return policy, preprocessor, postprocessor


def replace_relative_policy_state(batch: dict[str, torch.Tensor], gripper_width: float) -> None:
    """Replace state with current-TCP identity pose and simulated gripper feedback."""
    current = batch["observation.state"]
    state = np.array(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, gripper_width],
        dtype=np.float32,
    )
    value = torch.as_tensor(state, dtype=current.dtype, device=current.device).unsqueeze(0)
    if tuple(value.shape) != tuple(current.shape):
        raise ValueError(
            f"relative rollout state shape {tuple(value.shape)} does not match "
            f"dataset state shape {tuple(current.shape)}"
        )
    batch["observation.state"] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--policy-path", type=Path)
    parser.add_argument("--source", choices=["teacher", "policy"], default="policy")
    parser.add_argument("--repo-id", default="data")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--g1-xml", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("videos/mujoco_retarget"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--policy-action-steps",
        type=int,
        default=1,
        help="Number of predicted policy actions to consume before recomputing from a new observation.",
    )
    parser.add_argument(
        "--policy-state-source",
        choices=["dataset", "relative-rollout"],
        default="dataset",
        help=(
            "Policy observation.state source. relative-rollout uses the current-TCP identity pose "
            "and feeds the previous simulated gripper command back as state."
        ),
    )
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--use-row-pose6d", action="store_true")
    parser.add_argument("--single-arm", choices=["auto", "left", "right"], default="auto")
    parser.add_argument("--ik-backend", choices=["mujoco", "unitree"], default="mujoco")
    parser.add_argument("--motion-scale", type=float, default=1.0)
    parser.add_argument("--motion-scale-xyz", type=parse_vec3, default=parse_vec3("0.5 0.5 0.5"))
    parser.add_argument("--motion-align-euler", type=parse_vec3, default=parse_vec3("0 0 3.14159"))
    parser.add_argument("--action-reference", choices=["absolute", "relative-delta"], default="absolute")
    parser.add_argument("--relative-delta-frame", choices=["local", "world"], default="local")
    parser.add_argument("--relative-delta-align-euler", type=parse_vec3, default=None)
    parser.add_argument("--left-home-offset", type=parse_vec3, default=parse_vec3("0 0 0"))
    parser.add_argument("--right-home-offset", type=parse_vec3, default=parse_vec3("0 0 0"))
    parser.add_argument("--left-work-start", type=parse_vec3, default=parse_vec3("0.25 0.12 0.85"))
    parser.add_argument("--right-work-start", type=parse_vec3, default=parse_vec3("0.25 -0.12 0.85"))
    parser.add_argument(
        "--right-start-xyz",
        type=parse_vec3,
        default=None,
        help="Explicit right TCP start in MuJoCo coordinates; overrides the FK/work-start position.",
    )
    parser.add_argument("--left-work-start-euler", type=parse_vec3, default=None)
    parser.add_argument("--right-work-start-euler", type=parse_vec3, default=None)
    parser.add_argument("--initial-left-q", type=parse_vec7)
    parser.add_argument("--initial-right-q", type=parse_vec7)
    parser.add_argument(
        "--start-pose-config",
        type=Path,
        help="Canonical G1+PIKA start pose JSON; supplies arm q and validates it against the URDF.",
    )
    parser.add_argument("--use-initial-q-as-work-start", action="store_true")
    parser.add_argument("--ik-iterations", type=int, default=35)
    parser.add_argument("--ik-damping", type=float, default=0.04)
    parser.add_argument("--ik-gain", type=float, default=0.7)
    parser.add_argument("--max-joint-delta", type=float, default=0.08)
    parser.add_argument("--unitree-ik-smooth-weight", type=float, default=0.1)
    parser.add_argument("--unitree-ik-regularization-weight", type=float, default=0.02)
    parser.add_argument("--unitree-ik-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--unitree-ik-method", choices=["casadi", "scipy-right"], default="casadi")
    parser.add_argument("--ik-position-tolerance-m", type=float, default=0.005)
    parser.add_argument("--ik-rotation-tolerance-rad", type=float, default=0.05)
    parser.add_argument("--fail-on-ik-error", action="store_true")
    parser.add_argument("--gripper-open-width", type=float, default=0.098)
    parser.add_argument("--gripper-open-angle", type=float, default=1.67)
    parser.add_argument("--use-orientation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--orientation-mode", choices=["full", "tool-z"], default="full")
    parser.add_argument("--orientation-weight", type=float, default=0.03)
    parser.add_argument("--orientation-axis-weight", type=parse_vec3)
    parser.add_argument("--pika-pos", default="0.0415 0 0")
    parser.add_argument("--left-pika-mount-euler", default="0 0 0")
    parser.add_argument("--right-pika-mount-euler", default="0 0 0")
    parser.add_argument("--pika-euler", default="0 1.5708 0")
    parser.add_argument("--right-pika-euler", default=None)
    parser.add_argument("--pika-link7-pos", default="0 0 0.1358")
    parser.add_argument("--pika-link8-pos", default="0 0 0.1358")
    parser.add_argument("--pika-link7-euler", default="1.57 0 0")
    parser.add_argument("--pika-link8-euler", default="-1.54 0 -3.14")
    parser.add_argument("--left-tracker-to-tcp-pos", type=parse_vec3, default=parse_vec3("0 0 0"))
    parser.add_argument("--right-tracker-to-tcp-pos", type=parse_vec3, default=parse_vec3("0 0 0"))
    parser.add_argument("--left-tracker-to-tcp-euler", type=parse_vec3, default=parse_vec3("0 0 0"))
    parser.add_argument("--right-tracker-to-tcp-euler", type=parse_vec3, default=parse_vec3("0 0 0"))
    parser.add_argument("--left-wrist-to-pika-euler", type=parse_vec3, default=parse_vec3("0 0 0"))
    parser.add_argument("--right-wrist-to-pika-euler", type=parse_vec3, default=parse_vec3("0 0 0"))
    parser.add_argument("--camera-pos", default="1.75 -2.25 1.25")
    parser.add_argument("--camera-xyaxes", default="0.78 0.62 0 -0.28 0.35 0.89")
    parser.add_argument("--camera-layout", choices=["single", "four"], default="single")
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--render-start-step", type=int, default=0)
    parser.add_argument("--render-end-step", type=int)
    parser.add_argument("--show-trajectories", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--viewer-realtime", type=float, default=1.0)
    args = parser.parse_args()

    start_pose_config = None
    if args.start_pose_config is not None:
        start_pose_config = load_start_pose_config(args.start_pose_config)
        config_left_q = np.asarray(start_pose_config["left_arm_q_g1_rad"], dtype=np.float64)
        config_right_q = np.asarray(start_pose_config["right_arm_q_g1_rad"], dtype=np.float64)
        if config_left_q.shape != (7,) or config_right_q.shape != (7,):
            raise ValueError("start pose arm joint arrays must each contain 7 values")
        if args.initial_left_q is not None and not np.allclose(args.initial_left_q, config_left_q):
            raise ValueError("--initial-left-q conflicts with --start-pose-config")
        if args.initial_right_q is not None and not np.allclose(args.initial_right_q, config_right_q):
            raise ValueError("--initial-right-q conflicts with --start-pose-config")
        args.initial_left_q = config_left_q
        args.initial_right_q = config_right_q

    if args.source == "policy" and args.policy_path is None:
        raise ValueError("--policy-path is required when --source policy")
    if args.source != "policy" and args.policy_state_source != "dataset":
        raise ValueError("--policy-state-source is only valid when --source policy")
    if args.policy_action_steps <= 0:
        raise ValueError("--policy-action-steps must be positive")
    if (args.initial_left_q is None) != (args.initial_right_q is None):
        raise ValueError("--initial-left-q and --initial-right-q must be provided together")
    if args.use_initial_q_as_work_start and args.initial_right_q is None:
        raise ValueError("--use-initial-q-as-work-start requires initial arm joint values")
    if args.unitree_ik_method == "scipy-right" and args.single_arm not in {"auto", "right"}:
        raise ValueError("--unitree-ik-method scipy-right requires a right single-arm trajectory")
    if args.render_every <= 0:
        raise ValueError("--render-every must be positive")
    if args.render_start_step < 0:
        raise ValueError("--render-start-step must be non-negative")
    if args.render_end_step is not None and args.render_end_step <= args.render_start_step:
        raise ValueError("--render-end-step must be greater than --render-start-step")

    args.g1_xml = resolve_g1_xml(args.g1_xml)
    if not PIKA_MESH_DIR.is_dir():
        raise FileNotFoundError(
            f"PIKA mesh directory was not found: {PIKA_MESH_DIR}. "
            "Initialize the PikaAnyArm submodule first."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene_xml = prepare_scene_xml(
        args.g1_xml,
        args.output_dir,
        args.pika_pos,
        args.left_pika_mount_euler,
        args.right_pika_mount_euler,
        args.pika_euler,
        args.right_pika_euler if args.right_pika_euler is not None else args.pika_euler,
        args.pika_link7_pos,
        args.pika_link8_pos,
        args.pika_link7_euler,
        args.pika_link8_euler,
        args.camera_pos,
        args.camera_xyaxes,
    )

    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        episodes=[args.episode],
        return_uint8=True,
        video_backend=args.video_backend,
    )
    meta_dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        return_uint8=True,
        video_backend=args.video_backend,
    )

    policy = preprocessor = postprocessor = None
    if args.source == "policy":
        policy, preprocessor, postprocessor = load_policy(args, meta_dataset)

    unitree_ik = None
    if args.ik_backend == "unitree":
        from lerobot.robots.unitree_g1_pika.g1_pika_kinematics import G1PikaArmIK

        unitree_ik = G1PikaArmIK()
        if start_pose_config is not None:
            initial_q_g1 = np.concatenate([args.initial_left_q, args.initial_right_q])
            start_pos_err, start_rot_err = validate_start_pose_fk(
                unitree_ik, start_pose_config, initial_q_g1
            )
            print(f"start_pose_config: {start_pose_config['_path']}")
            print(f"start_pose_name  : {start_pose_config['name']}")
            print(
                "start_pose_right_tcp_pelvis_m: "
                f"{np.asarray(start_pose_config['right_pika_tcp']['position_pelvis_m'])}"
            )
            print(
                "start_pose_fk_validation: OK "
                f"(position={start_pos_err:.3e}m rotation={start_rot_err:.3e}rad)"
            )

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    if args.skip_video:
        renderer = None
    elif args.camera_layout == "four":
        renderer = mujoco.Renderer(model, height=360, width=640)
    else:
        renderer = mujoco.Renderer(model, height=720, width=1280)
    marker_ids = {
        "left_target": int(model.body_mocapid[model.body("left_target_marker").id]),
        "right_target": int(model.body_mocapid[model.body("right_target_marker").id]),
        "left_tcp": int(model.body_mocapid[model.body("left_tcp_marker").id]),
        "right_tcp": int(model.body_mocapid[model.body("right_tcp_marker").id]),
    }

    left_qpos, left_dofs = joint_addresses(model, LEFT_ARM_JOINTS)
    right_qpos, right_dofs = joint_addresses(model, RIGHT_ARM_JOINTS)
    qpos_ids = np.concatenate([left_qpos, right_qpos])
    dof_ids = np.concatenate([left_dofs, right_dofs])

    mujoco.mj_forward(model, data)
    left_tcp_body = "left_pika_tcp" if args.ik_backend == "unitree" else "left_wrist_yaw_link"
    right_tcp_body = "right_pika_tcp" if args.ik_backend == "unitree" else "right_wrist_yaw_link"
    mujoco_neutral_left_tf = transform_matrix(
        tcp_position(model, data, left_tcp_body),
        body_rotation(model, data, left_tcp_body),
    )
    mujoco_neutral_right_tf = transform_matrix(
        tcp_position(model, data, right_tcp_body),
        body_rotation(model, data, right_tcp_body),
    )
    unitree_to_mujoco_left_tf = np.eye(4, dtype=np.float64)
    unitree_to_mujoco_right_tf = np.eye(4, dtype=np.float64)
    if args.ik_backend == "unitree":
        assert unitree_ik is not None
        pin = unitree_ik._pin
        q_neutral = np.zeros(unitree_ik.reduced_robot.model.nq)
        pin_data = unitree_ik.reduced_robot.model.createData()
        pin.forwardKinematics(unitree_ik.reduced_robot.model, pin_data, q_neutral)
        pin.updateFramePlacements(unitree_ik.reduced_robot.model, pin_data)
        unitree_neutral_left_tf = pin_data.oMf[unitree_ik.L_hand_id].homogeneous
        unitree_neutral_right_tf = pin_data.oMf[unitree_ik.R_hand_id].homogeneous
        unitree_to_mujoco_left_tf = mujoco_neutral_left_tf @ invert_transform(unitree_neutral_left_tf)
        unitree_to_mujoco_right_tf = mujoco_neutral_right_tf @ invert_transform(unitree_neutral_right_tf)

    q_pin = None
    if args.initial_right_q is not None:
        initial_q_g1 = np.concatenate([args.initial_left_q, args.initial_right_q])
        data.qpos[qpos_ids] = initial_q_g1
        clip_joint_ranges(model, data, LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS)
        mujoco.mj_forward(model, data)
        if unitree_ik is not None:
            q_pin = initial_q_g1[unitree_ik._arm_reorder_g1_to_pin]

    unitree_left_hold_tf = None
    unitree_right_hold_tf = None
    if unitree_ik is not None:
        if q_pin is None:
            q_pin = np.zeros(unitree_ik.reduced_robot.model.nq)
        hold_data = unitree_ik.reduced_robot.model.createData()
        unitree_ik._pin.framesForwardKinematics(unitree_ik.reduced_robot.model, hold_data, q_pin)
        unitree_ik._pin.updateFramePlacements(unitree_ik.reduced_robot.model, hold_data)
        unitree_left_hold_tf = hold_data.oMf[unitree_ik.L_hand_id].homogeneous.copy()
        unitree_right_hold_tf = hold_data.oMf[unitree_ik.R_hand_id].homogeneous.copy()

    home_left = mujoco_neutral_left_tf[:3, 3] + args.left_home_offset
    home_right = mujoco_neutral_right_tf[:3, 3] + args.right_home_offset
    if args.left_work_start is not None:
        home_left = args.left_work_start
    if args.right_work_start is not None:
        home_right = args.right_work_start
    if args.use_initial_q_as_work_start:
        home_left = tcp_position(model, data, left_tcp_body)
        home_right = tcp_position(model, data, right_tcp_body)
    if args.right_start_xyz is not None:
        home_right = args.right_start_xyz
    home_left_rot = body_rotation(model, data, left_tcp_body)
    home_right_rot = body_rotation(model, data, right_tcp_body)
    if args.left_work_start_euler is not None:
        home_left_rot = euler_to_matrix(*args.left_work_start_euler)
    if args.right_work_start_euler is not None:
        home_right_rot = euler_to_matrix(*args.right_work_start_euler)

    if args.start_index < 0 or args.start_index >= len(dataset):
        raise ValueError(f"--start-index {args.start_index} is outside dataset length {len(dataset)}")
    first_item = dataset[args.start_index]
    action_dim = int(first_item["action"].numel())
    single_arm = args.single_arm
    if single_arm == "auto":
        single_arm = "right" if action_dim == 10 else "none"
    if action_dim == 10 and single_arm not in {"left", "right"}:
        raise ValueError("10D single-arm action requires --single-arm left or --single-arm right")
    if action_dim not in {10, 20}:
        raise ValueError(f"unsupported action dimension: {action_dim}")

    rollout_gripper_width = float(first_item["observation.state"][-1])
    if args.source == "teacher":
        first_left_raw, first_left_rot, first_right_raw, first_right_rot = action_pose(
            first_item["action"], use_row_pose6d=args.use_row_pose6d, single_arm=single_arm
        )
    elif args.action_reference == "relative-delta":
        # Relative actions are integrated from the configured home pose, so no
        # policy action is needed to establish an absolute-action baseline.
        first_left_raw = np.zeros(3, dtype=np.float64)
        first_right_raw = np.zeros(3, dtype=np.float64)
        first_left_rot = np.eye(3, dtype=np.float64)
        first_right_rot = np.eye(3, dtype=np.float64)
    else:
        assert policy is not None and preprocessor is not None and postprocessor is not None
        with torch.no_grad():
            first_batch = prepare_batch(first_item, dataset.meta.camera_keys)
            if args.policy_state_source == "relative-rollout":
                replace_relative_policy_state(first_batch, rollout_gripper_width)
            first_obs = preprocessor(first_batch)
            first_pred = postprocessor(policy.select_action(first_obs)).squeeze(0).cpu()
        first_left_raw, first_left_rot, first_right_raw, first_right_rot = action_pose(
            first_pred, use_row_pose6d=args.use_row_pose6d, single_arm=single_arm
        )
    left_tracker_to_tcp_rot = euler_to_matrix(*args.left_tracker_to_tcp_euler)
    right_tracker_to_tcp_rot = euler_to_matrix(*args.right_tracker_to_tcp_euler)
    left_wrist_to_pika_rot = euler_to_matrix(*args.left_wrist_to_pika_euler)
    right_wrist_to_pika_rot = euler_to_matrix(*args.right_wrist_to_pika_euler)
    first_left_raw, first_left_rot = apply_tracker_to_tcp(
        first_left_raw, first_left_rot, args.left_tracker_to_tcp_pos, left_tracker_to_tcp_rot
    )
    first_right_raw, first_right_rot = apply_tracker_to_tcp(
        first_right_raw, first_right_rot, args.right_tracker_to_tcp_pos, right_tracker_to_tcp_rot
    )

    rows: list[dict[str, float | int]] = []
    steps = min(args.steps, len(dataset) - args.start_index)
    mode = "pose" if args.use_orientation else "pos"
    stem = (
        f"episode{args.episode}_index{args.start_index}_"
        f"g1_pika_retarget_{args.source}_{mode}_{args.ik_backend}ik"
    )
    video_path = args.output_dir / f"{stem}.mp4"
    csv_path = args.output_dir / f"{stem}.csv"
    video_process = None
    rendered_frames = 0
    right_target_history: list[np.ndarray] = []
    right_tcp_history: list[np.ndarray] = []
    motion_scale = args.motion_scale_xyz if args.motion_scale_xyz is not None else np.full(3, args.motion_scale)
    motion_align_rot = euler_to_matrix(*args.motion_align_euler)
    if unitree_ik is not None:
        if args.relative_delta_align_euler is not None:
            raise ValueError(
                "--relative-delta-align-euler is not valid with --ik-backend unitree; "
                "the PIKAsense-to-TCP transform is defined by the G1+PIKA URDF"
            )
        relative_delta_align_rot = unitree_ik.tracker_to_tcp_rotation("right")
    else:
        relative_delta_align_rot = (
            motion_align_rot
            if args.relative_delta_align_euler is None
            else euler_to_matrix(*args.relative_delta_align_euler)
        )
    rotation_weight = orientation_error_weights(
        args.orientation_mode if args.use_orientation else "none",
        args.orientation_weight,
        args.orientation_axis_weight if args.use_orientation else None,
    )
    viewer = None
    if args.viewer:
        mujoco_viewer = importlib.import_module("mujoco.viewer")
        viewer = mujoco_viewer.launch_passive(model, data)

    accumulated_left_target = home_left.copy()
    accumulated_right_target = home_right.copy()
    if args.initial_right_q is not None:
        # The recorded deltas are expressed in the PIKA tracker frame. Recover
        # that frame from the measured wrist pose so the first zero delta holds
        # the connected G1 orientation after applying the wrist/tool offset.
        accumulated_left_rot = home_left_rot @ left_wrist_to_pika_rot.T
        accumulated_right_rot = home_right_rot @ right_wrist_to_pika_rot.T
    else:
        accumulated_left_rot = home_left_rot.copy()
        accumulated_right_rot = home_right_rot.copy()

    with torch.no_grad():
        for step in range(steps):
            dataset_idx = args.start_index + step
            item = dataset[dataset_idx]
            if args.source == "teacher":
                source_action = item["action"]
                policy_state_gripper = float(item["observation.state"][-1])
            else:
                assert policy is not None and preprocessor is not None and postprocessor is not None
                batch = prepare_batch(item, dataset.meta.camera_keys)
                policy_state_gripper = (
                    rollout_gripper_width
                    if args.policy_state_source == "relative-rollout"
                    else float(item["observation.state"][-1])
                )
                if args.policy_state_source == "relative-rollout":
                    replace_relative_policy_state(batch, policy_state_gripper)
                obs = preprocessor(batch)
                source_action = postprocessor(policy.select_action(obs)).squeeze(0).cpu()
                rollout_gripper_width = float(
                    np.clip(float(source_action[-1]), 0.0, args.gripper_open_width)
                )

            left_raw, left_rot, right_raw, right_rot = action_pose(
                source_action, use_row_pose6d=args.use_row_pose6d, single_arm=single_arm
            )
            left_raw, left_rot = apply_tracker_to_tcp(
                left_raw, left_rot, args.left_tracker_to_tcp_pos, left_tracker_to_tcp_rot
            )
            right_raw, right_rot = apply_tracker_to_tcp(
                right_raw, right_rot, args.right_tracker_to_tcp_pos, right_tracker_to_tcp_rot
            )
            left_gripper, right_gripper = action_grippers(
                source_action, args.gripper_open_width, args.gripper_open_angle, single_arm
            )
            left_width, right_width = action_gripper_widths(source_action, args.gripper_open_width, single_arm)

            left_rot_target = None
            right_rot_target = None
            if args.action_reference == "relative-delta":
                if args.relative_delta_frame == "local":
                    left_delta = accumulated_left_rot @ relative_delta_align_rot @ (left_raw * motion_scale)
                    right_delta = accumulated_right_rot @ relative_delta_align_rot @ (right_raw * motion_scale)
                    left_aligned_rot = relative_delta_align_rot @ left_rot @ relative_delta_align_rot.T
                    right_aligned_rot = relative_delta_align_rot @ right_rot @ relative_delta_align_rot.T
                else:
                    left_delta = motion_align_rot @ (left_raw * motion_scale)
                    right_delta = motion_align_rot @ (right_raw * motion_scale)
                    left_aligned_rot = motion_align_rot @ left_rot @ motion_align_rot.T
                    right_aligned_rot = motion_align_rot @ right_rot @ motion_align_rot.T
                accumulated_left_target = accumulated_left_target + left_delta
                accumulated_right_target = accumulated_right_target + right_delta
                left_target = accumulated_left_target
                right_target = accumulated_right_target
                if args.use_orientation:
                    accumulated_left_rot = accumulated_left_rot @ left_aligned_rot
                    accumulated_right_rot = accumulated_right_rot @ right_aligned_rot
                    left_rot_target = accumulated_left_rot @ left_wrist_to_pika_rot
                    right_rot_target = accumulated_right_rot @ right_wrist_to_pika_rot
            else:
                left_delta = motion_align_rot @ ((left_raw - first_left_raw) * motion_scale)
                right_delta = motion_align_rot @ ((right_raw - first_right_raw) * motion_scale)
                left_target = home_left + left_delta
                right_target = home_right + right_delta
                if args.use_orientation:
                    left_relative_rot = left_rot @ first_left_rot.T
                    right_relative_rot = right_rot @ first_right_rot.T
                    left_aligned_rot = motion_align_rot @ left_relative_rot @ motion_align_rot.T
                    right_aligned_rot = motion_align_rot @ right_relative_rot @ motion_align_rot.T
                    left_rot_target = left_aligned_rot @ home_left_rot @ left_wrist_to_pika_rot
                    right_rot_target = right_aligned_rot @ home_right_rot @ right_wrist_to_pika_rot

            if args.ik_backend == "unitree":
                assert unitree_ik is not None
                left_world_tf = transform_matrix(
                    left_target,
                    left_rot_target if left_rot_target is not None else home_left_rot,
                )
                right_world_tf = transform_matrix(
                    right_target,
                    right_rot_target if right_rot_target is not None else home_right_rot,
                )
                left_tf = invert_transform(unitree_to_mujoco_left_tf) @ left_world_tf
                right_tf = invert_transform(unitree_to_mujoco_right_tf) @ right_world_tf
                if single_arm == "right":
                    assert unitree_left_hold_tf is not None
                    left_tf = unitree_left_hold_tf
                elif single_arm == "left":
                    assert unitree_right_hold_tf is not None
                    right_tf = unitree_right_hold_tf
                if args.unitree_ik_method == "scipy-right":
                    q_pin, solver_position_error, solver_rotation_error = unitree_ik.solve_right_ik(
                        right_tf,
                        q_pin,
                        position_scale=args.ik_position_tolerance_m,
                        rotation_scale=args.ik_rotation_tolerance_rad,
                        regularization=args.unitree_ik_regularization_weight,
                    )
                    if args.fail_on_ik_error and (
                        solver_position_error > args.ik_position_tolerance_m
                        or solver_rotation_error > args.ik_rotation_tolerance_rad
                    ):
                        raise RuntimeError(
                            f"IK tolerance exceeded at step {step}: "
                            f"position={solver_position_error:.6f}m "
                            f"rotation={solver_rotation_error:.6f}rad"
                        )
                else:
                    q_pin, _ = unitree_ik.solve_ik(
                        left_tf,
                        right_tf,
                        current_lr_arm_motor_q=q_pin,
                        rotation_weight=rotation_weight,
                        smooth_weight=args.unitree_ik_smooth_weight,
                        regularization_weight=args.unitree_ik_regularization_weight,
                        use_filter=args.unitree_ik_filter,
                    )
                q_g1 = np.asarray(q_pin, dtype=np.float64)[unitree_ik._arm_reorder_pin_to_g1]
                data.qpos[qpos_ids] = q_g1
                clip_joint_ranges(model, data, LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS)
                mujoco.mj_forward(model, data)
                left_tcp = tcp_position(model, data, left_tcp_body)
                right_tcp = tcp_position(model, data, right_tcp_body)
                left_err = float(np.linalg.norm(left_target - left_tcp))
                right_err = float(np.linalg.norm(right_target - right_tcp))
                left_rot_err = 0.0
                right_rot_err = 0.0
                if left_rot_target is not None and right_rot_target is not None:
                    left_rot_err = float(
                        np.linalg.norm(rotation_error(left_rot_target, body_rotation(model, data, left_tcp_body)))
                    )
                    right_rot_err = float(
                        np.linalg.norm(rotation_error(right_rot_target, body_rotation(model, data, right_tcp_body)))
                    )
            else:
                left_tcp, right_tcp, left_err, right_err, left_rot_err, right_rot_err = solve_ik(
                    model,
                    data,
                    left_target,
                    right_target,
                    left_rot_target,
                    right_rot_target,
                    qpos_ids,
                    dof_ids,
                    args.ik_iterations,
                    args.ik_damping,
                    args.ik_gain,
                    args.max_joint_delta,
                    rotation_weight,
                )

            data.mocap_pos[marker_ids["left_target"]] = left_target
            data.mocap_pos[marker_ids["right_target"]] = right_target
            data.mocap_pos[marker_ids["left_tcp"]] = left_tcp
            data.mocap_pos[marker_ids["right_tcp"]] = right_tcp
            right_target_history.append(right_target.copy())
            right_tcp_history.append(right_tcp.copy())
            set_visual_grippers(model, data, left_width, right_width, args.gripper_open_width)
            mujoco.mj_forward(model, data)

            render_end_step = steps if args.render_end_step is None else min(args.render_end_step, steps)
            should_render = (
                renderer is not None
                and args.render_start_step <= step < render_end_step
                and (step - args.render_start_step) % args.render_every == 0
            )
            if should_render:
                frame = np.ascontiguousarray(
                    render_frame(
                        model,
                        data,
                        renderer,
                        args.camera_layout,
                        right_target_history if args.show_trajectories else None,
                        right_tcp_history if args.show_trajectories else None,
                    )
                )
                if video_process is None:
                    height, width = frame.shape[:2]
                    video_process = subprocess.Popen(
                        [
                            "ffmpeg",
                            "-loglevel",
                            "error",
                            "-y",
                            "-f",
                            "rawvideo",
                            "-pixel_format",
                            "rgb24",
                            "-video_size",
                            f"{width}x{height}",
                            "-framerate",
                            str(args.fps),
                            "-i",
                            "-",
                            "-an",
                            "-c:v",
                            "libx264",
                            "-pix_fmt",
                            "yuv420p",
                            str(video_path),
                        ],
                        stdin=subprocess.PIPE,
                    )
                assert video_process.stdin is not None
                video_process.stdin.write(frame.tobytes())
                rendered_frames += 1
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
                if args.viewer_realtime > 0:
                    time.sleep(1.0 / (args.fps * args.viewer_realtime))

            row = {
                "step": step,
                "dataset_index": dataset_idx,
                "left_err": left_err,
                "right_err": right_err,
                "mean_err": (left_err + right_err) / 2.0,
                "left_rot_err_rad": left_rot_err,
                "right_rot_err_rad": right_rot_err,
                "mean_rot_err_rad": (left_rot_err + right_rot_err) / 2.0,
                "policy_state_gripper": policy_state_gripper,
                "left_target_x": float(left_target[0]),
                "left_target_y": float(left_target[1]),
                "left_target_z": float(left_target[2]),
                "right_target_x": float(right_target[0]),
                "right_target_y": float(right_target[1]),
                "right_target_z": float(right_target[2]),
                "left_tcp_x": float(left_tcp[0]),
                "left_tcp_y": float(left_tcp[1]),
                "left_tcp_z": float(left_tcp[2]),
                "right_tcp_x": float(right_tcp[0]),
                "right_tcp_y": float(right_tcp[1]),
                "right_tcp_z": float(right_tcp[2]),
            }
            for name, qposadr in zip(LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS, qpos_ids, strict=True):
                row[name] = float(data.qpos[qposadr])
            for key, qposadr in zip(LEFT_ACTION_KEYS + RIGHT_ACTION_KEYS, qpos_ids, strict=True):
                row[key] = float(data.qpos[qposadr])
            row["left_gripper.pos"] = left_gripper
            row["right_gripper.pos"] = right_gripper
            rows.append(row)

    if not rows:
        raise RuntimeError("no rows were generated")
    if video_process is not None:
        assert video_process.stdin is not None
        video_process.stdin.close()
        return_code = video_process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")
    if viewer is not None:
        viewer.close()

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    mean_left = sum(float(row["left_err"]) for row in rows) / len(rows)
    mean_right = sum(float(row["right_err"]) for row in rows) / len(rows)
    mean_left_rot = sum(float(row["left_rot_err_rad"]) for row in rows) / len(rows)
    mean_right_rot = sum(float(row["right_rot_err_rad"]) for row in rows) / len(rows)
    if rendered_frames:
        print(f"video: {video_path}")
    else:
        print("video: skipped")
    print(f"csv: {csv_path}")
    print(f"scene_xml: {scene_xml}")
    print(f"steps: {steps}")
    print(f"mean_left_ik_err: {mean_left:.5f}")
    print(f"mean_right_ik_err: {mean_right:.5f}")
    print(f"mean_left_rot_err_rad: {mean_left_rot:.5f}")
    print(f"mean_right_rot_err_rad: {mean_right_rot:.5f}")


if __name__ == "__main__":
    main()
