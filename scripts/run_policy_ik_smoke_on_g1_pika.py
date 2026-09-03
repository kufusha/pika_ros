#!/usr/bin/env python3
"""Small-scale policy -> IK -> real G1 smoke test.

This replays dataset observations through a policy, converts the right-arm
relative TCP action to G1 arm joints with Unitree IK, and sends only a heavily
limited right-arm command plus right Pika gripper command. All other joints are
held at their initial observed positions.
"""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: F401 - registers "act"
from lerobot.policies.pi0.configuration_pi0 import PI0Config  # noqa: F401 - registers "pi0"
from lerobot.policies.factory import make_policy
from lerobot.robots.unitree_g1.g1_utils import G1_29_JointArmIndex
from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1
from lerobot.robots.unitree_g1_pika.g1_pika_kinematics import G1PikaArmIK


DEFAULT_GRIPPER_OPEN_ANGLE = 1.67


RIGHT_ARM_PAIRS = [
    ("right_shoulder_pitch", "kRightShoulderPitch.q"),
    ("right_shoulder_roll", "kRightShoulderRoll.q"),
    ("right_shoulder_yaw", "kRightShoulderYaw.q"),
    ("right_elbow", "kRightElbow.q"),
    ("right_wrist_roll", "kRightWristRoll.q"),
    ("right_wrist_pitch", "kRightWristPitch.q"),
    ("right_wrist_yaw", "kRightWristYaw.q"),
]


LEFT_ARM_PAIRS = [
    ("left_shoulder_pitch", "kLeftShoulderPitch.q"),
    ("left_shoulder_roll", "kLeftShoulderRoll.q"),
    ("left_shoulder_yaw", "kLeftShoulderYaw.q"),
    ("left_elbow", "kLeftElbow.q"),
    ("left_wrist_roll", "kLeftWristRoll.q"),
    ("left_wrist_pitch", "kLeftWristPitch.q"),
    ("left_wrist_yaw", "kLeftWristYaw.q"),
]


def load_right_joint_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"trajectory CSV has no data rows: {path}")

    columns = [key for _, key in RIGHT_ARM_PAIRS]
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise ValueError(f"trajectory CSV is missing right-arm columns: {missing}")

    trajectory = np.asarray(
        [[float(row[column]) for column in columns] for row in rows],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(trajectory)):
        raise ValueError(f"trajectory CSV contains non-finite right-arm values: {path}")

    dataset_indices = np.asarray(
        [int(row.get("dataset_index", index)) for index, row in enumerate(rows)],
        dtype=np.int64,
    )
    return trajectory, dataset_indices


def interpolate_servo_target(
    servo: "RightArmTrackingServo",
    start: np.ndarray,
    target: np.ndarray,
    *,
    rate_hz: float,
    max_step_rad: float,
    minimum_duration_sec: float,
) -> None:
    max_delta = float(np.max(np.abs(target - start)))
    increments = max(
        1,
        int(np.ceil(max_delta / max_step_rad)),
        int(np.ceil(minimum_duration_sec * rate_hz)),
    )
    period = 1.0 / rate_hz
    next_tick = time.monotonic()
    for index in range(1, increments + 1):
        servo.raise_if_failed()
        servo.set_target(start + (target - start) * (index / increments))
        next_tick += period
        time.sleep(max(0.0, next_tick - time.monotonic()))


class RightArmTrackingServo:
    """Publish a gravity-compensated right-arm command at a fixed rate."""

    def __init__(
        self,
        robot,
        ik: G1PikaArmIK,
        initial_observation: dict,
        *,
        rate_hz: float,
        outer_loop_gain: float,
        correction_step_rad: float,
        max_correction_rad: float,
        tracking_error_limit_rad: float,
        max_expected_deviation_rad: float | None,
    ) -> None:
        self.robot = robot
        self.ik = ik
        self.period = 1.0 / rate_hz
        self.outer_loop_gain = outer_loop_gain
        self.correction_step_rad = correction_step_rad
        self.max_correction_rad = max_correction_rad
        self.tracking_error_limit_rad = tracking_error_limit_rad
        self.max_expected_deviation_rad = max_expected_deviation_rad

        self._right_joints = tuple(
            joint for joint in G1_29_JointArmIndex if joint.name.startswith("kRight")
        )
        self._hold_action = {
            key: float(value) for key, value in initial_observation.items() if key.endswith(".q")
        }
        initial_right = np.asarray(
            [float(initial_observation[key]) for _, key in RIGHT_ARM_PAIRS], dtype=np.float64
        )
        self._command_origin = initial_right.copy()
        self._expected_origin = initial_right.copy()
        self._expected_target = initial_right.copy()
        self._nominal_target = initial_right.copy()
        self._correction = np.zeros(7, dtype=np.float64)
        self._correction_enabled = False

        self._kp = np.zeros(29, dtype=np.float32)
        self._kd = np.zeros(29, dtype=np.float32)
        for joint in self._right_joints:
            self._kp[joint.value] = robot.kp[joint.value]
            self._kd[joint.value] = robot.kd[joint.value]

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._latest_observed = initial_right.copy()
        self._latest_command = initial_right.copy()
        self._latest_error = np.zeros(7, dtype=np.float64)

    def start(self) -> None:
        # Do not command toward a state snapshot captured before model loading,
        # camera preflight, or operator confirmation. Prime from the latest G1
        # state so engaging the hold cannot immediately trip the error guard.
        observed = UnitreeG1.get_observation(self.robot)
        observed_right = np.asarray(
            [float(observed[key]) for _, key in RIGHT_ARM_PAIRS], dtype=np.float64
        )
        if not np.all(np.isfinite(observed_right)):
            raise RuntimeError("right-arm servo received a non-finite initial observation")
        with self._lock:
            self._hold_action = {
                key: float(value) for key, value in observed.items() if key.endswith(".q")
            }
            self._command_origin = observed_right.copy()
            self._expected_origin = observed_right.copy()
            self._expected_target = observed_right.copy()
            self._nominal_target = observed_right.copy()
            self._correction.fill(0.0)
            self._correction_enabled = False
            self._latest_observed = observed_right.copy()
            self._latest_command = observed_right.copy()
            self._latest_error.fill(0.0)
        print("TRACKING_SERVO: primed from current right-arm state", flush=True)
        self._thread = threading.Thread(target=self._run, name="g1-right-arm-servo", daemon=True)
        self._thread.start()

    def capture_settled_baseline(self, settle_sec: float) -> dict:
        deadline = time.monotonic() + settle_sec
        while time.monotonic() < deadline:
            self.raise_if_failed()
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        observed = UnitreeG1.get_observation(self.robot)
        settled_right = np.asarray([float(observed[key]) for _, key in RIGHT_ARM_PAIRS], dtype=np.float64)
        with self._lock:
            self._expected_origin = settled_right.copy()
            self._expected_target = settled_right.copy()
            self._nominal_target = self._command_origin.copy()
            self._correction.fill(0.0)
            self._correction_enabled = True
        return observed

    def set_target(self, expected_right: np.ndarray) -> None:
        expected = np.asarray(expected_right, dtype=np.float64)
        if expected.shape != (7,):
            raise ValueError(f"expected a 7D right-arm target, got {expected.shape}")
        if not np.all(np.isfinite(expected)):
            raise RuntimeError("right-arm servo target contains a non-finite value")
        with self._lock:
            deviation = expected - self._expected_origin
            if (
                self.max_expected_deviation_rad is not None
                and float(np.max(np.abs(deviation))) > self.max_expected_deviation_rad + 1e-9
            ):
                raise RuntimeError("right-arm servo target exceeds expected joint-deviation limit")
            self._expected_target = expected.copy()
            self._nominal_target = self._command_origin + deviation

    def snapshot(self) -> dict[str, np.ndarray | float]:
        with self._lock:
            return {
                "observed": self._latest_observed.copy(),
                "command": self._latest_command.copy(),
                "error": self._latest_error.copy(),
                "correction": self._correction.copy(),
                "max_error": float(np.max(np.abs(self._latest_error))),
            }

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("right-arm tracking servo failed") from self._error

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise RuntimeError("right-arm tracking servo did not stop")
        self.raise_if_failed()

    def _run(self) -> None:
        next_tick = time.monotonic()
        try:
            while not self._stop_event.is_set():
                observed = UnitreeG1.get_observation(self.robot)
                observed_right = np.asarray(
                    [float(observed[key]) for _, key in RIGHT_ARM_PAIRS], dtype=np.float64
                )
                with self._lock:
                    expected = self._expected_target.copy()
                    q_error = expected - observed_right
                    if self._correction_enabled:
                        correction_step = np.clip(
                            self.outer_loop_gain * q_error,
                            -self.correction_step_rad,
                            self.correction_step_rad,
                        )
                        self._correction = np.clip(
                            self._correction + correction_step,
                            -self.max_correction_rad,
                            self.max_correction_rad,
                        )
                    command = self._nominal_target + self._correction
                    if self.max_expected_deviation_rad is not None:
                        total_deviation = float(np.max(np.abs(command - self._command_origin)))
                        hard_limit = self.max_expected_deviation_rad + self.max_correction_rad
                        if total_deviation > hard_limit + 1e-9:
                            raise RuntimeError(
                                f"servo command deviation {total_deviation:.4f}rad exceeds {hard_limit:.4f}rad"
                            )
                    if float(np.max(np.abs(q_error))) > self.tracking_error_limit_rad:
                        raise RuntimeError(
                            f"servo tracking error {np.max(np.abs(q_error)):.4f}rad exceeds "
                            f"{self.tracking_error_limit_rad:.4f}rad"
                        )
                    self._latest_observed = observed_right.copy()
                    self._latest_command = command.copy()
                    self._latest_error = q_error.copy()

                action = dict(self._hold_action)
                for value, (_, key) in zip(command, RIGHT_ARM_PAIRS, strict=True):
                    action[key] = float(value)
                arm_q = np.asarray(
                    [float(action[f"{joint.name}.q"]) for joint in G1_29_JointArmIndex],
                    dtype=np.float64,
                )
                arm_tau = self.ik.solve_tau(arm_q)
                tau = np.zeros(29, dtype=np.float32)
                for local_index, joint in enumerate(G1_29_JointArmIndex):
                    if joint.name.startswith("kRight"):
                        tau[joint.value] = arm_tau[local_index]
                self.robot.publish_lowcmd(action, kp=self._kp, kd=self._kd, tau=tau)

                next_tick += self.period
                self._stop_event.wait(max(0.0, next_tick - time.monotonic()))
        except BaseException as exc:
            self._error = exc
            self._stop_event.set()


def parse_vec3(value: str) -> np.ndarray:
    parts = [float(x) for x in value.replace(",", " ").split()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected 3 numbers")
    return np.asarray(parts, dtype=np.float64)


def parse_vec7(value: str) -> np.ndarray:
    parts = [float(x) for x in value.replace(",", " ").split()]
    if len(parts) != 7:
        raise argparse.ArgumentTypeError("expected 7 numbers")
    return np.asarray(parts, dtype=np.float64)


def euler_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def transform_matrix(xyz: np.ndarray, rot: np.ndarray) -> np.ndarray:
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = rot
    tf[:3, 3] = xyz
    return tf


def invert_transform(tf: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = tf[:3, :3].T
    inv[:3, 3] = -(tf[:3, :3].T @ tf[:3, 3])
    return inv


def rotation6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    col0 = rot6d[:3].astype(np.float64)
    col1 = rot6d[3:6].astype(np.float64)
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
    return np.stack([col0, col1, np.cross(col0, col1)], axis=1)


def matrix_to_dataset_pose6d(rot: np.ndarray) -> np.ndarray:
    return np.asarray(
        [rot[0, 0], rot[0, 1], rot[1, 0], rot[1, 1], rot[2, 0], rot[2, 1]],
        dtype=np.float64,
    )


def prepare_batch(item: dict, camera_keys: list[str]) -> dict[str, Any]:
    batch = {key: value.unsqueeze(0) for key, value in item.items() if torch.is_tensor(value)}
    if isinstance(item.get("task"), str):
        batch["task"] = item["task"]
    for cam_key in camera_keys:
        if cam_key in batch and batch[cam_key].dtype == torch.uint8:
            batch[cam_key] = batch[cam_key].float() / 255.0
    return batch


def print_depth_stats(label: str, depth: np.ndarray) -> None:
    values = np.asarray(depth, dtype=np.float64)
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0]
    if positive.size == 0:
        print(f"CAMERA_DIAGNOSTICS: {label}: no positive finite pixels")
        return
    percentiles = np.percentile(positive, [1, 5, 50, 95, 99])
    print(
        f"CAMERA_DIAGNOSTICS: {label}: shape={values.shape} dtype={depth.dtype} "
        f"zero_fraction={np.mean(finite == 0):.4f} "
        f"min={positive.min():.1f} p01={percentiles[0]:.1f} p05={percentiles[1]:.1f} "
        f"p50={percentiles[2]:.1f} p95={percentiles[3]:.1f} "
        f"p99={percentiles[4]:.1f} max={positive.max():.1f}",
        flush=True,
    )


def write_depth_diagnostics(
    output_dir: Path,
    dataset_depth: np.ndarray,
    live_depth: np.ndarray,
    live_color_bgr: np.ndarray,
    live_fisheye_bgr: np.ndarray,
    cv2: Any,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print_depth_stats("dataset_depth", dataset_depth)
    print_depth_stats("live_depth", live_depth)
    for label, depth in (("dataset", dataset_depth), ("live", live_depth)):
        raw = np.clip(depth, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        preview = np.clip(depth, 0, 2000).astype(np.float32) * (255.0 / 2000.0)
        if not cv2.imwrite(str(output_dir / f"{label}_depth.png"), raw):
            raise RuntimeError(f"failed to write {label}_depth.png")
        if not cv2.imwrite(str(output_dir / f"{label}_depth_preview.png"), preview.astype(np.uint8)):
            raise RuntimeError(f"failed to write {label}_depth_preview.png")
    if not cv2.imwrite(str(output_dir / "live_color.jpg"), live_color_bgr):
        raise RuntimeError("failed to write live_color.jpg")
    if not cv2.imwrite(str(output_dir / "live_fisheye.jpg"), live_fisheye_bgr):
        raise RuntimeError("failed to write live_fisheye.jpg")
    print(f"CAMERA_DIAGNOSTICS: wrote {output_dir}", flush=True)


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
    if cfg.type == "act":
        from lerobot.policies.act.processor_act import make_act_pre_post_processors

        preprocessor, postprocessor = make_act_pre_post_processors(
            cfg, dataset_stats=meta_dataset.meta.stats
        )
    elif cfg.type == "pi0":
        from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors

        preprocessor, postprocessor = make_pi0_pre_post_processors(
            cfg, dataset_stats=meta_dataset.meta.stats
        )
    else:
        raise ValueError(f"unsupported policy type for live G1 execution: {cfg.type}")
    print(f"POLICY: loaded type={cfg.type} from {args.policy_path}", flush=True)
    return policy, preprocessor, postprocessor


def fk_tcp(ik: G1PikaArmIK, q_pin: np.ndarray, frame_id: int) -> tuple[np.ndarray, np.ndarray]:
    pin = ik._pin
    data = ik.reduced_robot.model.createData()
    pin.forwardKinematics(ik.reduced_robot.model, data, q_pin)
    pin.updateFramePlacements(ik.reduced_robot.model, data)
    frame = data.oMf[frame_id]
    return frame.translation.copy(), frame.rotation.copy()


def neutral_tcp_tf(ik: G1PikaArmIK, frame_id: int) -> np.ndarray:
    pos, rot = fk_tcp(ik, np.zeros(ik.reduced_robot.model.nq), frame_id)
    return transform_matrix(pos, rot)


def right_tcp_mujoco_pose_from_g1(
    ik: G1PikaArmIK,
    left_g1: np.ndarray,
    right_g1: np.ndarray,
    unitree_to_mujoco_right_tf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q_g1 = np.concatenate([left_g1, right_g1])
    q_pin = q_g1[ik._arm_reorder_g1_to_pin]
    pos, rot = fk_tcp(ik, q_pin, ik.R_hand_id)
    tcp_tf = unitree_to_mujoco_right_tf @ transform_matrix(pos, rot)
    return tcp_tf[:3, 3].copy(), tcp_tf[:3, :3].copy()


def right_tcp_mujoco_from_g1(
    ik: G1PikaArmIK,
    left_g1: np.ndarray,
    right_g1: np.ndarray,
    unitree_to_mujoco_right_tf: np.ndarray,
) -> np.ndarray:
    pos, _ = right_tcp_mujoco_pose_from_g1(ik, left_g1, right_g1, unitree_to_mujoco_right_tf)
    return pos


def live_observation_state(
    observed: dict,
    gripper_open_width: float,
    gripper_open_angle: float,
) -> np.ndarray:
    gripper_angle = float(observed.get("right_gripper.pos", 0.0))
    gripper_width = float(np.clip(gripper_angle / gripper_open_angle, 0.0, 1.0) * gripper_open_width)
    return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, gripper_width])


def replace_batch_state(batch: dict[str, torch.Tensor], state: np.ndarray) -> None:
    current = batch["observation.state"]
    value = torch.as_tensor(state, dtype=current.dtype, device=current.device).unsqueeze(0)
    if tuple(value.shape) != tuple(current.shape):
        raise ValueError(f"live state shape {tuple(value.shape)} does not match batch state {tuple(current.shape)}")
    batch["observation.state"] = value


def parse_opencv_device(value: str) -> int | str:
    return int(value) if value.isdigit() else value


class LiveOpenCVBatchCameras:
    def __init__(self, specs: dict[str, int | str], width: int, height: int, fps: float):
        import cv2

        self._cv2 = cv2
        self._captures: dict[str, Any] = {}
        self._width = width
        self._height = height
        for key, device in specs.items():
            cap = cv2.VideoCapture(device)
            if not cap.isOpened():
                raise RuntimeError(f"failed to open camera {device!r} for {key}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
            cap.set(cv2.CAP_PROP_FPS, float(fps))
            self._captures[key] = cap

    def replace(self, batch: dict[str, torch.Tensor]) -> None:
        for key, cap in self._captures.items():
            if key not in batch:
                raise KeyError(f"dataset batch has no camera key {key!r}")
            ok, bgr = cap.read()
            if not ok or bgr is None:
                raise RuntimeError(f"failed to read live camera for {key}")

            current = batch[key]
            _, channels, height, width = current.shape
            if channels != 3:
                raise ValueError(f"{key} expected 3 channels, got batch shape {tuple(current.shape)}")
            if (height, width) != bgr.shape[:2]:
                bgr = self._cv2.resize(bgr, (width, height), interpolation=self._cv2.INTER_AREA)
            rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
            chw = np.transpose(rgb, (2, 0, 1))
            value = torch.as_tensor(chw, device=current.device)
            if current.is_floating_point():
                value = value.to(dtype=current.dtype) / 255.0
            else:
                value = value.to(dtype=current.dtype)
            batch[key] = value.unsqueeze(0)

    def close(self) -> None:
        for cap in self._captures.values():
            cap.release()


class LivePikaZMQCameras:
    def __init__(
        self,
        server: str,
        port: int,
        timeout_ms: int,
        retries: int,
        replace_rgb: bool,
        replace_depth: bool,
        replace_fisheye: bool,
    ):
        import cv2
        import zmq

        if timeout_ms <= 0:
            raise ValueError("live PIKA camera timeout must be positive")
        if retries <= 0:
            raise ValueError("live PIKA camera retries must be positive")
        self._cv2 = cv2
        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self._endpoint = f"tcp://{server}:{port}"
        self._timeout_ms = timeout_ms
        self._retries = retries
        self._sock = self._open_socket()
        self._replace_rgb_enabled = replace_rgb
        self._replace_depth_enabled = replace_depth
        self._replace_fisheye = replace_fisheye

    def _open_socket(self):
        sock = self._ctx.socket(self._zmq.REQ)
        sock.setsockopt(self._zmq.RCVTIMEO, self._timeout_ms)
        sock.setsockopt(self._zmq.SNDTIMEO, self._timeout_ms)
        sock.connect(self._endpoint)
        return sock

    def _reset_socket(self) -> None:
        self._sock.close(linger=0)
        self._sock = self._open_socket()

    def _request_frames(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        parts = None
        for attempt in range(1, self._retries + 1):
            try:
                self._sock.send_string("frames")
                parts = self._sock.recv_multipart()
                break
            except self._zmq.Again as exc:
                self._reset_socket()
                if attempt == self._retries:
                    raise RuntimeError(
                        f"PIKA camera server {self._endpoint} did not respond after "
                        f"{self._retries} attempts ({self._timeout_ms} ms each)"
                    ) from exc
        assert parts is not None
        meta = json.loads(parts[0].decode("utf-8"))
        if "error" in meta:
            raise RuntimeError(meta["error"])
        if len(parts) != 4:
            raise RuntimeError(f"expected 4 frame parts, got {len(parts)}")
        color = self._decode_image(parts[1], self._cv2.IMREAD_COLOR)
        depth = self._decode_image(parts[2], self._cv2.IMREAD_UNCHANGED)
        fisheye = self._decode_image(parts[3], self._cv2.IMREAD_COLOR)
        return color, depth, fisheye

    def preflight(self) -> None:
        color, depth, fisheye = self._request_frames()
        print(
            "CAMERA_PREFLIGHT: OK "
            f"color={color.shape} depth={depth.shape} fisheye={fisheye.shape}"
        )

    def _decode_image(self, payload: bytes, flags: int) -> np.ndarray:
        image = self._cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), flags)
        if image is None:
            raise RuntimeError("failed to decode live camera frame")
        return image

    def _replace_rgb(self, batch: dict[str, torch.Tensor], key: str, bgr: np.ndarray) -> None:
        current = batch[key]
        _, channels, height, width = current.shape
        if channels != 3:
            raise ValueError(f"{key} expected 3 channels, got batch shape {tuple(current.shape)}")
        if (height, width) != bgr.shape[:2]:
            bgr = self._cv2.resize(bgr, (width, height), interpolation=self._cv2.INTER_AREA)
        rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
        chw = np.transpose(rgb, (2, 0, 1))
        value = torch.as_tensor(chw, device=current.device)
        if current.is_floating_point():
            value = value.to(dtype=current.dtype) / 255.0
        else:
            value = value.to(dtype=current.dtype)
        batch[key] = value.unsqueeze(0)

    def _replace_depth(self, batch: dict[str, torch.Tensor], key: str, depth: np.ndarray) -> None:
        current = batch[key]
        _, channels, height, width = current.shape
        if channels != 1:
            raise ValueError(f"{key} expected 1 channel, got batch shape {tuple(current.shape)}")
        if (height, width) != depth.shape[:2]:
            depth = self._cv2.resize(depth, (width, height), interpolation=self._cv2.INTER_NEAREST)
        value = torch.as_tensor(depth[None, :, :], device=current.device)
        batch[key] = value.to(dtype=current.dtype).unsqueeze(0)

    def replace(self, batch: dict[str, torch.Tensor]) -> None:
        d405_color_bgr, d405_depth_u16, fisheye_bgr = self._request_frames()
        if self._replace_rgb_enabled:
            self._replace_rgb(batch, "observation.images.pikaDepthCamera", d405_color_bgr)
        if self._replace_depth_enabled:
            self._replace_depth(batch, "observation.depths.pikaDepthCamera", d405_depth_u16)
        if self._replace_fisheye:
            self._replace_rgb(batch, "observation.images.pikaFisheyeCamera", fisheye_bgr)

    def close(self) -> None:
        self._sock.close(linger=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="data")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--policy-path",
        type=Path,
    )
    parser.add_argument("--robot-ip", default="192.168.123.164")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument(
        "--action-source",
        choices=("policy", "teacher", "trajectory"),
        default="policy",
        help="Use policy predictions, dataset actions, or a pre-retargeted joint trajectory.",
    )
    parser.add_argument(
        "--trajectory-csv",
        type=Path,
        help="CSV containing the verified kRight*.q trajectory for --action-source trajectory.",
    )
    parser.add_argument(
        "--trajectory-start-max-move-rad",
        type=float,
        default=0.5,
        help="Refuse the initial move to CSV row zero if any right joint exceeds this distance.",
    )
    parser.add_argument(
        "--trajectory-max-deviation-rad",
        type=float,
        default=0.5,
        help="Refuse the full CSV if any right joint exceeds this distance from the settled start.",
    )
    parser.add_argument("--translation-scale", type=float, default=1.0)
    parser.add_argument(
        "--use-connected-fk-start",
        action="store_true",
        help="Reset the right TCP target from the robot state read after this process connects.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--policy-action-steps",
        type=int,
        default=1,
        help="Number of predicted ACT actions to consume before recomputing from a new observation.",
    )
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--right-work-start", type=parse_vec3, default=parse_vec3("0.15 -0.20 0.90"))
    parser.add_argument("--left-work-start", type=parse_vec3, default=parse_vec3("0.35 0.20 0.85"))
    parser.add_argument("--right-work-start-euler", type=parse_vec3, default=parse_vec3("0 0.18 0.78"))
    parser.add_argument("--left-work-start-euler", type=parse_vec3, default=parse_vec3("0 0 0"))
    parser.add_argument("--orientation-weight", type=float, default=0.5)
    parser.add_argument("--gripper-open-width", type=float, default=0.098)
    parser.add_argument("--gripper-open-angle", type=float, default=DEFAULT_GRIPPER_OPEN_ANGLE)
    parser.add_argument("--max-joint-step", type=float, default=0.015)
    parser.add_argument("--max-joint-deviation", type=float, default=0.08)
    parser.add_argument(
        "--no-joint-deviation-limit",
        action="store_true",
        help="Disable clipping relative to the rollout's initial joint pose.",
    )
    parser.add_argument("--max-gripper-step", type=float, default=0.04)
    parser.add_argument("--no-gripper", action="store_true", help="Do not connect or command the PIKA gripper.")
    parser.add_argument("--ik-smooth-weight", type=float, default=0.1)
    parser.add_argument("--ik-regularization-weight", type=float, default=0.0)
    parser.add_argument("--dry-run-left-q", type=parse_vec7)
    parser.add_argument("--dry-run-right-q", type=parse_vec7)
    parser.add_argument("--dry-run-gripper", type=float, default=0.0)
    parser.add_argument("--print-observed", action="store_true")
    parser.add_argument("--print-tcp", action="store_true")
    parser.add_argument(
        "--print-action",
        action="store_true",
        help="Print the raw policy action and converted gripper target.",
    )
    parser.add_argument("--observe-delay", type=float, default=0.08)
    parser.add_argument(
        "--live-state",
        action="store_true",
        help="Replace observation.state with current real G1 FK + right gripper; images still come from the dataset.",
    )
    parser.add_argument(
        "--live-depth-rgb-device",
        default=None,
        help="OpenCV device/path for observation.images.pikaDepthCamera. Depth map remains from the dataset.",
    )
    parser.add_argument(
        "--live-fisheye-device",
        default=None,
        help="OpenCV device/path for observation.images.pikaFisheyeCamera.",
    )
    parser.add_argument("--live-camera-width", type=int, default=640)
    parser.add_argument("--live-camera-height", type=int, default=480)
    parser.add_argument(
        "--live-pika-camera-server",
        default=None,
        help="G1 host/IP running serve_pika_live_cameras.py. Replaces D405 RGB/depth and DECXIN fisheye.",
    )
    parser.add_argument("--live-pika-camera-port", type=int, default=5562)
    parser.add_argument("--live-pika-camera-timeout-ms", type=int, default=1000)
    parser.add_argument("--live-pika-camera-retries", type=int, default=3)
    parser.add_argument(
        "--live-pika-camera-no-rgb",
        action="store_true",
        help="Keep observation.images.pikaDepthCamera from the dataset.",
    )
    parser.add_argument(
        "--live-pika-camera-no-depth",
        action="store_true",
        help="Keep observation.depths.pikaDepthCamera from the dataset.",
    )
    parser.add_argument(
        "--live-pika-camera-no-fisheye",
        action="store_true",
        help="Replace only D405 RGB/depth; keep observation.images.pikaFisheyeCamera from the dataset.",
    )
    parser.add_argument(
        "--camera-diagnostics-dir",
        type=Path,
        help="Write one dataset/live camera comparison and exit without policy inference.",
    )
    parser.add_argument(
        "--hold-after-run",
        action="store_true",
        help=(
            "After a successful real rollout, keep publishing the final command instead of disconnecting. "
            "Support the arm before pressing Ctrl-C; disconnect then makes the robot passive."
        ),
    )
    parser.add_argument("--hold-rate-hz", type=float, default=50.0)
    parser.add_argument(
        "--tracking-servo",
        action="store_true",
        help="Use the validated gravity-compensated 50 Hz right-arm tracking servo.",
    )
    parser.add_argument(
        "--confirm-control",
        action="store_true",
        help="Wait for Enter immediately before starting the real tracking servo.",
    )
    parser.add_argument("--servo-rate-hz", type=float, default=50.0)
    parser.add_argument("--servo-settle-sec", type=float, default=3.0)
    parser.add_argument("--servo-outer-loop-gain", type=float, default=0.005)
    parser.add_argument("--servo-correction-step-rad", type=float, default=0.0001)
    parser.add_argument("--servo-max-correction-rad", type=float, default=0.04)
    parser.add_argument("--servo-tracking-error-limit-rad", type=float, default=0.12)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.live_state and args.dry_run:
        raise ValueError("--live-state requires a real robot connection; remove --dry-run")
    if args.policy_action_steps <= 0:
        raise ValueError("--policy-action-steps must be positive")
    if args.hold_rate_hz <= 0.0:
        raise ValueError("--hold-rate-hz must be positive")
    if args.servo_rate_hz <= 0.0 or args.servo_settle_sec <= 0.0:
        raise ValueError("--servo-rate-hz and --servo-settle-sec must be positive")
    if args.tracking_servo and args.dry_run:
        raise ValueError("--tracking-servo requires a real robot connection")
    if args.translation_scale <= 0.0:
        raise ValueError("--translation-scale must be positive")
    if args.steps < 0:
        raise ValueError("--steps must be non-negative; use 0 to run through the end of the episode")
    if args.max_joint_step <= 0.0:
        raise ValueError("--max-joint-step must be positive")
    if args.action_source == "trajectory" and args.trajectory_csv is None:
        raise ValueError("--trajectory-csv is required for --action-source trajectory")
    if args.action_source == "policy" and args.policy_path is None:
        raise ValueError("--policy-path is required for --action-source policy")
    if not args.dry_run and args.action_source == "policy":
        required_real_policy_flags = {
            "--use-connected-fk-start": args.use_connected_fk_start,
            "--live-state": args.live_state,
            "--live-pika-camera-server": args.live_pika_camera_server is not None,
            "--tracking-servo": args.tracking_servo,
            "--confirm-control": args.confirm_control,
            "--hold-after-run": args.hold_after_run,
        }
        missing = [name for name, enabled in required_real_policy_flags.items() if not enabled]
        if missing:
            raise ValueError(
                "real policy control requires these safety options: " + ", ".join(missing)
            )
        if args.live_pika_camera_no_rgb or args.live_pika_camera_no_fisheye:
            raise ValueError("real policy control requires live PIKA RGB and fisheye streams")
        if args.max_joint_step > 0.010:
            raise ValueError("real policy --max-joint-step must be <= 0.010rad")
        if not args.no_joint_deviation_limit and args.max_joint_deviation > 0.10:
            raise ValueError("real policy --max-joint-deviation must be <= 0.10rad")
    if args.action_source != "trajectory" and args.trajectory_csv is not None:
        raise ValueError("--trajectory-csv requires --action-source trajectory")
    if args.trajectory_start_max_move_rad <= 0.0 or args.trajectory_max_deviation_rad <= 0.0:
        raise ValueError("trajectory movement limits must be positive")
    if (args.dry_run_left_q is None) != (args.dry_run_right_q is None):
        raise ValueError("--dry-run-left-q and --dry-run-right-q must be provided together")
    if (
        args.live_pika_camera_server is not None
        and args.live_pika_camera_no_rgb
        and args.live_pika_camera_no_depth
        and args.live_pika_camera_no_fisheye
    ):
        raise ValueError("at least one live PIKA camera stream must be enabled")
    live_camera_specs = {}
    if args.live_depth_rgb_device is not None:
        live_camera_specs["observation.images.pikaDepthCamera"] = parse_opencv_device(args.live_depth_rgb_device)
    if args.live_fisheye_device is not None:
        live_camera_specs["observation.images.pikaFisheyeCamera"] = parse_opencv_device(args.live_fisheye_device)

    dataset = None
    trajectory = trajectory_indices = None
    if args.action_source == "trajectory":
        trajectory, trajectory_indices = load_right_joint_trajectory(args.trajectory_csv)
        print(f"ACTION_SOURCE: replaying verified right-arm trajectory: {args.trajectory_csv}")
    else:
        dataset = LeRobotDataset(
            args.repo_id,
            root=args.dataset_root,
            episodes=[args.episode],
            return_uint8=True,
            video_backend=args.video_backend,
        )
    policy = preprocessor = postprocessor = None
    if args.action_source == "policy":
        meta_dataset = LeRobotDataset(
            args.repo_id,
            root=args.dataset_root,
            return_uint8=True,
            video_backend=args.video_backend,
        )
    elif args.action_source == "teacher":
        print("ACTION_SOURCE: replaying recorded dataset actions (policy disabled)")
    ik = G1PikaArmIK()

    unitree_neutral_left_tf = neutral_tcp_tf(ik, ik.L_hand_id)
    unitree_neutral_right_tf = neutral_tcp_tf(ik, ik.R_hand_id)
    mujoco_neutral_left_tf = transform_matrix(np.array([0.3798, 0.1486, 0.8882]), np.eye(3))
    mujoco_neutral_right_tf = transform_matrix(np.array([0.3798, -0.1486, 0.8882]), np.eye(3))
    unitree_to_mujoco_right_tf = mujoco_neutral_right_tf @ invert_transform(unitree_neutral_right_tf)
    mujoco_to_unitree_left_tf = invert_transform(mujoco_neutral_left_tf @ invert_transform(unitree_neutral_left_tf))
    mujoco_to_unitree_right_tf = invert_transform(mujoco_neutral_right_tf @ invert_transform(unitree_neutral_right_tf))

    left_target = args.left_work_start.copy()
    right_target = args.right_work_start.copy()
    left_rot = euler_to_matrix(*args.left_work_start_euler)
    right_rot = euler_to_matrix(*args.right_work_start_euler)
    align_rot = ik.tracker_to_tcp_rotation("right")
    rotation_weight = np.array([0.0, 0.0, 0.0, args.orientation_weight, args.orientation_weight, args.orientation_weight])

    robot = None
    tracking_servo = None
    if not args.dry_run:
        if args.no_gripper:
            from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config

            robot = UnitreeG1(UnitreeG1Config(is_simulation=False, robot_ip=args.robot_ip))
        else:
            from lerobot.robots.unitree_g1_pika.config_unitree_g1_pika import UnitreeG1PikaConfig
            from lerobot.robots.unitree_g1_pika.unitree_g1_pika import UnitreeG1Pika

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
        real_obs = robot.get_observation()
    else:
        real_obs = {}
        print("DRY RUN: no robot command will be sent")

    if args.dry_run and args.dry_run_right_q is not None:
        for value, (_, key) in zip(args.dry_run_left_q, LEFT_ARM_PAIRS, strict=True):
            real_obs[key] = float(value)
        for value, (_, key) in zip(args.dry_run_right_q, RIGHT_ARM_PAIRS, strict=True):
            real_obs[key] = float(value)
        real_obs["right_gripper.pos"] = args.dry_run_gripper

    if not real_obs:
        # Conservative fallback: start from zero only in dry-run mode.
        real_obs = {key: 0.0 for _, key in RIGHT_ARM_PAIRS}
        real_obs["right_gripper.pos"] = 0.0

    hold_action = {key: value for key, value in real_obs.items() if key.endswith(".q")}
    initial_left = np.array([float(real_obs.get(key, 0.0)) for _, key in LEFT_ARM_PAIRS], dtype=np.float64)
    initial_right = np.array([float(real_obs[key]) for _, key in RIGHT_ARM_PAIRS], dtype=np.float64)
    connected_tcp, connected_rot = right_tcp_mujoco_pose_from_g1(
        ik,
        initial_left,
        initial_right,
        unitree_to_mujoco_right_tf,
    )
    if args.use_connected_fk_start:
        right_target = connected_tcp.copy()
        right_rot = connected_rot
        print("right TCP target reset from connected robot state:", np.round(right_target, 4))
    sent_right = initial_right.copy()
    current_gripper = float(real_obs.get("right_gripper.pos", 0.0))
    q_g1_initial = np.concatenate([initial_left, initial_right])
    q_pin = q_g1_initial[ik._arm_reorder_g1_to_pin]
    dt = 1.0 / args.fps
    last_robot_action = None
    trajectory_run = trajectory_run_indices = None
    if trajectory is not None:
        trajectory_steps = len(trajectory) if args.steps == 0 else min(args.steps, len(trajectory))
        if trajectory_steps == 0:
            raise ValueError("no trajectory rows selected")
        trajectory_run = trajectory[:trajectory_steps].copy()
        trajectory_run_indices = trajectory_indices[:trajectory_steps].copy()
        start_move = float(np.max(np.abs(trajectory_run[0] - initial_right)))
        total_deviation = float(np.max(np.abs(trajectory_run - initial_right)))
        print(
            f"trajectory rows: 0..{trajectory_steps - 1} "
            f"(dataset_idx={trajectory_run_indices[0]}..{trajectory_run_indices[-1]})"
        )
        print(f"trajectory start max joint move: {start_move:.4f}rad")
        print(f"trajectory total max deviation : {total_deviation:.4f}rad")

    print("initial right arm:", np.round(initial_right, 4))
    print("initial right_gripper.pos:", round(current_gripper, 4))
    if args.live_state:
        print("LIVE_STATE: policy observation.state is relative identity pose + live right gripper")
        print("LIVE_STATE: camera observations are replaced below when live camera options are enabled")
    live_cameras = None
    if live_camera_specs:
        assert dataset is not None
        live_cameras = LiveOpenCVBatchCameras(
            live_camera_specs, args.live_camera_width, args.live_camera_height, args.fps
        )
        print("LIVE_CAMERA: replacing", ", ".join(sorted(live_camera_specs)))
        if "observation.depths.pikaDepthCamera" in dataset.meta.features:
            print("LIVE_CAMERA: observation.depths.pikaDepthCamera still comes from the dataset")
    if args.live_pika_camera_server is not None:
        assert dataset is not None
        if live_cameras is not None:
            raise ValueError("use either OpenCV live camera args or --live-pika-camera-server, not both")
        live_cameras = LivePikaZMQCameras(
            args.live_pika_camera_server,
            args.live_pika_camera_port,
            args.live_pika_camera_timeout_ms,
            args.live_pika_camera_retries,
            not args.live_pika_camera_no_rgb,
            not args.live_pika_camera_no_depth,
            not args.live_pika_camera_no_fisheye,
        )
        live_streams = []
        if not args.live_pika_camera_no_rgb:
            live_streams.append("D405 RGB")
        if not args.live_pika_camera_no_depth:
            live_streams.append("D405 depth")
        if not args.live_pika_camera_no_fisheye:
            live_streams.append("DECXIN fisheye")
        print(
            f"LIVE_CAMERA: replacing {', '.join(live_streams)} from "
            f"tcp://{args.live_pika_camera_server}:{args.live_pika_camera_port}"
        )
    if args.print_tcp:
        print("initial right_tcp_mujoco:", np.round(connected_tcp, 4))
        print("start target_tcp_mujoco :", np.round(right_target, 4))
    joint_deviation_limit = None if args.no_joint_deviation_limit else args.max_joint_deviation
    print(
        "limits: max_joint_step",
        args.max_joint_step,
        "max_joint_deviation",
        "disabled" if joint_deviation_limit is None else joint_deviation_limit,
    )

    try:
        if isinstance(live_cameras, LivePikaZMQCameras):
            print("CAMERA_PREFLIGHT: checking live cameras before enabling G1 control")
            live_cameras.preflight()
            if args.camera_diagnostics_dir is not None:
                assert dataset is not None
                item = dataset[args.start_index]
                batch = prepare_batch(item, dataset.meta.camera_keys)
                depth_key = "observation.depths.pikaDepthCamera"
                if depth_key not in batch:
                    raise KeyError(f"dataset has no {depth_key}")
                dataset_depth = batch[depth_key].squeeze(0).squeeze(0).detach().cpu().numpy()
                live_color, live_depth, live_fisheye = live_cameras._request_frames()
                write_depth_diagnostics(
                    args.camera_diagnostics_dir,
                    dataset_depth,
                    live_depth,
                    live_color,
                    live_fisheye,
                    live_cameras._cv2,
                )
                return
        if trajectory_run is not None:
            start_move = float(np.max(np.abs(trajectory_run[0] - initial_right)))
            total_deviation = float(np.max(np.abs(trajectory_run - initial_right)))
            if start_move > args.trajectory_start_max_move_rad + 1e-9:
                raise RuntimeError(
                    f"trajectory start move {start_move:.4f}rad exceeds "
                    f"{args.trajectory_start_max_move_rad:.4f}rad"
                )
            if total_deviation > args.trajectory_max_deviation_rad + 1e-9:
                raise RuntimeError(
                    f"trajectory deviation {total_deviation:.4f}rad exceeds "
                    f"{args.trajectory_max_deviation_rad:.4f}rad"
                )
        if args.tracking_servo:
            assert robot is not None
            if args.confirm_control:
                input(
                    "Press Enter to engage the G1 right-arm tracking servo; "
                    "Ctrl-C cancels without sending LowCmd: "
                )
            pre_engage_left = initial_left.copy()
            pre_engage_right = initial_right.copy()
            tracking_servo = RightArmTrackingServo(
                robot,
                ik,
                real_obs,
                rate_hz=args.servo_rate_hz,
                outer_loop_gain=args.servo_outer_loop_gain,
                correction_step_rad=args.servo_correction_step_rad,
                max_correction_rad=args.servo_max_correction_rad,
                tracking_error_limit_rad=args.servo_tracking_error_limit_rad,
                max_expected_deviation_rad=(
                    args.trajectory_max_deviation_rad
                    if args.action_source == "trajectory"
                    else joint_deviation_limit
                ),
            )
            tracking_servo.start()
            print(
                f"TRACKING_SERVO: engaging right arm at {args.servo_rate_hz:g} Hz; "
                f"settling for {args.servo_settle_sec:g}s"
            )
            settled_body_observation = tracking_servo.capture_settled_baseline(args.servo_settle_sec)
            real_obs.update(settled_body_observation)
            initial_left = np.array(
                [float(real_obs.get(key, 0.0)) for _, key in LEFT_ARM_PAIRS], dtype=np.float64
            )
            initial_right = np.array(
                [float(real_obs[key]) for _, key in RIGHT_ARM_PAIRS], dtype=np.float64
            )
            connected_tcp, connected_rot = right_tcp_mujoco_pose_from_g1(
                ik,
                initial_left,
                initial_right,
                unitree_to_mujoco_right_tf,
            )
            if args.use_connected_fk_start:
                right_target = connected_tcp.copy()
                right_rot = connected_rot
            sent_right = initial_right.copy()
            q_g1_initial = np.concatenate([initial_left, initial_right])
            q_pin = q_g1_initial[ik._arm_reorder_g1_to_pin]
            print("TRACKING_SERVO: engage left q shift :", np.round(initial_left - pre_engage_left, 5))
            print("TRACKING_SERVO: engage right q shift:", np.round(initial_right - pre_engage_right, 5))
            print("TRACKING_SERVO: settled right TCP  :", np.round(connected_tcp, 5))

            if trajectory_run is not None:
                settled_start_move = float(np.max(np.abs(trajectory_run[0] - initial_right)))
                settled_total_deviation = float(np.max(np.abs(trajectory_run - initial_right)))
                print(f"TRAJECTORY: settled start move    : {settled_start_move:.4f}rad")
                print(f"TRAJECTORY: settled max deviation : {settled_total_deviation:.4f}rad")
                if settled_start_move > args.trajectory_start_max_move_rad + 1e-9:
                    raise RuntimeError(
                        f"settled trajectory start move {settled_start_move:.4f}rad exceeds "
                        f"{args.trajectory_start_max_move_rad:.4f}rad"
                    )
                if settled_total_deviation > args.trajectory_max_deviation_rad + 1e-9:
                    raise RuntimeError(
                        f"settled trajectory deviation {settled_total_deviation:.4f}rad exceeds "
                        f"{args.trajectory_max_deviation_rad:.4f}rad"
                    )

        if trajectory_run is not None:
            if tracking_servo is None:
                raise RuntimeError("trajectory replay requires --tracking-servo")
            assert robot is not None and trajectory_run_indices is not None

            print(
                "TRAJECTORY_HOME: moving to verified CSV row 0 at "
                f"{args.servo_rate_hz:g} Hz / {args.max_joint_step:.4f}rad max step",
                flush=True,
            )
            interpolate_servo_target(
                tracking_servo,
                initial_right,
                trajectory_run[0],
                rate_hz=args.servo_rate_hz,
                max_step_rad=args.max_joint_step,
                minimum_duration_sec=0.0,
            )
            time.sleep(args.servo_settle_sec)
            tracking_servo.raise_if_failed()
            home_status = tracking_servo.snapshot()
            print(
                f"TRAJECTORY_HOME: arrived q_error={home_status['max_error']:.4f}rad "
                f"correction={np.round(home_status['correction'], 4)}",
                flush=True,
            )

            previous_target = trajectory_run[0].copy()
            for row_index, (target_q, dataset_index) in enumerate(
                zip(trajectory_run, trajectory_run_indices, strict=True)
            ):
                interpolate_servo_target(
                    tracking_servo,
                    previous_target,
                    target_q,
                    rate_hz=args.servo_rate_hz,
                    max_step_rad=args.max_joint_step,
                    minimum_duration_sec=0.0 if row_index == 0 else dt,
                )
                previous_target = target_q.copy()
                tracking_servo.raise_if_failed()

                robot_action = dict(hold_action)
                for value, (_, key) in zip(target_q, RIGHT_ARM_PAIRS, strict=True):
                    robot_action[key] = float(value)
                robot_action["right_gripper.pos"] = current_gripper
                last_robot_action = robot_action

                status = tracking_servo.snapshot()
                print(
                    f"trajectory_row={row_index:04d} dataset_idx={dataset_index:04d} "
                    f"q_error={status['max_error']:.4f}rad "
                    f"correction={np.round(status['correction'], 4)} "
                    f"q={np.round(target_q, 4)}",
                    flush=True,
                )
                if args.print_tcp:
                    cmd_tcp = right_tcp_mujoco_from_g1(
                        ik, initial_left, target_q, unitree_to_mujoco_right_tf
                    )
                    observed_tcp = right_tcp_mujoco_from_g1(
                        ik,
                        initial_left,
                        np.asarray(status["observed"], dtype=np.float64),
                        unitree_to_mujoco_right_tf,
                    )
                    print(
                        f"        cmd_tcp={np.round(cmd_tcp, 4)} "
                        f"obs_tcp={np.round(observed_tcp, 4)} "
                        f"obs-cmd={np.round(observed_tcp - cmd_tcp, 4)}",
                        flush=True,
                    )

        if args.action_source == "policy":
            assert meta_dataset is not None
            print("POLICY: loading after right-arm hold is active", flush=True)
            policy, preprocessor, postprocessor = load_policy(args, meta_dataset)

        remaining_steps = 0 if dataset is None else max(0, len(dataset) - args.start_index)
        run_steps = (
            0
            if dataset is None
            else remaining_steps if args.steps == 0 else min(args.steps, remaining_steps)
        )
        if run_steps == 0 and trajectory_run is None:
            raise ValueError(
                f"no dataset steps selected: start_index={args.start_index}, episode_length={len(dataset)}"
            )
        if dataset is not None:
            print(
                f"run range: dataset_idx={args.start_index}..{args.start_index + run_steps - 1} "
                f"({run_steps} steps)",
                flush=True,
            )

        with torch.no_grad():
            for step in range(run_steps):
                assert dataset is not None
                if tracking_servo is not None:
                    tracking_servo.raise_if_failed()
                dataset_idx = args.start_index + step
                item = dataset[dataset_idx]
                if args.action_source == "teacher":
                    action = item["action"].detach().cpu().numpy().astype(np.float64)
                else:
                    assert policy is not None and preprocessor is not None and postprocessor is not None
                    batch = prepare_batch(item, dataset.meta.camera_keys)
                    if live_cameras is not None:
                        live_cameras.replace(batch)
                    if args.live_state:
                        assert robot is not None
                        live_state = live_observation_state(
                            robot.get_observation(),
                            args.gripper_open_width,
                            args.gripper_open_angle,
                        )
                        replace_batch_state(batch, live_state)
                    obs = preprocessor(batch)
                    action = postprocessor(policy.select_action(obs)).squeeze(0).detach().cpu().numpy().astype(np.float64)

                if action.shape != (10,):
                    raise ValueError(f"expected a 10D action, got shape {action.shape} at dataset index {dataset_idx}")

                raw_delta = args.translation_scale * action[:3]
                raw_rot = rotation6d_to_matrix(action[3:9])
                gripper_width = float(action[9])
                right_target = right_target + right_rot @ align_rot @ raw_delta
                right_rot = right_rot @ (align_rot @ raw_rot @ align_rot.T)

                left_tf = transform_matrix(left_target, left_rot)
                right_tf = transform_matrix(right_target, right_rot)
                left_ik_tf = mujoco_to_unitree_left_tf @ left_tf
                right_ik_tf = mujoco_to_unitree_right_tf @ right_tf
                q_pin, _ = ik.solve_ik(
                    left_ik_tf,
                    right_ik_tf,
                    current_lr_arm_motor_q=q_pin,
                    rotation_weight=rotation_weight,
                    smooth_weight=args.ik_smooth_weight,
                    regularization_weight=args.ik_regularization_weight,
                    use_filter=False,
                )
                q_g1 = np.asarray(q_pin, dtype=np.float64)[ik._arm_reorder_pin_to_g1]
                target_right = q_g1[7:14]
                if not np.all(np.isfinite(target_right)):
                    raise RuntimeError("IK returned a non-finite right-arm target")
                if joint_deviation_limit is None:
                    clipped_target = target_right
                else:
                    clipped_target = np.clip(
                        target_right,
                        initial_right - joint_deviation_limit,
                        initial_right + joint_deviation_limit,
                    )
                sent_right = sent_right + np.clip(
                    clipped_target - sent_right,
                    -args.max_joint_step,
                    args.max_joint_step,
                )

                gripper_target = float(np.clip(gripper_width / args.gripper_open_width, 0.0, 1.0) * args.gripper_open_angle)
                current_gripper = float(
                    current_gripper
                    + np.clip(gripper_target - current_gripper, -args.max_gripper_step, args.max_gripper_step)
                )

                if args.print_action:
                    print(
                        f"        raw_action_gripper_width={gripper_width:.6f} "
                        f"gripper_target_angle={gripper_target:.6f} "
                        f"commanded_gripper_angle={current_gripper:.6f}",
                        flush=True,
                    )

                robot_action = dict(hold_action)
                for value, (_, key) in zip(sent_right, RIGHT_ARM_PAIRS, strict=True):
                    robot_action[key] = float(value)
                robot_action["right_gripper.pos"] = current_gripper

                print(
                    f"step={step:04d} dataset_idx={dataset_idx:04d} "
                    f"right_delta_norm={np.linalg.norm(sent_right - initial_right):.4f} "
                    f"gripper={current_gripper:.4f} q={np.round(sent_right, 4)}"
                )
                if args.print_tcp:
                    cmd_tcp = right_tcp_mujoco_from_g1(ik, initial_left, sent_right, unitree_to_mujoco_right_tf)
                    print(
                        f"        target_tcp={np.round(right_target, 4)} "
                        f"cmd_tcp={np.round(cmd_tcp, 4)} "
                        f"cmd-target={np.round(cmd_tcp - right_target, 4)}"
                    )
                if robot is not None:
                    if tracking_servo is not None:
                        tracking_servo.set_target(sent_right)
                        if getattr(robot, "right_gripper", None) is not None:
                            robot.right_gripper.set_angle(current_gripper)
                    else:
                        robot.send_action(robot_action)
                    last_robot_action = robot_action
                    if args.print_observed or args.print_tcp:
                        time.sleep(args.observe_delay)
                        observed = robot.get_observation()
                        observed_right = np.array(
                            [float(observed[key]) for _, key in RIGHT_ARM_PAIRS],
                            dtype=np.float64,
                        )
                        observed_gripper = float(observed.get("right_gripper.pos", float("nan")))
                        if args.print_observed:
                            print(
                                f"        observed_err={np.linalg.norm(observed_right - sent_right):.4f} "
                                f"observed_gripper={observed_gripper:.4f} "
                                f"observed_q={np.round(observed_right, 4)}"
                            )
                            if tracking_servo is not None:
                                servo_status = tracking_servo.snapshot()
                                print(
                                    f"        servo_q_error={servo_status['max_error']:.4f} "
                                    f"servo_correction={np.round(servo_status['correction'], 4)}"
                                )
                        if args.print_tcp:
                            observed_left = np.array(
                                [float(observed.get(key, 0.0)) for _, key in LEFT_ARM_PAIRS],
                                dtype=np.float64,
                            )
                            obs_tcp = right_tcp_mujoco_from_g1(
                                ik, observed_left, observed_right, unitree_to_mujoco_right_tf
                            )
                            print(
                                f"        obs_tcp={np.round(obs_tcp, 4)} "
                                f"obs-cmd={np.round(obs_tcp - cmd_tcp, 4)}"
                            )
                time.sleep(dt)

        if args.hold_after_run and robot is not None and last_robot_action is not None:
            hold_dt = 1.0 / args.hold_rate_hz
            print(
                f"POST_RUN_HOLD: holding final command at {args.hold_rate_hz:g} Hz. "
                "Support the G1 right arm before pressing Ctrl-C.",
                flush=True,
            )
            try:
                next_hold_report = 0.0
                while True:
                    if tracking_servo is not None:
                        tracking_servo.raise_if_failed()
                        now = time.monotonic()
                        if now >= next_hold_report:
                            status = tracking_servo.snapshot()
                            print(
                                "POST_RUN_HOLD: "
                                f"q_error={status['max_error']:.4f}rad "
                                f"correction={np.round(status['correction'], 4)}",
                                flush=True,
                            )
                            next_hold_report = now + 1.0
                    else:
                        robot.send_action(last_robot_action)
                    time.sleep(hold_dt)
            except KeyboardInterrupt:
                print("POST_RUN_HOLD: operator requested passive disconnect", flush=True)
    finally:
        if live_cameras is not None:
            live_cameras.close()
        if robot is not None:
            if tracking_servo is not None:
                try:
                    tracking_servo.stop()
                finally:
                    robot.disconnect()
            else:
                if last_robot_action is not None:
                    robot.send_action(last_robot_action)
                else:
                    # No command was sent (for example, Ctrl-C at the confirmation prompt).
                    robot.config.is_simulation = True
                robot.disconnect()


if __name__ == "__main__":
    main()
