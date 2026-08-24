#!/usr/bin/env bash
set -euo pipefail

ROBOT_IP="${ROBOT_IP:-192.168.123.164}"
REPO_ID="${REPO_ID:-data}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIKA_ROS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd "$PIKA_ROS_DIR/../.." && pwd)}"
DATASET_ROOT="${DATASET_ROOT:-${HF_DATASET_ROOT:-$WORKSPACE_DIR/datasets}}"
EPISODE="${EPISODE:-0}"
START_INDEX="${START_INDEX:-56}"
STEPS="${STEPS:-20}"
FPS="${FPS:-5}"
ACTION_SOURCE="${ACTION_SOURCE:-policy}"
POLICY_PATH="${POLICY_PATH:-}"
POLICY_ACTION_STEPS="${POLICY_ACTION_STEPS:-1}"
TRAJECTORY_CSV="${TRAJECTORY_CSV:-}"
TRAJECTORY_START_MAX_MOVE_RAD="${TRAJECTORY_START_MAX_MOVE_RAD:-0.5}"
TRAJECTORY_MAX_DEVIATION_RAD="${TRAJECTORY_MAX_DEVIATION_RAD:-0.5}"
TRANSLATION_SCALE="${TRANSLATION_SCALE:-0.1}"
ORIENTATION_WEIGHT="${ORIENTATION_WEIGHT:-0.0}"
MAX_JOINT_STEP="${MAX_JOINT_STEP:-0.002}"
MAX_JOINT_DEVIATION="${MAX_JOINT_DEVIATION:-0.03}"
MAX_GRIPPER_STEP="${MAX_GRIPPER_STEP:-0.005}"
CONTROL_GRIPPER="${CONTROL_GRIPPER:-0}"
IK_SMOOTH_WEIGHT="${IK_SMOOTH_WEIGHT:-0.1}"
IK_REGULARIZATION_WEIGHT="${IK_REGULARIZATION_WEIGHT:-0.0}"
OBSERVE_DELAY="${OBSERVE_DELAY:-0.03}"
PRINT_TCP="${PRINT_TCP:-1}"
PRINT_OBSERVED="${PRINT_OBSERVED:-1}"
LIVE_STATE="${LIVE_STATE:-0}"
LIVE_DEPTH_RGB_DEVICE="${LIVE_DEPTH_RGB_DEVICE:-}"
LIVE_FISHEYE_DEVICE="${LIVE_FISHEYE_DEVICE:-}"
LIVE_CAMERA_WIDTH="${LIVE_CAMERA_WIDTH:-640}"
LIVE_CAMERA_HEIGHT="${LIVE_CAMERA_HEIGHT:-480}"
LIVE_PIKA_CAMERA_SERVER="${LIVE_PIKA_CAMERA_SERVER:-}"
LIVE_PIKA_CAMERA_PORT="${LIVE_PIKA_CAMERA_PORT:-5562}"
LIVE_PIKA_CAMERA_TIMEOUT_MS="${LIVE_PIKA_CAMERA_TIMEOUT_MS:-1000}"
LIVE_PIKA_CAMERA_RETRIES="${LIVE_PIKA_CAMERA_RETRIES:-3}"
LIVE_PIKA_CAMERA_NO_FISHEYE="${LIVE_PIKA_CAMERA_NO_FISHEYE:-0}"
HOLD_AFTER_RUN="${HOLD_AFTER_RUN:-1}"
HOLD_RATE_HZ="${HOLD_RATE_HZ:-50}"
TRACKING_SERVO="${TRACKING_SERVO:-1}"
CONFIRM_CONTROL="${CONFIRM_CONTROL:-1}"
SERVO_RATE_HZ="${SERVO_RATE_HZ:-50}"
SERVO_SETTLE_SEC="${SERVO_SETTLE_SEC:-3}"
SERVO_OUTER_LOOP_GAIN="${SERVO_OUTER_LOOP_GAIN:-0.005}"
SERVO_CORRECTION_STEP_RAD="${SERVO_CORRECTION_STEP_RAD:-0.0001}"
SERVO_MAX_CORRECTION_RAD="${SERVO_MAX_CORRECTION_RAD:-0.04}"
SERVO_TRACKING_ERROR_LIMIT_RAD="${SERVO_TRACKING_ERROR_LIMIT_RAD:-0.12}"

SMOKE_SCRIPT="${SMOKE_SCRIPT:-$SCRIPT_DIR/run_policy_ik_smoke_on_g1_pika.py}"

if [[ ! -f "$SMOKE_SCRIPT" ]]; then
  echo "missing smoke script: $SMOKE_SCRIPT" >&2
  exit 1
fi

cmd=(
  python "$SMOKE_SCRIPT"
  --repo-id "$REPO_ID"
  --dataset-root "$DATASET_ROOT"
  --episode "$EPISODE"
  --start-index "$START_INDEX"
  --steps "$STEPS"
  --fps "$FPS"
  --action-source "$ACTION_SOURCE"
  --translation-scale "$TRANSLATION_SCALE"
  --use-connected-fk-start
  --orientation-weight "$ORIENTATION_WEIGHT"
  --max-joint-step "$MAX_JOINT_STEP"
  --max-joint-deviation "$MAX_JOINT_DEVIATION"
  --max-gripper-step "$MAX_GRIPPER_STEP"
  --ik-smooth-weight "$IK_SMOOTH_WEIGHT"
  --ik-regularization-weight "$IK_REGULARIZATION_WEIGHT"
)

if [[ "$ACTION_SOURCE" == "policy" ]]; then
  if [[ -z "$POLICY_PATH" ]]; then
    echo "POLICY_PATH is required when ACTION_SOURCE=policy" >&2
    exit 2
  fi
  if [[ ! -d "$POLICY_PATH" ]]; then
    echo "policy directory not found: $POLICY_PATH" >&2
    exit 2
  fi
  cmd+=(--policy-path "$POLICY_PATH")
  cmd+=(--policy-action-steps "$POLICY_ACTION_STEPS")
fi

if [[ "$ACTION_SOURCE" == "trajectory" ]]; then
  if [[ -z "$TRAJECTORY_CSV" ]]; then
    echo "TRAJECTORY_CSV is required when ACTION_SOURCE=trajectory" >&2
    exit 2
  fi
  cmd+=(
    --trajectory-csv "$TRAJECTORY_CSV"
    --trajectory-start-max-move-rad "$TRAJECTORY_START_MAX_MOVE_RAD"
    --trajectory-max-deviation-rad "$TRAJECTORY_MAX_DEVIATION_RAD"
  )
fi

if [[ "$PRINT_TCP" != "0" ]]; then
  cmd+=(--print-tcp)
fi
if [[ "$PRINT_OBSERVED" != "0" ]]; then
  cmd+=(--print-observed --observe-delay "$OBSERVE_DELAY")
fi
if [[ "$LIVE_STATE" != "0" ]]; then
  cmd+=(--live-state)
fi
if [[ -n "$LIVE_DEPTH_RGB_DEVICE" ]]; then
  cmd+=(--live-depth-rgb-device "$LIVE_DEPTH_RGB_DEVICE")
fi
if [[ -n "$LIVE_FISHEYE_DEVICE" ]]; then
  cmd+=(--live-fisheye-device "$LIVE_FISHEYE_DEVICE")
fi
if [[ -n "$LIVE_DEPTH_RGB_DEVICE$LIVE_FISHEYE_DEVICE" ]]; then
  cmd+=(--live-camera-width "$LIVE_CAMERA_WIDTH" --live-camera-height "$LIVE_CAMERA_HEIGHT")
fi
if [[ -n "$LIVE_PIKA_CAMERA_SERVER" ]]; then
  cmd+=(
    --live-pika-camera-server "$LIVE_PIKA_CAMERA_SERVER"
    --live-pika-camera-port "$LIVE_PIKA_CAMERA_PORT"
    --live-pika-camera-timeout-ms "$LIVE_PIKA_CAMERA_TIMEOUT_MS"
    --live-pika-camera-retries "$LIVE_PIKA_CAMERA_RETRIES"
  )
  if [[ "$LIVE_PIKA_CAMERA_NO_FISHEYE" != "0" ]]; then
    cmd+=(--live-pika-camera-no-fisheye)
  fi
fi
if [[ "$HOLD_AFTER_RUN" != "0" ]]; then
  cmd+=(--hold-after-run --hold-rate-hz "$HOLD_RATE_HZ")
fi
if [[ "$TRACKING_SERVO" != "0" ]]; then
  cmd+=(
    --tracking-servo
    --servo-rate-hz "$SERVO_RATE_HZ"
    --servo-settle-sec "$SERVO_SETTLE_SEC"
    --servo-outer-loop-gain "$SERVO_OUTER_LOOP_GAIN"
    --servo-correction-step-rad "$SERVO_CORRECTION_STEP_RAD"
    --servo-max-correction-rad "$SERVO_MAX_CORRECTION_RAD"
    --servo-tracking-error-limit-rad "$SERVO_TRACKING_ERROR_LIMIT_RAD"
  )
fi
if [[ "$CONFIRM_CONTROL" != "0" ]]; then
  cmd+=(--confirm-control)
fi
if [[ "$CONTROL_GRIPPER" == "0" ]]; then
  cmd+=(--no-gripper)
fi

exec "${cmd[@]}"
