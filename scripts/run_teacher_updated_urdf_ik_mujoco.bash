#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIKA_ROS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd "$PIKA_ROS_DIR/../.." && pwd)}"
DATASET_ROOT="${DATASET_ROOT:-$WORKSPACE_DIR/datasets/replace_tape_g1_pika_relative_h1_v2}"
CONDA_ENV="${CONDA_ENV:-pika_g1_ik}"
EPISODE="${EPISODE:-10}"
START_INDEX="${START_INDEX:-0}"
STEPS="${STEPS:-all}"
FPS="${FPS:-30}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKSPACE_DIR/videos/updated_urdf_ik_episode${EPISODE}}"
START_POSE_CONFIG="${START_POSE_CONFIG:-$DATASET_ROOT/meta/retarget_start_pose.json}"
if [[ ! -f "$START_POSE_CONFIG" ]]; then
  START_POSE_CONFIG="$PIKA_ROS_DIR/config/g1_pika_retarget_start_pose.json"
fi
MOTION_SCALE_XYZ="${MOTION_SCALE_XYZ:-1 1 1}"
ORIENTATION_WEIGHT="${ORIENTATION_WEIGHT:-1.0}"
UNITREE_IK_METHOD="${UNITREE_IK_METHOD:-scipy-right}"
IK_POSITION_TOLERANCE_M="${IK_POSITION_TOLERANCE_M:-0.005}"
IK_ROTATION_TOLERANCE_RAD="${IK_ROTATION_TOLERANCE_RAD:-0.05}"
FAIL_ON_IK_ERROR="${FAIL_ON_IK_ERROR:-1}"
SKIP_VIDEO="${SKIP_VIDEO:-0}"
CAMERA_LAYOUT="${CAMERA_LAYOUT:-four}"
RENDER_START_STEP="${RENDER_START_STEP:-0}"
RENDER_END_STEP="${RENDER_END_STEP:-1000000000}"

if [[ "$STEPS" == "all" ]]; then
  STEPS_ARG=1000000000
else
  STEPS_ARG="$STEPS"
fi

FAIL_ON_IK_ERROR_ARGS=()
if [[ "$FAIL_ON_IK_ERROR" == "1" ]]; then
  FAIL_ON_IK_ERROR_ARGS+=(--fail-on-ik-error)
fi

SKIP_VIDEO_ARGS=()
if [[ "$SKIP_VIDEO" == "1" ]]; then
  SKIP_VIDEO_ARGS+=(--skip-video)
fi

echo "DATASET_ROOT: $DATASET_ROOT"
echo "EPISODE     : $EPISODE"
echo "OUTPUT_DIR  : $OUTPUT_DIR"
echo "START_POSE  : $START_POSE_CONFIG"

cd "$WORKSPACE_DIR"
env -u PYTHONPATH -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH \
  PYTHONPATH="$WORKSPACE_DIR/src/lerobot/src" \
  HF_HOME="$WORKSPACE_DIR/.cache/huggingface" \
  XDG_CACHE_HOME="$WORKSPACE_DIR/.cache" \
  HF_HUB_OFFLINE=1 \
  MUJOCO_GL=egl \
  conda run --no-capture-output -n "$CONDA_ENV" \
    python "$SCRIPT_DIR/g1_pika_mujoco_retarget.py" \
      --dataset-root "$DATASET_ROOT" \
      --source teacher \
      --episode "$EPISODE" \
      --start-index "$START_INDEX" \
      --steps "$STEPS_ARG" \
      --single-arm right \
      --ik-backend unitree \
      --unitree-ik-method "$UNITREE_IK_METHOD" \
      --motion-scale-xyz "$MOTION_SCALE_XYZ" \
      --action-reference relative-delta \
      --relative-delta-frame local \
      --start-pose-config "$START_POSE_CONFIG" \
      --use-initial-q-as-work-start \
      --left-pika-mount-euler "0 3.14159265 0" \
      --pika-euler "0 0 0" \
      --right-pika-mount-euler "1.5708 0 0" \
      --right-pika-euler "1.57 1.5708 0" \
      --right-wrist-to-pika-euler "0 0 0" \
      --use-orientation \
      --orientation-weight "$ORIENTATION_WEIGHT" \
      --unitree-ik-smooth-weight 0.1 \
      --unitree-ik-regularization-weight 0.002 \
      --no-unitree-ik-filter \
      --ik-position-tolerance-m "$IK_POSITION_TOLERANCE_M" \
      --ik-rotation-tolerance-rad "$IK_ROTATION_TOLERANCE_RAD" \
      "${FAIL_ON_IK_ERROR_ARGS[@]}" \
      "${SKIP_VIDEO_ARGS[@]}" \
      --fps "$FPS" \
      --render-every 1 \
      --render-start-step "$RENDER_START_STEP" \
      --render-end-step "$RENDER_END_STEP" \
      --show-trajectories \
      --camera-layout "$CAMERA_LAYOUT" \
      --output-dir "$OUTPUT_DIR"
