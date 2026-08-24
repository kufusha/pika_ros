#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIKA_ROS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd "$PIKA_ROS_DIR/../.." && pwd)}"
DATASET_ROOT="${DATASET_ROOT:-$WORKSPACE_DIR/datasets/replace_tape_g1_pika_relative_h1_v2}"
POLICY_PATH="${POLICY_PATH:-$WORKSPACE_DIR/models/replace_tape_g1_pika_relative_h1_v2_act/pretrained_model}"
CONDA_ENV="${CONDA_ENV:-pika_g1_ik}"
EPISODE="${EPISODE:-10}"
START_INDEX="${START_INDEX:-0}"
STEPS="${STEPS:-all}"
FPS="${FPS:-30}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKSPACE_DIR/videos/policy_updated_urdf_episode${EPISODE}}"

INITIAL_LEFT_Q="${INITIAL_LEFT_Q:-0.1092 0.0023 1.2527 1.2047 0.8746 0.1338 -0.36}"
INITIAL_RIGHT_Q="${INITIAL_RIGHT_Q:-0.73653047 -0.80950896 0.38782465 0.08547420 0.47622498 -0.84839036 0.42185469}"
ORIENTATION_WEIGHT="${ORIENTATION_WEIGHT:-1.0}"
UNITREE_IK_METHOD="${UNITREE_IK_METHOD:-scipy-right}"
IK_POSITION_TOLERANCE_M="${IK_POSITION_TOLERANCE_M:-0.005}"
IK_ROTATION_TOLERANCE_RAD="${IK_ROTATION_TOLERANCE_RAD:-0.05}"
CAMERA_LAYOUT="${CAMERA_LAYOUT:-four}"
DEVICE="${DEVICE:-cpu}"
POLICY_ACTION_STEPS="${POLICY_ACTION_STEPS:-1}"
FAIL_ON_IK_ERROR="${FAIL_ON_IK_ERROR:-0}"

if [[ "$STEPS" == "all" ]]; then
  STEPS_ARG=1000000000
else
  STEPS_ARG="$STEPS"
fi

if [[ ! -f "$POLICY_PATH/model.safetensors" || ! -f "$POLICY_PATH/config.json" ]]; then
  echo "invalid POLICY_PATH: $POLICY_PATH" >&2
  exit 2
fi

cd "$WORKSPACE_DIR"
echo "POLICY_PATH : $POLICY_PATH"
echo "DATASET_ROOT: $DATASET_ROOT"
echo "EPISODE     : $EPISODE"
echo "OUTPUT_DIR  : $OUTPUT_DIR"

EXTRA_ARGS=()
if [[ "$FAIL_ON_IK_ERROR" != "0" ]]; then
  EXTRA_ARGS+=(--fail-on-ik-error)
fi

env -u PYTHONPATH -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH \
  PYTHONPATH="$WORKSPACE_DIR/src/lerobot/src" \
  HF_HOME="$WORKSPACE_DIR/.cache/huggingface" \
  XDG_CACHE_HOME="$WORKSPACE_DIR/.cache" \
  HF_HUB_OFFLINE=1 \
  MUJOCO_GL=egl \
  conda run --no-capture-output -n "$CONDA_ENV" \
    python "$SCRIPT_DIR/g1_pika_mujoco_retarget.py" \
      --dataset-root "$DATASET_ROOT" \
      --policy-path "$POLICY_PATH" \
      --source policy \
      --episode "$EPISODE" \
      --start-index "$START_INDEX" \
      --steps "$STEPS_ARG" \
      --device "$DEVICE" \
      --policy-action-steps "$POLICY_ACTION_STEPS" \
      --single-arm right \
      --ik-backend unitree \
      --unitree-ik-method "$UNITREE_IK_METHOD" \
      --motion-scale-xyz "1 1 1" \
      --action-reference relative-delta \
      --relative-delta-frame local \
      --initial-left-q "$INITIAL_LEFT_Q" \
      --initial-right-q "$INITIAL_RIGHT_Q" \
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
      --fps "$FPS" \
      --show-trajectories \
      --camera-layout "$CAMERA_LAYOUT" \
      --output-dir "$OUTPUT_DIR" \
      "${EXTRA_ARGS[@]}"
