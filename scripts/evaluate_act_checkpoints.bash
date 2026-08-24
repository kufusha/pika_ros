#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIKA_ROS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd "$PIKA_ROS_DIR/../.." && pwd)}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$WORKSPACE_DIR/models/replace_tape_g1_pika_relative_h1_v2_act/checkpoints}"
LAST_POLICY_PATH="${LAST_POLICY_PATH:-$WORKSPACE_DIR/models/replace_tape_g1_pika_relative_h1_v2_act/pretrained_model}"
DATASET_ROOT="${DATASET_ROOT:-$WORKSPACE_DIR/datasets/replace_tape_g1_pika_relative_h1_v2}"
EPISODES="${EPISODES:-10}"
MAX_STEPS="${MAX_STEPS:-1000000000}"
DEVICE="${DEVICE:-cpu}"
CONDA_ENV="${CONDA_ENV:-pika_g1_ik}"

cd "$WORKSPACE_DIR"

policies=()
if [[ -d "$CHECKPOINT_ROOT" ]]; then
  while IFS= read -r path; do
    policies+=("$path")
  done < <(find "$CHECKPOINT_ROOT" -mindepth 2 -maxdepth 2 -type d -name pretrained_model | sort)
fi
if [[ -f "$LAST_POLICY_PATH/model.safetensors" ]]; then
  policies+=("$LAST_POLICY_PATH")
fi

if (( ${#policies[@]} == 0 )); then
  echo "no pretrained_model directories found" >&2
  echo "CHECKPOINT_ROOT=$CHECKPOINT_ROOT" >&2
  echo "LAST_POLICY_PATH=$LAST_POLICY_PATH" >&2
  exit 2
fi

echo "episodes: $EPISODES"
echo "policy_action_steps: 1"
echo

for policy_path in "${policies[@]}"; do
  if [[ "$policy_path" == "$LAST_POLICY_PATH" ]]; then
    label=last
  else
    label="$(basename "$(dirname "$policy_path")")"
  fi
  echo "===== $label ====="
  env -u PYTHONPATH -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH \
    PYTHONPATH="$WORKSPACE_DIR/src/lerobot/src" \
    HF_HOME="$WORKSPACE_DIR/.cache/huggingface" \
    XDG_CACHE_HOME="$WORKSPACE_DIR/.cache" \
    HF_HUB_OFFLINE=1 \
    conda run --no-capture-output -n "$CONDA_ENV" \
      python "$SCRIPT_DIR/replay_lerobot_policy_eval.py" \
        --policy-path "$policy_path" \
        --dataset-root "$DATASET_ROOT" \
        --episodes "$EPISODES" \
        --max-steps "$MAX_STEPS" \
        --device "$DEVICE" \
        --policy-action-steps 1
  echo
done
