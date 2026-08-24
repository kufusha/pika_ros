#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="$HOME/agilex/0723_open_pet/lerobot_0_70_10fps"
OUT_DIR="videos/lerobot_pose_viewer_0723_0_70_check"
FROM=0
TO=70
STEPS=80
FPS=10
FIELD="action"

usage() {
  cat <<'USAGE'
Usage:
  visualize_lerobot_pose_episodes.bash [options]

Options:
  --dataset-root DIR  LeRobot dataset root.
  --output-dir DIR    Output directory. Default: videos/lerobot_pose_viewer_0723_0_70_check
  --from N            First episode. Default: 0
  --to N              Last episode. Default: 70
  --steps N           Steps per episode. Default: 80
  --fps N             Output video fps. Default: 10
  --field NAME        action or observation.state. Default: action

Example:
  ./src/pika_ros/scripts/visualize_lerobot_pose_episodes.bash \
    --dataset-root "$HOME/agilex/0723_open_pet/lerobot_0_70_10fps" \
    --from 0 --to 70 --steps 80
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --output-dir) OUT_DIR="$2"; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    --to) TO="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --field) FIELD="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

cd "$(dirname "$0")/../../.."
export HF_HOME="$PWD/.cache/huggingface"
export XDG_CACHE_HOME="$PWD/.cache"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

mkdir -p "$OUT_DIR"

for ep in $(seq "$FROM" "$TO"); do
  echo "episode ${ep}"
  ./pika_lerobot/bin/python src/pika_ros/scripts/visualize_lerobot_pose_dataset.py \
    --dataset-root "$DATASET_ROOT" \
    --episode "$ep" \
    --steps "$STEPS" \
    --field "$FIELD" \
    --output-dir "$OUT_DIR/episode${ep}" \
    --fps "$FPS"
done

echo "done: $OUT_DIR"
