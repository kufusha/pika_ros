#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIKA_ROS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd "$PIKA_ROS_DIR/../.." && pwd)}"
LEROBOT_DIR="${LEROBOT_DIR:-$WORKSPACE_DIR/src/lerobot}"

if [[ ! -f "$LEROBOT_DIR/pyproject.toml" ]]; then
  echo "LeRobot checkout not found: $LEROBOT_DIR" >&2
  echo "Set LEROBOT_DIR to the LeRobot fork containing unitree_g1_pika." >&2
  exit 2
fi

if ! command -v colcon >/dev/null 2>&1; then
  echo "colcon was not found. Source ROS 2 Humble before running this script." >&2
  exit 2
fi

cd "$WORKSPACE_DIR"
colcon build \
  --base-paths "$PIKA_ROS_DIR/src/g1_pika_description" \
  --packages-select g1_pika_description \
  --symlink-install

cat <<EOF

Build complete. Configure the current shell with:
  source "$WORKSPACE_DIR/install/setup.bash"
  export PYTHONPATH="$LEROBOT_DIR/src:\${PYTHONPATH:-}"
  export G1_PIKA_URDF="$PIKA_ROS_DIR/src/g1_pika_description/urdf/g1_29dof_pika.urdf"

For MuJoCo, also set:
  export G1_MUJOCO_XML=/path/to/g1_29dof_no_hand.xml
EOF
