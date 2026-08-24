#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  convert_multi_pika_to_lerobot.bash --dataset-dir DIR --episode-index N --target-dir DIR [options]
  convert_multi_pika_to_lerobot.bash --dataset-dir DIR --all --target-dir DIR [options]

Options:
  --dataset-dir DIR     Directory containing episodeN folders.
  --episode-index N     Episode number, e.g. 13 for episode13.
  --episode-name NAME   Episode folder name. Alternative to --episode-index.
  --all                Process all episodeN folders in numeric order.
  --from N             First episode index when using --all.
  --to N               Last episode index when using --all.
  --target-dir DIR      LeRobot output directory.
  --fps N              LeRobot fps. Default: 15.
  --type NAME          data_tools config type. Default: multi_pika.
  --width N            Image width. Default: 640.
  --height N           Image height. Default: 480.
  --sync               Run data_tools_dataSync before HDF5 conversion.
  --reuse-hdf5         Reuse an existing data.hdf5 in every episode directory.
  --time-diff-limit S  Sync timeDiffLimit when --sync is used. Default: 0.03.
  --no-resample        Do not pass --resample to hdf5_to_lerobot.py.
  --relative-trajectory
                       Convert action to current-TCP-relative future trajectory.
  --relative-action-horizon N
                       Future horizon for --relative-trajectory. Default: 1.

Example:
  ./convert_multi_pika_to_lerobot.bash \
    --dataset-dir "$HOME/agilex/0723_open_pet/data" \
    --episode-index 13 \
    --target-dir "$HOME/agilex/0723_open_pet/lerobot_episode13_15fps" \
    --fps 15 \
    --sync

  ./convert_multi_pika_to_lerobot.bash \
    --dataset-dir "$HOME/agilex/0723_open_pet/data" \
    --all \
    --target-dir "$HOME/agilex/0723_open_pet/lerobot_all_15fps" \
    --fps 15 \
    --sync
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_TO_HDF5_DIR="$SCRIPT_DIR"
LEROBOT_SCRIPT_DIR="$WORKSPACE_DIR/src/pika_ros/src/data_tools/scripts"
LEROBOT_PY="${LEROBOT_PY:-$WORKSPACE_DIR/pika_lerobot/bin/python}"

DATASET_DIR=""
EPISODE_NAME=""
PROCESS_ALL="0"
FROM_INDEX=""
TO_INDEX=""
TARGET_DIR=""
FPS="15"
TYPE="multi_pika"
WIDTH="640"
HEIGHT="480"
RUN_SYNC="0"
REUSE_HDF5="0"
TIME_DIFF_LIMIT="0.03"
RESAMPLE="1"
RELATIVE_TRAJECTORY="0"
RELATIVE_ACTION_HORIZON="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-dir)
      DATASET_DIR="$2"; shift 2 ;;
    --episode-index)
      EPISODE_NAME="episode$2"; shift 2 ;;
    --episode-name)
      EPISODE_NAME="$2"; shift 2 ;;
    --all)
      PROCESS_ALL="1"; shift ;;
    --from)
      FROM_INDEX="$2"; shift 2 ;;
    --to)
      TO_INDEX="$2"; shift 2 ;;
    --target-dir)
      TARGET_DIR="$2"; shift 2 ;;
    --fps)
      FPS="$2"; shift 2 ;;
    --type)
      TYPE="$2"; shift 2 ;;
    --width)
      WIDTH="$2"; shift 2 ;;
    --height)
      HEIGHT="$2"; shift 2 ;;
    --sync)
      RUN_SYNC="1"; shift ;;
    --reuse-hdf5)
      REUSE_HDF5="1"; shift ;;
    --time-diff-limit)
      TIME_DIFF_LIMIT="$2"; shift 2 ;;
    --no-resample)
      RESAMPLE="0"; shift ;;
    --relative-trajectory)
      RELATIVE_TRAJECTORY="1"; shift ;;
    --relative-action-horizon)
      RELATIVE_ACTION_HORIZON="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "$DATASET_DIR" || -z "$TARGET_DIR" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 2
fi

if [[ "$PROCESS_ALL" == "0" && -z "$EPISODE_NAME" ]]; then
  echo "Specify --episode-index N, --episode-name NAME, or --all." >&2
  usage
  exit 2
fi

if [[ "$PROCESS_ALL" == "1" && -n "$EPISODE_NAME" ]]; then
  echo "Use either --all or a single episode, not both." >&2
  usage
  exit 2
fi

if [[ "$RUN_SYNC" == "1" && "$REUSE_HDF5" == "1" ]]; then
  echo "Use either --sync or --reuse-hdf5, not both." >&2
  exit 2
fi

DATASET_DIR="$(realpath -m "$DATASET_DIR")"
TARGET_DIR="$(realpath -m "$TARGET_DIR")"

if [[ ! -x "$LEROBOT_PY" ]]; then
  echo "LeRobot Python not found: $LEROBOT_PY" >&2
  exit 1
fi

mapfile -t EPISODES < <(
  find "$DATASET_DIR" -maxdepth 1 -type d -regextype posix-extended -regex '.*/episode[0-9]+' -printf '%f\n' \
    | sed 's/^episode//' \
    | sort -n \
    | while read -r n; do
        if [[ -n "$FROM_INDEX" && "$n" -lt "$FROM_INDEX" ]]; then
          continue
        fi
        if [[ -n "$TO_INDEX" && "$n" -gt "$TO_INDEX" ]]; then
          continue
        fi
        echo "episode$n"
      done
)

if [[ "$PROCESS_ALL" == "0" ]]; then
  EPISODES=("$EPISODE_NAME")
fi

if [[ "${#EPISODES[@]}" -eq 0 ]]; then
  echo "No episodes matched under: $DATASET_DIR" >&2
  exit 1
fi

for episode in "${EPISODES[@]}"; do
  if [[ ! -d "$DATASET_DIR/$episode" ]]; then
    echo "Episode directory not found: $DATASET_DIR/$episode" >&2
    exit 1
  fi
done

echo "Dataset dir : $DATASET_DIR"
echo "Episodes    : ${EPISODES[*]}"
echo "Target dir  : $TARGET_DIR"
echo "FPS         : $FPS"
echo

if [[ "$RUN_SYNC" == "1" ]]; then
  echo "[1/3] Running data sync..."
  for episode in "${EPISODES[@]}"; do
    EPISODE_INDEX="${episode#episode}"
    if [[ "$EPISODE_INDEX" == "$episode" ]]; then
      echo "--sync requires episode name like episode13." >&2
      exit 1
    fi
    echo "  sync $episode"
    ros2 launch data_tools run_data_sync.launch.py \
      type:="$TYPE" \
      datasetDir:="$DATASET_DIR" \
      episodeIndex:="$EPISODE_INDEX" \
      timeDiffLimit:="$TIME_DIFF_LIMIT"
  done
else
  echo "[1/3] Skipping data sync."
fi

if [[ "$REUSE_HDF5" == "1" ]]; then
  echo "[2/3] Reusing existing HDF5 files."
  for episode in "${EPISODES[@]}"; do
    if [[ ! -f "$DATASET_DIR/$episode/data.hdf5" ]]; then
      echo "Existing HDF5 not found: $DATASET_DIR/$episode/data.hdf5" >&2
      exit 1
    fi
  done
else
  echo "[2/3] Creating HDF5..."
  for episode in "${EPISODES[@]}"; do
    echo "  hdf5 $episode"
    (
      cd "$DATA_TO_HDF5_DIR"
      "$LEROBOT_PY" data_to_hdf5.py \
        --type "$TYPE" \
        --useCameraPointCloud "" \
        --datasetDir "$DATASET_DIR" \
        --episodeName "$episode"
    )
  done
fi

echo "[3/3] Converting HDF5 to LeRobot..."
TMP_HDF5_DIR="$(mktemp -d -p "$WORKSPACE_DIR" .tmp_lerobot_hdf5_XXXXXX)"
cleanup() {
  rm -rf "$TMP_HDF5_DIR"
}
trap cleanup EXIT

for episode in "${EPISODES[@]}"; do
  src="$DATASET_DIR/$episode/data.hdf5"
  if [[ ! -f "$src" ]]; then
    echo "HDF5 not found after conversion: $src" >&2
    exit 1
  fi
  tmp_episode_dir="$TMP_HDF5_DIR/$episode"
  mkdir -p "$tmp_episode_dir"
  ln -s "$src" "$tmp_episode_dir/$episode.hdf5"

  # data_to_hdf5.py stores image/depth paths relative to the episode directory
  # (for example camera/color/...). Keep that layout valid while still giving
  # the HDF5 file a unique episodeN.hdf5 stem for LeRobot metadata.
  for rel in camera localization gripper arm force imu array lidar robotBase lift tf instructions.npy; do
    if [[ -e "$DATASET_DIR/$episode/$rel" ]]; then
      ln -s "$DATASET_DIR/$episode/$rel" "$tmp_episode_dir/$rel"
    fi
  done
done

LEROBOT_ARGS=(
  hdf5_to_lerobot.py
  --type "$TYPE"
  --datasetDir "$TMP_HDF5_DIR"
  --targetDir "$TARGET_DIR"
  --imageWidth "$WIDTH"
  --imageHeight "$HEIGHT"
  --fps "$FPS"
)
if [[ "$RESAMPLE" == "1" ]]; then
  LEROBOT_ARGS+=(--resample)
fi
if [[ "$RELATIVE_TRAJECTORY" == "1" ]]; then
  LEROBOT_ARGS+=(--relative_trajectory --relative_action_horizon "$RELATIVE_ACTION_HORIZON")
fi

(
  cd "$LEROBOT_SCRIPT_DIR"
  "$LEROBOT_PY" "${LEROBOT_ARGS[@]}"
)

echo
echo "Done: $TARGET_DIR"
