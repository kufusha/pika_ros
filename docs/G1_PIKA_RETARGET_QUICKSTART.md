# G1 + PIKA リターゲット Quick Start

PIKAで収録した手先姿勢をG1右腕の関節軌道へ変換し、MuJoCo動画とCSVを生成する手順です。

## 変換の流れ

```text
PIKA収録データ
  -> LeRobot相対TCP action
  -> G1基準座標のTCP目標
  -> G1+PIKA URDFによるIK
  -> G1右腕7関節のCSV + MuJoCo確認動画
```

右TCPの軸は次の定義で固定されています。

```text
right_pika_tcp +X = グリッパ上方向
right_pika_tcp +Y = グリッパ開閉方向
right_pika_tcp +Z = グリッパ前方
```

## 1. 初回セットアップ

次の配置を前提とします。

```text
$WORKSPACE/
  src/pika_ros/
  src/lerobot/
  datasets/
  videos/
```

最初に実際のワークスペース絶対パスを設定します。

```bash
export WORKSPACE=/absolute/path/to/pika_ws2
```

```bash
cd "$WORKSPACE/src/pika_ros"

WORKSPACE_DIR="$WORKSPACE" \
  scripts/setup_g1_pika_retarget.bash

source "$WORKSPACE/install/setup.bash"
export PYTHONPATH="$WORKSPACE/src/lerobot/src:${PYTHONPATH:-}"
export G1_PIKA_URDF="$WORKSPACE/src/pika_ros/src/g1_pika_description/urdf/g1_29dof_pika.urdf"
```

MuJoCoのG1 XMLが自動検出されない場合だけ設定します。

```bash
export G1_MUJOCO_XML=/absolute/path/to/g1_29dof_no_hand.xml
```

URDFを確認する場合:

```bash
source "$WORKSPACE/install/setup.bash"
ros2 launch g1_pika_description display.launch.py
```

## 2. PIKA収録データを相対TCP actionへ変換

入力datasetは次の構成を想定します。

```text
/path/to/raw_dataset/
  episode0/data.hdf5
  episode1/data.hdf5
  ...
```

既存の `data.hdf5` を使って全episodeを変換します。

```bash
cd "$WORKSPACE/src/pika_ros"

LEROBOT_PY=/path/to/python \
scripts/convert_multi_pika_to_lerobot.bash \
  --dataset-dir /path/to/raw_dataset \
  --all --from 0 --to 59 \
  --target-dir "$WORKSPACE/datasets/g1_pika_relative" \
  --fps 30 \
  --type single_pika \
  --reuse-hdf5 \
  --relative-trajectory \
  --relative-action-horizon 1
```

`LEROBOT_PY` の例:

```bash
LEROBOT_PY="$WORKSPACE/pika_lerobot/bin/python"
```

生成されるactionは10次元です。

```text
[delta_x, delta_y, delta_z, rotation_6d(6), gripper]
```

## 3. 1 episodeをG1 IKへリターゲット

最初はepisode 10だけを処理します。

```bash
cd "$WORKSPACE/src/pika_ros"

WORKSPACE_DIR="$WORKSPACE" \
DATASET_ROOT="$WORKSPACE/datasets/g1_pika_relative" \
EPISODE=10 \
STEPS=all \
CAMERA_LAYOUT=four \
scripts/run_teacher_updated_urdf_ik_mujoco.bash
```

出力先:

```text
$WORKSPACE/videos/updated_urdf_ik_episode10/
```

生成物:

```text
episode10_index0_g1_pika_retarget_teacher_pose_unitreeik.mp4
episode10_index0_g1_pika_retarget_teacher_pose_unitreeik.csv
g1_pika_scene/g1_pika_retarget.xml
```

CSVには各stepのTCP目標、IK誤差、G1右腕7関節が保存されます。

```text
kRightShoulderPitch.q
kRightShoulderRoll.q
kRightShoulderYaw.q
kRightElbow.q
kRightWristRoll.q
kRightWristPitch.q
kRightWristYaw.q
```

## 4. 動画で確認する項目

- `Target TCP` と `IK TCP` が追従している
- PIKAの上下動がG1でも上下動になっている
- 手首が上下反転していない
- roll方向へ90度ずれていない
- 関節角がstep間で飛んでいない
- position/rotation IK errorが許容値内にある

この確認に失敗したCSVは実機で使用しないでください。

## 5. 複数episodeを一括リターゲット

episode 10、20、30、40、50を処理する例:

```bash
cd "$WORKSPACE/src/pika_ros"

for ep in 10 20 30 40 50; do
  WORKSPACE_DIR="$WORKSPACE" \
  DATASET_ROOT="$WORKSPACE/datasets/g1_pika_relative" \
  EPISODE="$ep" STEPS=all CAMERA_LAYOUT=four \
  scripts/run_teacher_updated_urdf_ik_mujoco.bash
done
```

episode 0から59までを処理する例:

```bash
cd "$WORKSPACE/src/pika_ros"

for ep in $(seq 0 59); do
  WORKSPACE_DIR="$WORKSPACE" \
  DATASET_ROOT="$WORKSPACE/datasets/g1_pika_relative" \
  EPISODE="$ep" STEPS=all CAMERA_LAYOUT=four \
  OUTPUT_DIR="$WORKSPACE/videos/g1_pika_retarget/episode${ep}" \
  scripts/run_teacher_updated_urdf_ik_mujoco.bash
done
```

## 6. リターゲット済みCSVを実機で確認

MuJoCo確認済みCSVを、最初は5stepだけ再生します。

```bash
cd "$WORKSPACE/src/pika_ros"

ACTION_SOURCE=trajectory \
TRAJECTORY_CSV="$WORKSPACE/videos/updated_urdf_ik_episode10/episode10_index0_g1_pika_retarget_teacher_pose_unitreeik.csv" \
STEPS=5 FPS=5 \
MAX_JOINT_STEP=0.002 \
TRAJECTORY_START_MAX_MOVE_RAD=0.60 \
TRAJECTORY_MAX_DEVIATION_RAD=0.60 \
CONTROL_GRIPPER=0 \
HOLD_AFTER_RUN=1 \
scripts/run_policy_ik_from_current_fk.sh
```

5step、20step、全stepの順で増やします。全step指定は `STEPS=0` です。

実機試験中はアームを支持し、`Ctrl-C` 後にpassiveになることを前提にしてください。

## 7. よくあるエラー

`G1_PIKA_URDF was not found`:

```bash
export G1_PIKA_URDF="$WORKSPACE/src/pika_ros/src/g1_pika_description/urdf/g1_29dof_pika.urdf"
```

`g1_29dof_no_hand.xml was not found`:

```bash
export G1_MUJOCO_XML=/absolute/path/to/g1_29dof_no_hand.xml
```

CUDAがない場合、policyのMuJoCo検証では `DEVICE=cpu` を指定します。Teacherの
リターゲットにはpolicy GPUは不要です。

`IK tolerance exceeded` の場合は許容値を無制限に緩めず、失敗stepの目標姿勢、
姿勢連続性、TCP軸定義を確認してください。
