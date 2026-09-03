# G1 + PIKA リターゲティング

この文書は、PIKA収録データをUnitree G1 + PIKAへ変換し、MuJoCoで検証してから
学習・実機推論へ進むための再現手順です。

リターゲット処理だけをすぐに実行する場合は
[G1_PIKA_RETARGET_QUICKSTART.md](G1_PIKA_RETARGET_QUICKSTART.md) を先に参照してください。

## 1. 構成と前提

想定するワークスペース構成です。

```text
<workspace>/
  src/pika_ros/
  src/lerobot/
```

必要な環境:

- Ubuntu 22.04 / ROS 2 Humble
- Python 3.12環境（例: `pika_g1_ik`）
- MuJoCo、Pinocchio/Unitree IK、FFmpeg
- `pika_ros` の `PikaAnyArm` と `data_tools` submodule
- `unitree_g1_pika` を含むLeRobot fork

この実装で確認したLeRobot基準コミットは `79edf6a9` です。少なくとも
`lerobot.robots.unitree_g1_pika` と、G1 IKが外部URDFを受け取れる変更が必要です。

```bash
git clone --recurse-submodules <pika_ros-fork-url> <workspace>/src/pika_ros
git clone https://github.com/taigasasaki6331/lerobot.git <workspace>/src/lerobot
git -C <workspace>/src/lerobot checkout 79edf6a9
cd <workspace>/src/pika_ros
WORKSPACE_DIR=<workspace> scripts/setup_g1_pika_retarget.bash
source <workspace>/install/setup.bash
export PYTHONPATH=<workspace>/src/lerobot/src:${PYTHONPATH:-}
export G1_PIKA_URDF=<workspace>/src/pika_ros/src/g1_pika_description/urdf/g1_29dof_pika.urdf
```

MuJoCoモデルはLeRobotの `unitree-g1-mujoco` assetを使用します。自動検出できない場合:

```bash
export G1_MUJOCO_XML=/absolute/path/to/g1_29dof_no_hand.xml
```

## 2. URDFと座標系の確認

```bash
cd <workspace>
colcon build \
  --base-paths src/pika_ros/src/g1_pika_description \
  --packages-select g1_pika_description \
  --symlink-install
source install/setup.bash
ros2 launch g1_pika_description display.launch.py
```

右TCPの軸契約:

- `right_pika_tcp +X`: グリッパ上方向
- `right_pika_tcp +Y`: グリッパ開閉方向
- `right_pika_tcp +Z`: グリッパ前方

取り付け姿勢の正本は
`src/g1_pika_description/scripts/generate_urdf.py` です。URDFを直接変更せず、
ジェネレータ変更後に再生成してRVizとMuJoCoの両方を確認してください。

## 3. 学習データの変換

既存の各 `episodeN/data.hdf5` を再利用し、1ステップ先の相対TCP actionへ変換する例:

```bash
cd <workspace>/src/pika_ros
LEROBOT_PY=<python-environment>/bin/python \
scripts/convert_multi_pika_to_lerobot.bash \
  --dataset-dir /path/to/raw_dataset \
  --all --from 0 --to 59 \
  --target-dir <workspace>/datasets/replace_tape_g1_pika_relative_h1_v2 \
  --fps 30 \
  --type single_pika \
  --reuse-hdf5 \
  --relative-trajectory \
  --relative-action-horizon 1
```

変換データのactionは相対TCPの `xyz + rotation-6D + gripper`（10次元）です。
状態は相対基準姿勢とグリッパを表す10次元で、現在の実装は視覚主体の学習を想定します。

変換時にはG1全関節0の開始姿勢
`config/g1_pika_retarget_start_pose.json` が
`<dataset>/meta/retarget_start_pose.json` にコピーされます。開始姿勢を変更する場合は
関節値だけでなく、同JSONのTCP位置・回転行列を現在のURDFのFKから更新し、次を実行して
`FK validation: OK` を確認してください。

```bash
PYTHONPATH=<workspace>/src/lerobot/src \
G1_PIKA_URDF=<workspace>/src/g1_pika_description/urdf/g1_29dof_pika.urdf \
python scripts/print_g1_pika_start_pose.py
```

基準右TCPはpelvis座標で
`[0.379774281, -0.148617218, 0.095222941] m`、姿勢quaternion (xyzw) は
`[0.7071248133, 0.0003692098, 0.7070886136, 0.0002337141]` です。
相対軌道はこの基準姿勢から積分されるため、PIKA Stationの絶対配置は開始位置へ加算しません。

既存episode 10の相対軌道では全関節0開始時にIK位置誤差が最大許容値を超えたため、
再生成後に代表episodeの全step到達性を必ず確認してください。

## 4. Teacher軌道のMuJoCo検証

まず代表episodeを4視点、TCP/IKラベル付きで確認します。

```bash
cd <workspace>/src/pika_ros
for ep in 10 20 30 40 50; do
  WORKSPACE_DIR=<workspace> \
  DATASET_ROOT=<workspace>/datasets/replace_tape_g1_pika_relative_h1_v2 \
  EPISODE="$ep" STEPS=all CAMERA_LAYOUT=four \
  scripts/run_teacher_updated_urdf_ik_mujoco.bash
done
```

動画とCSVは既定で `<workspace>/videos/updated_urdf_ik_episodeN/` に出力されます。
位置・姿勢誤差、上下反転、roll 90度ずれ、関節不連続がないことを確認します。

## 5. ACT学習

相対action horizonが1なので、ACTのaction chunkも1に合わせます。古い
`chunk_size=100` のモデルはこのデータ形式では再利用せず、再学習してください。

```bash
cd <workspace>/src/lerobot
lerobot-train \
  --dataset.repo_id=kfstiger/replace_tape_g1_pika_relative_h1_v2 \
  --dataset.root=<workspace>/datasets/replace_tape_g1_pika_relative_h1_v2 \
  --policy.type=act \
  --policy.chunk_size=1 \
  --policy.n_action_steps=1 \
  --policy.device=cuda \
  --output_dir=<workspace>/outputs/train/replace_tape_g1_pika_relative_h1_v2_act \
  --job_name=replace_tape_g1_pika_relative_h1_v2_act \
  --steps=40000 \
  --save_freq=10000
```

CPU学習では `--policy.device=cpu` に変更します。バッチサイズ等はGPUメモリに合わせて
追加してください。

## 6. Policyのオフライン評価とMuJoCo動画

チェックポイントの教師actionとの誤差比較:

```bash
WORKSPACE_DIR=<workspace> \
CHECKPOINT_ROOT=<workspace>/outputs/train/replace_tape_g1_pika_relative_h1_v2_act/checkpoints \
LAST_POLICY_PATH=<workspace>/outputs/train/replace_tape_g1_pika_relative_h1_v2_act/checkpoints/last/pretrained_model \
DATASET_ROOT=<workspace>/datasets/replace_tape_g1_pika_relative_h1_v2 \
EPISODES="10 20 30 40 50" DEVICE=cpu \
<workspace>/src/pika_ros/scripts/evaluate_act_checkpoints.bash
```

Policy動画:

```bash
WORKSPACE_DIR=<workspace> \
POLICY_PATH=/path/to/checkpoint/pretrained_model \
DATASET_ROOT=<workspace>/datasets/replace_tape_g1_pika_relative_h1_v2 \
EPISODE=10 STEPS=all DEVICE=cpu POLICY_ACTION_STEPS=1 \
<workspace>/src/pika_ros/scripts/run_policy_updated_urdf_ik_mujoco.bash
```

`POLICY_ACTION_STEPS=1` は必須です。毎stepで最新観測から再推論し、古いACT action
queueを連続消費しない設定です。IK誤差を診断中は `FAIL_ON_IK_ERROR=0`、合否判定では
`FAIL_ON_IK_ERROR=1` を使用します。

## 7. 実機へ進む前の段階試験

実機試験は必ず次の順序で行います。

1. MuJoCoでteacher全stepを確認
2. 5 mm TCP dry-run
3. 5 mm TCP実機試験
4. 検証済みCSVを5step、20step、全stepの順で再生
5. policyを5step、20stepの順で試験
6. 最後にグリッパ制御を有効化

TCP dry-run（`--real` を付けるまでLowCmdは送信しません）:

```bash
python scripts/test_g1_pika_tcp_step.py \
  --robot-ip 192.168.123.164 \
  --frame base \
  --delta "0 0 0.005"
```

実機ではアームを支持し、非常停止可能な状態で実行します。

```bash
python scripts/test_g1_pika_tcp_step.py \
  --robot-ip 192.168.123.164 \
  --frame base \
  --delta "0 0 0.005" \
  --gravity-compensation \
  --closed-loop-hold \
  --outer-loop-gain 0.005 \
  --outer-correction-step-rad 0.0001 \
  --real
```

検証済みCSVを5step再生:

```bash
ACTION_SOURCE=trajectory \
TRAJECTORY_CSV=/path/to/verified_unitreeik.csv \
STEPS=5 FPS=5 MAX_JOINT_STEP=0.002 \
TRAJECTORY_START_MAX_MOVE_RAD=0.60 \
TRAJECTORY_MAX_DEVIATION_RAD=0.60 \
CONTROL_GRIPPER=0 HOLD_AFTER_RUN=1 \
scripts/run_policy_ik_from_current_fk.sh
```

## 8. Live cameraとpolicy実機試験

G1側でカメラサーバを起動します。

```bash
python scripts/serve_pika_live_cameras.py \
  --host 0.0.0.0 --port 5562 \
  --d405-serial 315122271825 \
  --fisheye-device /dev/v4l/by-id/<DECXIN-video-index0>
```

推論PC側の5step試験:

```bash
LIVE_STATE=1 \
LIVE_PIKA_CAMERA_SERVER=192.168.123.164 \
LIVE_PIKA_CAMERA_TIMEOUT_MS=1000 \
LIVE_PIKA_CAMERA_RETRIES=3 \
ACTION_SOURCE=policy \
POLICY_PATH=/path/to/pretrained_model \
POLICY_ACTION_STEPS=1 \
STEPS=5 FPS=5 ORIENTATION_WEIGHT=0.0 \
MAX_JOINT_STEP=0.002 MAX_JOINT_DEVIATION=0.03 \
SERVO_TRACKING_ERROR_LIMIT_RAD=0.05 \
CONTROL_GRIPPER=0 HOLD_AFTER_RUN=1 \
scripts/run_policy_ik_from_current_fk.sh
```

`CAMERA_PREFLIGHT: OK` が出る前にG1制御は開始されません。`Ctrl-C` 後はpassiveに
なるため、常にアームを支持してください。

## 9. よくある問題

- `G1_PIKA_URDF was not found`: `G1_PIKA_URDF` を絶対パスで設定する。
- `g1_29dof_no_hand.xml was not found`: `G1_MUJOCO_XML` を設定する。
- CUDA driver error: GPUのないPCでは `DEVICE=cpu` を指定する。
- `zmq.error.Again`: G1側のカメラサーバ、TCP 5562、IP到達性を確認する。
- IK tolerance exceeded: toleranceを無制限に緩めず、最初にpolicy actionと姿勢連続性を確認する。
- 動作が途中でほぼ止まる: `MAX_JOINT_DEVIATION` 制限とaction積算方式をログの
  `cmd-target` で確認する。

## 10. 共有アーカイブとforkへのpush

コミット済みHEADから再現可能なtar.gzを作成します。

```bash
cd <workspace>/src/pika_ros
scripts/package_g1_pika_retarget.bash
sha256sum -c dist/*.tar.gz.sha256
```

fork作成後、利用者自身がremoteを設定してpushします。

```bash
git remote rename origin upstream
git remote add origin <your-fork-url>
git push -u origin feature/g1-pika-retarget-package
```

モデル重み、収録データ、生成動画、`build/`、`install/` はGitへ含めません。
