# g1_pika_description

Unitree G1 29DoFモデルへPIKAグリッパとTCPフレームを取り付けたROS 2 description packageです。

## TCP軸の定義

- `right_pika_tcp +X`: グリッパ上方向
- `right_pika_tcp +Y`: グリッパ開閉方向
- `right_pika_tcp +Z`: グリッパ前方

PIKAトラッカーの計測軸は、実測結果に基づき次のように対応させています。

- PIKA `+X` -> TCP `+Z`（前方）
- PIKA `+Y` -> TCP `-Y`（開閉軸の符号反転）
- PIKA `+Z` -> TCP `+X`（上方向）

取り付け姿勢の正本は `scripts/generate_urdf.py` です。生成後の
`urdf/g1_29dof_pika.urdf` を直接編集せず、ジェネレータを変更してください。

## 表示

```bash
colcon build \
  --base-paths src/pika_ros/src/g1_pika_description \
  --packages-select g1_pika_description \
  --symlink-install
source install/setup.bash
ros2 launch g1_pika_description display.launch.py
```

RVizには `base_link`、`pelvis`、左右のTCPを表示します。`base_link -> pelvis` の
固定変換はlaunch引数 `base_x/y/z` と `base_roll/pitch/yaw` で変更できます。
