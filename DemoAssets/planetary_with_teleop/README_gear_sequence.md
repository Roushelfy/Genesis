# Gear Sequence Pipeline

将遥操序列（Phase 1）与 IK 跟随序列（Phase 2）拼接为完整轨迹并渲染。

---

## 目录结构

```
planetary_with_teleop/
├── sim_gear_with_robot_setup.py   # UIPC 仿真 + IK，支持无 GUI 导出
├── stitch_gear_sequences.py       # 拼接两段序列（样条过渡）
└── README_gear_sequence.md        # 本文档

planetary_gear/
└── trajectory_gear_sharpa.npz     # Phase 1 遥操轨迹（原始）

examples/IPC_Solver/
└── replay_gear_traj_render.py     # Genesis 渲染脚本
```

---

## 工作流程

### Step 1 — Setup：调整抓握姿态（GUI）

在正式录制前，先调整右手抓握位置并保存配置。

```bash
cd Genesis_IPC_demo && .venv/Scripts/python.exe DemoAssets/planetary_with_teleop/sim_gear_with_robot_setup.py
```

操作步骤：
1. 拖动黄色 **wrist gizmo**（手腕，6-DOF）→ 右臂 IK 调整整体位置
2. 拖动橙色 **fingertip gizmos**（5 个指尖，仅平移）→ 逐指调整握持
3. 展开 **Right finger angles** 滑块精调
4. 点击 **"Save Setup Only"** 保存到 `grip_setup.json`（不启动仿真）

---

### Step 2 — 生成 IK 序列（无 GUI）

加载 `grip_setup.json`，无界面运行 IK track 阶段，导出机械臂关节角 + 齿轮 transform。

```bash
cd Genesis_IPC_demo && .venv/Scripts/python.exe DemoAssets/planetary_with_teleop/sim_gear_with_robot_setup.py --headless --ik-frames 600 --output DemoAssets/planetary_with_teleop/ik_sequence.npz
```

参数说明：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--ik-frames` | 600 | 录制帧数（×0.01 s = 秒数，600 → 6 s）|
| `--output` | `ik_sequence.npz` | 输出路径 |
| `--dt` | 0.01 | UIPC 时间步长 (s) |

输出 `.npz` 键与 `trajectory_gear_sharpa.npz` 格式完全一致，可直接送入渲染脚本。

---

### Step 3 — 拼接序列（GUI 编辑器）

打开 GUI，可交互剪辑 Phase 1 / Phase 2 的起止帧，调整过渡长度，实时预览后保存。

```bash
cd Genesis_IPC_demo && .venv/Scripts/python.exe DemoAssets/planetary_with_teleop/stitch_gear_sequences.py
```

GUI 操作：
1. **Phase 1** 折叠面板：P1 start / P1 end 滑块剪辑遥操段
2. **Transition** 折叠面板：Blend frames 调整过渡帧数
3. **Phase 2** 折叠面板：P2 start / P2 end 滑块剪辑 IK 段
4. Frame 滑块 + ▶ Play 预览；timeline 显示三段结构
5. **"💾 Save combined sequence"** 保存到 `--output` 路径

无 GUI 直接保存（可通过参数指定剪辑范围）：

```bash
cd Genesis_IPC_demo && .venv/Scripts/python.exe DemoAssets/planetary_with_teleop/stitch_gear_sequences.py --no-gui --phase1 DemoAssets/planetary_gear/trajectory_gear_sharpa.npz --phase2 DemoAssets/planetary_with_teleop/ik_sequence.npz --blend-frames 60 --p1-end 400 --p2-end 500 --output DemoAssets/planetary_with_teleop/full_sequence.npz
```

参数说明：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--phase1` | — | Phase 1 轨迹（遥操 npz）|
| `--phase2` | — | Phase 2 轨迹（IK 导出 npz）|
| `--blend-frames` | 60 | 过渡帧数（机器人关节角三次样条，齿轮 pose slerp）|
| `--output` | `full_sequence.npz` | 输出路径 |

过渡区间插值方式：
- **robot_qpos (58 DOF)**：端点速度为零的 cubic spline
- **齿轮位姿 (pos + quat)**：位置线性插值，旋转 slerp

---

### Step 4 — 渲染完整序列

```bash
cd Genesis_IPC_demo && .venv/Scripts/python.exe examples/IPC_Solver/replay_gear_traj_render.py --traj DemoAssets/planetary_with_teleop/full_sequence.npz --render --camera-traj custom
```

常用渲染参数：

| 参数 | 说明 |
|---|---|
| `--render` | 输出视频（Luisa 渲染器）|
| `--render --nyx` | 使用 Nyx 渲染器 |
| `--camera-traj custom` | 使用 `custom_camera_keyframes()` 定义的镜头路径 |
| `--camera-traj surround` | 环绕镜头 |
| `--spp 256` | 每像素采样数（默认 256）|
| `--res 1920 1080` | 分辨率 |
| `--start-frame N` | 从第 N 帧开始 |
| `--end-frame N` | 在第 N 帧结束 |

---

## 数据格式

所有 `.npz` 文件共享同一格式：

| 键 | 形状 | 说明 |
|---|---|---|
| `sim_time` | `(N,)` float32 | 仿真时间戳 (s) |
| `robot_qpos` | `(N, 58)` float32 | 机械臂关节角（Genesis qpos 顺序）|
| `rigid_sun_gear` | `(N, 7)` float32 | `[px,py,pz, qw,qx,qy,qz]` |
| `rigid_carrier` | `(N, 7)` float32 | 同上 |
| `rigid_ring_gear` | `(N, 7)` float32 | 同上（固定，各帧相同）|
| `rigid_planet_gear_0/1/2` | `(N, 7)` float32 | 同上 |

---

## 完整一键命令（按顺序执行）

```bash
cd Genesis_IPC_demo && .venv/Scripts/python.exe DemoAssets/planetary_with_teleop/sim_gear_with_robot_setup.py --headless --ik-frames 600 --output DemoAssets/planetary_with_teleop/ik_sequence.npz
```

```bash
cd Genesis_IPC_demo && .venv/Scripts/python.exe DemoAssets/planetary_with_teleop/stitch_gear_sequences.py --phase1 DemoAssets/planetary_gear/trajectory_gear_sharpa.npz --phase2 DemoAssets/planetary_with_teleop/ik_sequence.npz --blend-frames 60 --output DemoAssets/planetary_with_teleop/full_sequence.npz
```

```bash
cd Genesis_IPC_demo && .venv/Scripts/python.exe examples/IPC_Solver/replay_gear_traj_render.py --traj DemoAssets/planetary_with_teleop/full_sequence.npz --render --camera-traj custom --spp 256
```
