# 自动半径＋DPR 使用范例

普通 Options 和 `.dpr()` 默认选择 pressure/patch/Hertz 半径/Hertz 压力轮廓。体积加载默认采用 adaptive/q16，仅对未通过独立积分检查的接触升级至 q64、q128。薄壳仍默认 Q4 加载；下面显式选择已验证的 planar/k32 配方。公共示例、完整历史及三组配对计时均已完成，精度、速度和显存取舍见[结果报告](AUTOMATIC_DPR_FAST_PATH_RESULTS_20261028.md)。

在仓库根目录使用已安装 Genesis／Quadrants 的环境。集群上遵循 [运行位置说明](RUNNING.md)，在计算节点执行；下面不涉及 Slurm 或缓存配置。所有 stress 配置在 `scene.build()` 之前完成。

## 在自己的场景里配置

`link` 为已有刚体 link。3D 的 `elastic.npz` 是体积 FEM 网格；薄壳的 `shell.npz` 是中面四边形网格，不可互换。仓库鸡蛋资产位于 `examples/rigid/assets/hollow_egg/level1/`。

3D 使用公共默认值，只需指定弹性网格、夹爪边缘策略和输出模式：

```python
link.configure_stress_recovery(
    gs.options.RigidStressOptions.dpr(
        mesh="examples/rigid/assets/hollow_egg/level1/elastic.npz",
        box_edge_policy="planar",
        output_mode="max",
    )
)
```

薄壳的大批量优选配置保持原 Q4 弹性，仅细分三角加载面：

```python
link.configure_stress_recovery(
    gs.options.RigidStressOptions.dpr(
        mesh="examples/rigid/assets/hollow_egg/level1/shell.npz",
        backend="shell",
        shell_thickness=0.0005,
        shell_loading="planar",
        loading_subdivisions=32,
        radius_model="hertz",
        pressure_profile="hertz",
        box_edge_policy="planar",
        quadrature=10,
        scatter_tasks_per_env=128,
        inverse_precision="64",
        history_size=0,
        output_mode="full",
    )
)
```

两例的材料仍使用当前默认 E=10 GPa、ν=0.3；实际物体应提供相应材料和固定厚度。`box_edge_policy="planar"` 是鸡蛋／锐边夹爪范例显式采用的廉价边缘曲率近似，不表示夹爪有软垫，也不保证锐边物理峰值准确。其他几何应选择适用的支持策略。

短入口 [franka_egg_automatic_stress.py](../../../examples/rigid/franka_egg_automatic_stress.py) 直接构造公共 Options，默认执行730步抓取。可分别运行体积和薄壳、最大值和完整场：

```bash
QD_GRAPH=0 python -m examples.rigid.franka_egg_automatic_stress --envs 64
env -u GS_HEADLESS PYOPENGL_PLATFORM=glx PYGLET_HEADLESS=false QD_GRAPH=0 \
  python -m examples.rigid.franka_egg_automatic_stress --output-mode full --viewer
QD_GRAPH=0 python -m examples.rigid.franka_egg_automatic_stress --backend shell --envs 64
env -u GS_HEADLESS PYOPENGL_PLATFORM=glx PYGLET_HEADLESS=false QD_GRAPH=0 \
  python -m examples.rigid.franka_egg_automatic_stress --backend shell --output-mode full --viewer
```

Viewer 命令需要可用的 X11 显示。图形平台变量必须在导入 Genesis/OpenGL 前设置；集群镜像的 headless EGL 与 X11 GLX 窗口采用不同的上下文。自动验证使用私有 Xvfb 和 Mesa 软件 GLX 渲染，物理仍为 CUDA/FP64；这项验证不提供渲染吞吐量。体积／薄壳的 730 步 headless 和实际 GLX viewer 入口均已验证通过。

## 完整 Franka 抓蛋：小批量与并行

以下完整程序调用仓库已有的 [FrankaEgg 范例](../../../examples/rigid/franka_egg_stress.py)。在仓库根目录保存后运行，或将代码传给 `python -`。改变 `BACKEND` 即可选择 3D／薄壳；`ENVS` 控制并行数，`VIEWER=True` 用于有显示环境的小批量观察。

```python
import os
os.environ["QD_GRAPH"] = "0"  # 当前测量配置；必须在导入 Genesis／Quadrants 前设置。

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg

BACKEND = "shell"  # 或 "volume"
OUTPUT = "max"     # 或 "full"
ENVS = 4          # headless 并行可从 64、1024 开始，按设备显存增加。
VIEWER = False

gs.init(backend=gs.gpu, precision="64")
options = dict(
    stress_backend=BACKEND,
    radius_model="hertz",
    pressure_profile="hertz",
    box_edge_policy="planar",
    inverse_precision="64",
    history_size=0,
    output_mode=OUTPUT,
)
if BACKEND == "shell":
    options.update(
        shell_loading="planar",
        loading_subdivisions=32,
        quadrature=10,
        scatter_tasks_per_env=128,
    )

grasp = FrankaEgg(n_envs=ENVS, level=1, viewer=VIEWER, **options)
for _ in range(600):
    grasp.step()
grasp.scene.rigid_solver.check_errno()
print("Sampled maximum von Mises stress [Pa]:", grasp.link.get_max_stress())
if OUTPUT == "full":
    stress, von_mises = grasp.link.get_stress_field()
    print("Stress / von Mises shapes:", stress.shape, von_mises.shape)
    if BACKEND == "shell":
        membrane, bending, shear = grasp.link.get_shell_resultants()
        print("N / M / Q shapes:", membrane.shape, bending.shape, shear.shape)
```

这是脚本抓取使用示例，未重现报告的策略网络、冻结条件库、warmup 或配对计时，不应用它的运行时间冒充报告吞吐量。实际公共入口运行记录见执行记录。无显示的大并行使用 `VIEWER=False`。完整场占用更多显存，先从较小批量开始。

## 输出与旧配置对照

`output_mode="max"` 在 build 前选择，避免完整场／壳合力场的输出存储；`get_max_stress()` 返回全物体采样点的最大 von Mises 应力，单位 Pa。完整场选择 `"full"`，调用 `get_stress_field()` 返回应力张量及 von Mises。体积张量为 `[B,n_tets,4,6]`，VM 为 `[B,n_tets,4]`。薄壳张量为 `[B,n_quads,2,4,6]`，包括内外两侧、四个 Gauss 点和六个分量；壳 N/M 的形状为 `[B,n_quads,4,6]`，Q 为 `[B,n_quads,4,3]`，单位分别为 N/m、N、N/m。无批维场景省略 B。壳应力是膜＋弯曲的面内应力，与 3D 接触附近的三维应力不同。更多输出和失效状态定义见 [API](NATIVE_API.md) 与 [壳指南](RUNNING_NATIVE_SHELL_DPR.md)。

要显式复现旧半径／压力选择，在相应配置中使用：

```python
legacy = gs.options.RigidStressOptions.dpr(
    mesh="examples/rigid/assets/hollow_egg/level1/elastic.npz",
    backend="volume",
    radius_model="manual",
    contact_radius=0.006,
    pressure_profile="dpr",
)
```

该配置保留手动6mm半径、旧压力轮廓与 edge/q10。复现历史性能仍需原报告的源码、积分阶数和场景。显式给定 `quadrature` 而省略规则时，也保留固定 edge 积分；需要自适应时显式选 `pressure_quadrature="adaptive"`，其起始阶数固定为16。`.scr(...)` 仍选择手动半径和 body-source-prior，不继承体积自适应积分。

下面通过常规 Franka CLI 对照手动旧配置与固定高阶参考；原来的力矩、拟合、弹性残差检查均保留：

```bash
QD_GRAPH=0 python -m examples.rigid.franka_egg_stress --envs 4 --steps 730 \
  --radius-model manual --pressure-profile dpr --quadrature 10 --box-edge-policy planar
QD_GRAPH=0 python -m examples.rigid.franka_egg_stress --envs 4 --steps 730 \
  --pressure-quadrature angle --quadrature 128 --box-edge-policy planar --output-mode full
```
