# 自动半径＋DPR 使用范例

## 表面自动转换为薄壳（2026-10-31）

选择 `--backend shell --mesh model.obj --thickness 0.0005` 可自动生成并缓存 Q4 壳资产，
再调用原生自动半径＋DPR。输入表面明确作为**中面**；`--scale` 把输入坐标转换为米，
`--thickness` 始终是米，不再乘以 `--scale`。把实体的外表面指定为中面，会改成薄壳物理模型，
不会推断实体的内部空腔或真实中面。验证结果、性能与限制见[报告](AUTOMATIC_SURFACE_TO_SHELL_RESULTS_20261031.md)。

```bash
# 具有实际厚度的闭合或开口中面；有可用显示环境时着色。
QD_GRAPH=0 python -m examples.rigid.franka_egg_automatic_stress \
  --backend shell --mesh model.obj --thickness 0.0005 --viewer

# 毫米制 STL，中面坐标转为米，厚度仍为 0.5 mm；批量只输出最大值。
QD_GRAPH=0 python -m examples.rigid.franka_egg_automatic_stress \
  --backend shell --mesh model.stl --scale 0.001 --thickness 0.0005 --envs 64

# 原有 authored egg 无需转换；volume 仍是省略 --backend 时的默认。
QD_GRAPH=0 python -m examples.rigid.franka_egg_automatic_stress --backend shell --viewer
```

转换将每个输入三角形划分成三个共形 Q4，新增边中点和面中心，不移动原中面、不进行平滑或近邻焊接。
STL 重复顶点只在坐标完全相等时合并。F 个三角形生成 3F 个 Q4、V+E+F 个节点及 6(V+E+F) 个弹性自由度。
建议从数十至数百个质量良好的三角形开始；这不是减面工具，密集逆矩阵的构造／存储仍随自由度快速增长。
闭合／开口、可定向、连通的非凸曲面可转换，孔和边界会保留；退化、重复、非流形、自交、不可定向、
不连通组件及厚度偏移后相交或倒置的输入会明确拒绝。尖锐折线不会被圆角化，过厚或过急折叠仍可能被拒绝。

自定义壳的碰撞面是两侧偏移面与开口封边组成的材料边界，禁用凸包化、减面和补洞包覆；质量、质心与惯性
从该边界计算。无需人为填实空腔。厚度、缩放、源几何（包括外部 glTF buffers）、转换配方参与缓存身份，
`--cache-dir` 可显式指定项目数据目录；缓存只用于一次构造，不在每个环境或时间步重复转换。
在本集群运行前应把 `GS_CACHE_FILE_PATH`、`--cache-dir` 和日志路径设在项目 data 目录或私有节点临时目录，
参见 [DATA_LOCATION.md](../../../DATA_LOCATION.md)。

自定义 shell 的材料碰撞面默认使用 4 个物理子步；原有 authored egg 和 volume 保持 1 个。
可用 `--substeps N` 显式指定正整数。四子步设置已通过转换后 egg 的 730 步夹持、非凸窄颈与开口 dome 的 140 步检查，减少粗时间步穿过
薄壁而产生内侧接触的情况；它不是任意厚度／运动速度的稳定性保证。总步长仍为 0.01 s。多子步时，
最大值覆盖该步所有子步，完整场是最后子步的场，两者的最大值不必相同。

转换成功并不保证每种接触都能使用 Hertz 半径。现有曲率、压力拟合、力／力矩和弹性残差检查仍生效；
平面与平面的零相对曲率、凹面或未被支持的边缘接触可能被拒绝。也不自动生成适合任意形状的抓取轨迹。
Q4 同网格解的一致性与空间精度是不同检查；壳面内应力不等于三维接触峰值，锐边峰值仍依赖采样和离散。

## 默认 egg、自定义 mesh 与应力着色

`franka_egg_automatic_stress.py` 复用现有 Franka 抓取场景。现在省略 `--viewer` 时默认选择
`output_mode="max"`，定期输出 Pa 单位的最大应力；加 `--viewer` 后自动选择 `"full"`，在鸡蛋或
自定义刚体的未变形边界上显示 von Mises 应力颜色。仍可显式选择无 viewer 的 full 输出。

```bash
# 默认空壳 egg，只有最大应力输出；--envs 可增加并行环境。
QD_GRAPH=0 python -m examples.rigid.franka_egg_automatic_stress --envs 64

# 同一空壳 egg，显示应力场；也可添加 --backend shell。
QD_GRAPH=0 python -m examples.rigid.franka_egg_automatic_stress --viewer

# 闭合表面网格：首次运行生成实心体积弹性网格并缓存，之后复用。
QD_GRAPH=0 python -m examples.rigid.franka_egg_automatic_stress --mesh model.obj --viewer

# 毫米制 STL 转为米；也可以省略 --viewer，只输出最大应力。
QD_GRAPH=0 python -m examples.rigid.franka_egg_automatic_stress --mesh model.stl --scale 0.001 --viewer

# 空腔、薄壁等物体使用已经准备好的匹配弹性网格。
QD_GRAPH=0 python -m examples.rigid.franka_egg_automatic_stress \
  --mesh collision.obj --stress-mesh elastic.npz --viewer
```

通用表面加载使用 Trimesh 支持的格式，如 OBJ/STL/PLY/GLB；自动四面体化需要一个闭合、方向一致的
实心区域，不会从外表面猜测壁厚或内部空腔。默认 egg 始终复用原空壳资产，不会重新填成实心。
`--mesh volume.npz` 可直接从体积资产提取碰撞边界。自定义 shell 可自动转换，也可用
`--backend shell --mesh midsurface.npz` 直接重载 Q4 资产，或通过 `--stress-mesh midsurface.npz` 显式指定匹配资产。
中面 NPZ 必须含 Q4 connectivity，可用 `--thickness` 指定米制壁厚。自定义 shell 的碰撞面始终从
选定 Q4 中面重建；显式提供 NPZ 时，以该资产的几何为准。表面与弹性 NPZ 必须属于同一 authored frame；预处理对二者一起缩放并移到共同包围盒
中心，保留彼此的几何配准。模型朝向不变。自定义刚体的质量、质心和惯性从弹性材料边界导出并写入
缓存 URDF；薄壳包含内外侧和开口边缘的壁厚封边，不会按外包络的实心体积估计惯性。默认 egg 仍
使用其原始 URDF 惯性。`--cache-dir` 可指定缓存位置；否则遵循 Genesis 的
`GS_CACHE_FILE_PATH` / XDG 缓存目录。`--max-volume` 控制新四面体的最大体积，单位 m³。

viewer 显示单个环境，headless 支持批量。蓝色表示零应力，红色表示颜色范围上限；默认上限跟随本次
运行已经观察到的全域峰值递增，具体 Pa 数值会打印。使用 `--color-max 1000000` 可固定为 0–1 MPa，
便于比较不同帧。3D 颜色取自每个边界三角形所属四面体的角点，不跨单元平滑；shell 每个面使用该侧
四个 Gauss 点的最大值。开口厚度封边显示为灰色，表示这里没有独立的三维封边应力采样。内部热点不一定在可见表面，终端报告仍是全域最大值。这个显示不产生弹性
变形，也不改变刚体碰撞形状。默认材料为 E=10 GPa、ν=0.3、ρ=2000 kg/m³。

现有抓取轨迹不是任意物体的自动抓取规划器。网格尺寸、朝向和抓取部位需要适合 Panda 的开口与工作
空间；高模应先准备合适的低模弹性资产，以控制构造和恢复开销。原生接触半径、载荷及弹性求解的
适用范围和拒绝检查仍然有效。

没有 GPU 时可添加 `--device cpu`。此前 `aa5a7d093` 入口在 CPU 上检查默认 egg 的 130 步非零应力输出、
自定义 STL 的 30 步加载流程，以及软件 EGL viewer 的 volume/full 30 步与 shell/full 10 步着色调用；
另检查了 volume 在第 130 步的实际渲染画面。网格缓存、比例缩放、边界归属、壳内外侧和颜色范围有
独立测试。该阶段没有 GPU 或完整 730 步抓取的新增性能结论。October 31 自动壳转换的新增验证与性能记录另见上方进展。下文此前的
性能与 viewer 资格仍属于其各自冻结版本。

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
