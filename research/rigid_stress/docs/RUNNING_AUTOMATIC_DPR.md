# 自动半径＋DPR 使用范例

这些范例使用现有原生 API。**截至 `73af7b0d1`，代码默认仍是 manual 半径和旧 DPR 压力轮廓，所以下面显式选择 Hertz。** [下一阶段目标](AUTOMATIC_DPR_FAST_PATH_DEFAULTS_GOAL_20261028.md)要求实际修改默认值、恢复便宜且准确的路径，并运行验证这些范例；本页不是该工作的完成声明。

在仓库根目录使用已安装 Genesis／Quadrants 的环境。集群上遵循 [运行位置说明](RUNNING.md)，在计算节点执行；下面不涉及 Slurm 或缓存配置。所有 stress 配置在 `scene.build()` 之前完成。

## 在自己的场景里配置

`link` 为已有刚体 link。3D 的 `elastic.npz` 是体积 FEM 网格；薄壳的 `shell.npz` 是中面四边形网格，不可互换。仓库鸡蛋资产位于 `examples/rigid/assets/hollow_egg/level1/`。

当前已经通过完整策略控制的 3D 保守配置如下。它使用较贵的 angle/q128；新目标应将这里替换成验证通过的快路径／选择性回退配置。

```python
link.configure_stress_recovery(
    gs.options.RigidStressOptions.dpr(
        mesh="examples/rigid/assets/hollow_egg/level1/elastic.npz",
        backend="volume",
        radius_model="hertz",
        pressure_profile="hertz",
        box_edge_policy="planar",
        pressure_quadrature="angle",
        quadrature=128,
        inverse_precision="64",
        history_size=0,
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
else:
    options.update(pressure_quadrature="angle", quadrature=128)

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

这是脚本抓取使用示例，未重现报告的策略网络、冻结条件库、warmup 或配对计时，不应用它的运行时间冒充报告吞吐量。本次文档更新只核对语法和当前接口；GPU运行验证属于新目标。无显示的大并行使用 `VIEWER=False`。完整场占用更多显存，先从较小批量开始。

## 输出与旧配置对照

`output_mode="max"` 在 build 前选择，避免完整场／壳合力场的输出存储；`get_max_stress()` 返回全物体采样点的最大 von Mises 应力，单位 Pa。完整场选择 `"full"`，调用 `get_stress_field()` 返回应力张量及 von Mises。薄壳张量为 `[B,n_quads,2,4,6]`，包括内外两侧、四个 Gauss 点和六个分量；壳 N/M/Q 的单位分别为 N/m、N、N/m。壳应力是膜＋弯曲的面内应力，与 3D 接触附近的三维应力不同。更多输出和失效状态定义见 [API](NATIVE_API.md) 与 [壳指南](RUNNING_NATIVE_SHELL_DPR.md)。

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

这只指定旧的半径／压力组合；复现历史性能还需使用原报告的源码、积分阶数和场景。新默认迁移后仍应保留这类显式覆盖，以及原严格检查和高阶参考模式。
