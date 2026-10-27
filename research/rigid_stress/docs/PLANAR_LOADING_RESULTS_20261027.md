# Planar loading DPR：原生实现与测量，2026-10-27

**已完成：原生实现、准确性、完整优化与168项正式性能资格均已交付。**
本文对应 [远端目标](PLANAR_LOADING_DPR_NATIVE_GOAL_20261027.md)。已经合格的配置为
shell planar k32、radial q10、cell blocks、每环境128个任务、FP64、history0，
并在本任务进程内设置 `QD_GRAPH=0`。它保留原 Q4 弹性模型，新增加载细分不增加弹性自由度。
完整策略准确性通过不代表任意几何或连续体接触峰值已经收敛。

## 实现与建议配置

测量引擎固定为 `6dedc79579ac0d113e4baac26eb538bcf51ca57d`，独立 CPU oracle 为
`797203553f246ca155bbef909c1bb33e7cd7b90c`。交付分支后续提交包含研究汇总和文档，
不能把后续 HEAD 当成测量时的源码。原始数据、日志、缓存和大型数组位于：

```text
/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261027-planar-loading-dpr
```

以下称该目录为 `R`。各 campaign 保存 `source.json`、`analysis-files.json`、
`tasks.json` 和实际 `workload-*.sh`；研究脚本的捕获身份与引擎 revision 分别记录。
科学执行均使用 Slurm `rtx-mid` / image `1_26`。源码与文档留在 workspace；
没有将新运行数据回写用户主目录，也没有迁移活动环境或改动共享集群缓存设置。

31份本阶段冻结源码中，30份可从交付任务分支恢复。独立serial Armijo因果对照
`af7abc497e30df4704684997a434496cc2293046`另有远端tag
`rigid-stress-planar-20261027/planar-loading-pressure-delta-v1`，不依赖本地悬空Git对象。
精确远端可达性记录为 `R/meta/frozen-source-archive-verified-v19.json`。

公开默认 `shell_loading="q4"` 保持原加载路径。仅选择 `"planar"` 时，初始
k8/cap32 默认值**不是**本文完整异质策略轨迹的推荐配置。推荐显式设置：

```python
# 在导入 Quadrants/Genesis 之前，为本任务进程设置 QD_GRAPH=0；gs.init 使用 precision="64"。
options = gs.options.RigidStressOptions(
    mesh=shell_mesh,
    backend="shell",
    shell_loading="planar",
    loading_subdivisions=32,
    radius_model="hertz",
    box_edge_policy="planar",
    pressure_profile="hertz",
    contact_model="pressure",
    quadrature=10,
    scatter_tasks_per_env=128,
    inverse_precision="64",
    history_size=0,
    output_mode="max",  # 需要全场时改为 full。
)
```

内部默认采用 cooperative anchor，k≥16 使用保守的4×4 cell blocks。研究 benchmark
有独立开关，会覆盖内部选择，复现时须传 `--cooperative-anchor --planar-query blocks`。
不要将只修改这些研究开关的对照结果当成未改动公开默认的结果。

原 Q4 K、质量、gauge、inertia relief、位移响应与全物体 stress recovery 不变，
总弹性自由度为1452，原生平移 gauge 后为1446。QD 生成内外表面的三角加载面与稀疏
Q4 位移映射，以 `u=Jq`、`f=Jᵀg` 传递载荷。固定 cell 几何与层次包围盒共享；
每环境仅保留有界接触／任务 workspace。Host负责原始输入读取、整数拓扑、配置及
dtype epsilon/tiny常量；物理几何、插值、曲率、包围盒等浮点准备和运行在现有rigid
stress feature内的Quadrants中执行，CPU独立实现作为oracle。

Hertz 半径继续取原几何与曲率，原 source point、anchor、法向力、摩擦及偏心力矩不变。
保留原 surface-ball 的平方根压力律。径向积分解析化，角向仍作分段求积；
非负压力拟合保留 active-pressure 边界。有限 footprint 不按空间距离跨接相邻材料片。
共同边必须在开球内实际连通；单 Q4 的充分连通证书不成立时保守拒绝。
能确定原材料分量时，proxy unsupported、warped miss、压力失败和容量溢出通过原路径
处理整个受影响环境，避免只累加部分代理载荷。若原contact-ball连通性本身无法获得
证书，则保留MATERIAL拒绝（contact status10／errno128），不制造一个通过的fallback。

## 准确性与原门限

`R/meta/error-budgets.json` 与 `guard-budgets.json` 在调参前冻结。
原未缩放 K 残差门为 `max(1e-11, 1e-8*||rhs||)`，每 source 的力与原作用点力矩门
仍为1e-8。CPU/native 算子门1e-10、载荷与场门1e-8 使用各物理量记录的绝对下限。
这些数值门与加载面的空间近似误差分别检查。

原空间预算为 mean field5%、worst field10%、sampled peak5%、N/M/Q各10%。
k32 继承更严格的 k16 目标，未放宽：field0.5%、peak0.3%、N0.15%、M0.75%、Q0.4%。
场误差使用面积加权对称 tensor Frobenius 范数，并单列整个物体的采样峰值与 N/M/Q；
守恒通过本身不构成应力准确性证书。

| 最终 GraphOff 实际策略资格 | B4，711步 | B64，1192步 | k32 上限 |
| --- | ---: | ---: | ---: |
| worst full tensor field | 0.0498702% | 0.0828008% | 0.5% |
| worst sampled peak | 0.0472387% | 0.0735927% | 0.3% |
| N | 0.0187961% | 0.0289603% | 0.15% |
| M | 0.0889661% | 0.162235% | 0.75% |
| Q | 0.0618653% | 0.0803782% | 0.4% |
| proxy contacts | 9,988 | 218,030 | — |
| 真正 native task 最大占用 | 50 | 63 | 容量128 |
| fallback / overflow / invalid | 0 / 0 / 0 | 0 / 0 / 0 | — |

该轨迹使用730步实际策略 warmup，再 restart；延迟条件保证各环境完成首个600步周期。
低阶原 Q4 对照分别为 q24/q16，独立 resolved 对照均为 q48。每份结果同时记录
planar/resolved、低阶Q4/resolved的不确定度、planar/低阶Q4，不能互相替代。
策略是固定 seed99173 的26→128→128→7 FP32 MLP inference，权重 SHA256 为
`37952a933b9ceb5bfcd86473abed687ee3fc7bce9409385f1567ef0bd4f7aa64`。
FP64 权重副本 warmup 对照的动作差≤1.0873e-11 rad。这里没有训练或优化器吞吐量结论。

资格细节见 `R/meta/final-accuracy-config-handoff-v19.json` 与
`policy-blocks-cap128-graphs-off-qualification-v19.json`。其余必要控制为：

| 控制 | 证据与结果 | 适用范围 |
| --- | --- | --- |
| CPU、原算子、加载映射 | CPU原17检查；egg/torus各36算子检查；77 primitives通过 | CPU单侧 bank 与生产双侧轨迹分别验证 |
| k8/k16/k32 bank | D1与最终blocks各1092检查通过 | 原 K、raw loads、完整tensor、peak、N/M/Q；保留2D Gauss对照 |
| 径向数学 | 267独立高精度检查通过 | contained/clipped、active-pressure与原q24数值 port |
| 保守层次查询 | 8300检查通过 | k4/8/16/32真实三角集合、serial/cells/blocks、overflow、输入字节、book/saddle |
| 原作用点与 anchor | serial／cooperative 字节一致，含 signed ties、ULP、NaN、无pool、slots与reload | 原 source 属性与下游应力门分别检查 |
| 非凸 U／warped | 原U1060检查；post-fix serial U273检查；guard892+213+213检查通过 | 双侧、摩擦、远端材料、四种fallback、reset/checkpoint |
| open30 folded book | 2737检查、56 phases通过；376次错误远端raw load严格为0 | cap1/4096、四种fallback、七类生命周期；native完整K相对残差最大9.802e-9 |
| 稳定 Armijo | native22与独立400位Decimal71检查通过 | 只修 serial 压力目标差的浮点消减，不改迭代数／容差／物理律 |
| 最终公共与后端回归 | shell15 + volume11 = 26项通过 | F18与F19的genesis/tests逐字节一致 |
| Volume fused／separate | B4/B1024同一冻结输入的raw/full tensor/peak与原门通过 | 独立volume模型；不能继承shell空间准确性 |

open30 的独立 CPU q48 使用同一 K、同一 LU 最多三次残差修正，原1e-8门不变；
近闭合180° book 原模型条件数过高，原 CPU/native 均有残差拒绝，仍列为未合格对照。
最终 source 相对 v17 的接触变化仅为 serial Armijo 稳定代数及执行默认选择，
继承关系与精确源码差见 `R/meta/final-v19-source-inheritance.json`。

## 正式配对性能

<!-- PRIMARY_RESULTS_START -->
全部168个worker自然终态、RC0，48个配对block与16个cohort通过严格身份和运行门。
完整汇总为 `R/meta/primary-summary-v19.json.gz`；紧凑摘要绑定完整原文件SHA256，
下表使用原始600步墙钟，ms/batch为三个新进程的中位数。

### 冻结接触完整模块

| B | 输出 | Q4 ms/batch | planar ms/batch | volume ms/batch | planar env recoveries/s | Q4/planar，配对中位[范围] |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | max | 1.746 | 1.905 | 7.341 | 2,100 | 0.916 [0.915, 0.916] |
| 4 | full | 1.760 | 1.920 | 7.274 | 2,084 | 0.917 [0.917, 0.917] |
| 1024 | max | 7.957 | 4.537 | 64.676 | 225,685 | 1.754 [1.753, 1.757] |
| 1024 | full | 8.096 | 4.789 | 64.677 | 213,808 | 1.690 [1.690, 1.693] |
| 8192 | max | 53.141 | 26.435 | 480.030 | 309,892 | 2.010 [2.009, 2.011] |
| 8192 | full | 54.243 | 27.723 | 476.637 | 295,494 | 1.957 [1.956, 1.957] |
| 32768 | max | 201.912 | 101.248 | 1869.089 | 323,641 | 1.994 [1.994, 1.994] |
| 32768 | full | 208.016 | 107.187 | 1868.965 | 305,707 | 1.940 [1.940, 1.941] |

### 完整策略 inference workflow

| B | 输出 | Q4 ms/batch | planar ms/batch | volume ms/batch | pure rigid ms/batch | planar valid env transitions/s | Q4/planar，配对中位[范围] |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | max | 4.859 | 4.466 | 9.989 | 3.068 | 896 | 1.091 [1.087, 1.091] |
| 4 | full | 4.875 | 4.458 | 9.997 | 3.086 | 897 | 1.093 [1.087, 1.100] |
| 1024 | max | 11.948 | 7.559 | 71.027 | 3.742 | 135,461 | 1.581 [1.576, 1.583] |
| 1024 | full | 12.437 | 7.961 | 72.605 | 3.968 | 128,631 | 1.562 [1.562, 1.578] |
| 8192 | max | 67.705 | 34.477 | 506.117 | 10.076 | 237,609 | 1.964 [1.961, 1.965] |
| 8192 | full | 69.216 | 35.765 | 506.954 | 10.037 | 229,052 | 1.934 [1.931, 1.935] |
| 32768 | max | 254.720 | 125.316 | 2020.295 | 29.708 | 261,484 | 2.033 [2.032, 2.033] |
| 32768 | full | 263.243 | 132.392 | 2044.781 | 29.965 | 247,507 | 1.987 [1.987, 1.989] |


B32768时，冻结模块max/full的Q4/planar配对速度比为1.994/1.940，完整策略为
2.033/1.987。相同四项的volume/planar比为18.460/17.436/16.120/15.446；
volume属于独立弹性模型，这些只是运行成本比较。B4冻结模块约慢9%，完整策略约快9%；
不能将大批量收益泛化为小批量模块收益。所有正式进程invalid和overflow均为0。
<!-- PRIMARY_RESULTS_END -->

冻结计划 `R/meta/primary-pairing-plan-v3.json` 包含168个新进程、16个cohort与48个
配对 block。B=4/1024/8192/32768，module/policy与max/full各自比较；每个进程
warmup730，计时600步、1个repeat、minimum-seconds0，三个新进程配对重复。
该固定600步计时窗口与准确性用的711／1192步延迟完整周期分别记录，不互换步数范围。
module每block使用Q4/planar/volume三种顺序；policy另含pure rigid，采用预先固定的
三行部分Latin顺序，不能称为四位置完全平衡。每block在同一独占GPU分配内顺序执行。

全部设置 job-local GraphOff。原 Q4 已调优为 B4 q24、其余q16；planar为上述
k32/q10配置；volume使用已通过该策略 warmup 的angle/q128 fused路径。
正式benchmark的`cuda_graphs_enabled`记录进程环境；GraphOff由captured helper在导入前
设置0及固定Quadrants的导入语义共同证明，该字段没有直接读回库内部常量。
有界graph wrapper对照另有导入后常量的直接读回，两个证据来源分别保留。
`shell_recovery_graph_selected`表示共同恢复调用路径；在GraphOff时仍可为true，
不表示该进程实际执行了CUDA graph launch。
计时保留实际原运行时检查，研究 oracle 导出、内存采样与 profiler均另跑。
策略计时包含任务指标的device收集kernels与public position getter；指标的边界host导出
在计时结束后执行。该范围与纯inference算子速度或无指标的裸simulator速度不同。
表中速度比来自同block的墙钟时间，报告三个配对比的中位数和范围，不把它当作置信区间。

module使用真实脚本运行到730步后恢复的冻结接触，Q4/planar/volume的每环境
source point/F/normal/friction/R共12列float64身份在block内完全一致；volume的anchor
单独记录，因弹性表面不同而不强求一致。policy使用相同初始条件、控制器与实际权重；
stress进入观测后，不同后端的后续动作与接触可不同，不能声称共享未来contact tape。
Pure rigid无stress配置，max/full标签不赋予其应力输出语义。

预先冻结的实用平台规则是8192→32768的valid env/s增长≤10%；若超过该值且估计
65536 full内存低于设备90%，最多追加一个有界probe。有限批量范围不能证明全局最大吞吐量。

该内存条件已独立完成筛选：8192与32768的已终态full测量作两点affine外推，
`M65536=(7*M32768-4*M8192)/3`，Q4／planar预计process NVIDIA峰为
90.9375／103.4421GiB；对应设备容量94.9724GiB，90%阈值85.4752GiB。
两者均不满足预先规定的内存条件，因此不追加65536实验。Native stress库存自身的预测
25.8351／38.2660GiB是另一个范围，不能与process峰相加。实测采样峰是下界，但含负系数
的外推对目标值既不是严格上界也不是严格下界；这里只作为容量规划screen，不能证明OOM。
原始点、匹配GPU/PID与假设见 `R/meta/practical-batch-memory-projection-v19.json`。

## 批量扩展与实际路由

8192→32768的scaling有14个唯一条目：recovery/policy×max/full×Q4/planar/volume共12项，另有policy的pure rigid两项。每个rate先按 `(B*600-invalid)/elapsed` 在各新进程计算，再取3个rate的中位数；growth是两批量中位rate之比减一。recovery为env recoveries/s，policy为valid env transitions/s。

| scope | 输出 | variant | B8192 中位有效rate/s | B32768 中位有效rate/s | 增长 | ≤10% |
| --- | --- | --- | ---: | ---: | ---: | --- |
| recovery | max | Q4 | 154,157.037 | 162,288.159 | 5.275% | True |
| recovery | max | planar32 | 309,891.507 | 323,640.511 | 4.437% | True |
| recovery | max | volume | 17,065.601 | 17,531.531 | 2.730% | True |
| recovery | full | Q4 | 151,023.962 | 157,526.652 | 4.306% | True |
| recovery | full | planar32 | 295,493.675 | 305,707.317 | 3.456% | True |
| recovery | full | volume | 17,187.068 | 17,532.698 | 2.011% | True |
| policy | max | Q4 | 120,995.495 | 128,643.468 | 6.321% | True |
| policy | max | planar32 | 237,609.029 | 261,483.962 | 10.0479906723% | False |
| policy | max | volume | 16,185.994 | 16,219.410 | 0.206% | True |
| policy | max | pureRigid | 812,989.037 | 1,103,020.347 | 35.675% | False |
| policy | full | Q4 | 118,354.977 | 124,478.120 | 5.174% | True |
| policy | full | planar32 | 229,052.115 | 247,506.918 | 8.057% | True |
| policy | full | volume | 16,159.271 | 16,025.190 | -0.830% | True |
| policy | full | pureRigid | 816,171.156 | 1,093,531.383 | 33.983% | False |

policy planar max的精确增长为 **10.0479906723%，≤10%判定为False**；full为8.057%，判定True。Pure rigid仍增长33.983%/35.675%（full/max），其max/full标签没有应力输出含义。因此只能逐项报告预声明的近似实用平台标准，不能称所有路径已经平台化。不同batch来自不同allocation；该scaling不是跨batch同GPU配对加速比，也不证明全局最大吞吐量。

预声明的65536 full进程内存模型分别为Q4 **90.9375 GiB**、planar **103.4420572917 GiB**，均超过捕获设备容量90%的 **85.4751708984 GiB**；matched memory screen为False，未追加65536 probe。该外推保留原26/28内存快照及冻结准则，最终28/28内存资格另行绑定；它是有不确定性的实用筛选，不能作为严格OOM证明。

下表给出16个candidate cohort的原生route汇总，每行累加3个完整600步新进程。全部 `full_interval=True` 且计时区间内reset为0；工作负载的累计reset字段不当作区间reset。每行attempted与proxy逐值相等，fallback、overflow_steps、material_queries及material_rejections均为0；fallback频率按fallback/attempted计算，所有本表分母均大于0。

| scope | B | 输出 | attempted = proxy | analytic_sections（绝对数） | fallback |
| --- | ---: | --- | ---: | ---: | ---: |
| recovery | 4 | max | 18,000 | 0 | 0 |
| recovery | 4 | full | 18,000 | 0 | 0 |
| recovery | 1024 | max | 4,608,000 | 0 | 0 |
| recovery | 1024 | full | 4,608,000 | 0 | 0 |
| recovery | 8192 | max | 36,864,000 | 0 | 0 |
| recovery | 8192 | full | 36,864,000 | 0 | 0 |
| recovery | 32768 | max | 147,456,000 | 0 | 0 |
| recovery | 32768 | full | 147,456,000 | 0 | 0 |
| policy | 4 | max | 27,675 | 126 | 0 |
| policy | 4 | full | 27,675 | 126 | 0 |
| policy | 1024 | max | 7,084,800 | 32,256 | 0 |
| policy | 1024 | full | 7,084,800 | 32,256 | 0 |
| policy | 8192 | max | 56,678,400 | 264,192 | 0 |
| policy | 8192 | full | 56,678,400 | 264,192 | 0 |
| policy | 32768 | max | 226,713,600 | 1,056,768 | 0 |
| policy | 32768 | full | 226,713,600 | 1,056,768 | 0 |

16行合计 **958,900,950 attempted = 958,900,950 proxy**、fallback与overflow均为0；`analytic_sections=2,706,684`是解析section的绝对调用计数，不能解释为contact概率。物理fallback、容量溢出和material拒绝的独立负控制保留在各自资格回执中；本表的零计数只描述正式primary窗口。

上述三个表仅作已资格化primary的标量/计数派生，不建立新的物理或目标资格。完整raw向量仍由完整JSON、无损gzip与原始worker JSON保留。

## 内存与准备时间

<!-- MEMORY_RESULTS_START -->
28个内存worker全部终态并通过默认严格审计，独立解压与CSV身份校验通过；完整内容保留在 `R/meta/final-memory-summary-v19.json.gz`。以下数字使用GiB／MiB／KiB（2的幂），进程采样峰值为下界。

| B | 输出 | native Q4 GiB | native planar GiB | native volume GiB | planar sampled NVIDIA峰 GiB | planar process RSS峰 GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | max | 0.0471 | 0.0754 | 0.1079 | 2.271 | 2.584 |
| 4 | full | 0.0480 | 0.0762 | 0.1083 | 2.271 | 2.503 |
| 1024 | max | 0.2318 | 0.4582 | 0.3263 | 3.621 | 2.729 |
| 1024 | full | 0.4443 | 0.6706 | 0.4288 | 3.840 | 2.709 |
| 8192 | max | 1.5662 | 3.1487 | 1.8610 | 13.197 | 3.340 |
| 8192 | full | 3.2655 | 4.8479 | 2.6813 | 14.885 | 3.265 |
| 32768 | max | 6.1413 | 12.3731 | 7.1229 | 46.025 | 4.870 |
| 32768 | full | 12.9382 | 19.1700 | 10.4041 | 52.838 | 4.805 |

以下是B8192的实际调优配置：Q4 q16/cap32，planar q10/cap128。per-env列为该组总字节/B，包含固定标量counter。原surface组同时包含求积网格，因此两条路径的该组字节不同。

| 具名每环境组，KiB/env | Q4 max | Q4 full | planar max | planar full |
| --- | ---: | ---: | ---: | ---: |
| Response/output | 47.425 | 264.925 | 47.425 | 264.925 |
| Original contacts | 139.215 | 139.215 | 139.215 | 139.215 |
| Original fallback scatter | 8.563 | 8.563 | 34.250 | 34.250 |
| Planar source/tasks | 0.000 | 0.000 | 116.280 | 116.280 |
| Material bitmap/diagnostics | 0.000 | 0.000 | 56.404 | 56.404 |

| 固定具名组，MiB | Q4 max | planar max | 分类范围 |
| --- | ---: | ---: | --- |
| Original surface/curvature | 5.514 | 2.936 | 具名共享几何 |
| Planar geometry/J | 0.000 | 36.028 | 具名共享几何，已处理相等指针alias |
| Other geometry/operators | 36.697 | 36.697 | 未完成sharing分类，不作全共享声明 |

| 构造观测（含部分JIT） | worker数 | setup中位s | setup范围s |
| --- | ---: | ---: | ---: |
| Q4 | 8 | 68.877 | [18.916, 124.193] |
| planar | 8 | 26.303 | [22.238, 30.448] |
| volume | 8 | 102.341 | [53.537, 154.704] |
| pure_rigid | 4 | 44.387 | [42.418, 46.600] |

setup在科学库导入和Genesis初始化之后计时；process采样则从导入之前开始。上述setup包含不同编译／缓存状态，不作为独立proxy准备时间或其加速比。弹性factor/inverse/operators与原surface分项timer完整保存在lossless内存汇总。

| Pure rigid control | sampled NVIDIA峰 GiB | process RSS峰 GiB |
| --- | ---: | ---: |
| B4 | 2.162 | 2.571 |
| B1024 | 3.043 | 2.725 |
| B8192 | 9.150 | 3.279 |
| B32768 | 30.135 | 4.673 |

代理路径增加了具名共享几何和每环境workspace，完整大批量也使用更多进程显存。这里的收益需要结合正式墙钟吞吐量评估；该路径不是内存赢家。Volume来自独立memory campaign，不把跨campaign的GPU样本称为同GPU配对内存测量。
<!-- MEMORY_RESULTS_END -->

内存使用独立新进程，覆盖同四个batch的Q4/planar/volume max/full与pure rigid control。
采样器在科学库导入之前独立启动，仅跟踪自己parent PID；请求50ms轮询，实测间隔另存。
NVIDIA采样峰值是可能漏过短峰值的下界，RSS为全进程高水位。两者包括JIT、CUDA context、
rigid/collider/controller/reset等，不能等同于stress数组字节。

native stress按暴露tensor view的**相等指针**去重，另列共享固定几何、每环境contacts、
fallback／proxy workspace与response/output；不同offset的view缺少完整backing-storage
区间，不能据此声称做过所有alias的严格去重。`other`组也不能统称共享。
setup计时包括整个构造、分配及部分JIT，另存弹性构造和原surface component timer；
没有独立测量纯proxy准备时间。不同任务顺序／缓存状态的setup不能直接解释为proxy加速。
内存worker中的速度均属intrusive诊断，排除在正式性能表之外。

## 剖析与逐族优化决策

<!-- PROFILE_RESULTS_START -->
最终14个worker均自然终态、原运行时门通过。以下为planar32冻结模块的实际kernel duration，单位µs/次恢复；每列10次。

| 实际kernel类别 | B4 max | B1024 max | B8192 max | B8192 full |
| --- | ---: | ---: | ---: | ---: |
| Anchor | 88.99 | 285.34 | 1624.54 | 1635.18 |
| 原曲率／半径关联 | 260.86 | 221.85 | 534.68 | 537.01 |
| 材料证书 | 9.60 | 14.24 | 58.31 | 57.64 |
| 候选查询 | 634.78 | 713.81 | 2492.27 | 2497.85 |
| 一次性task几何准备 | 32.79 | 52.08 | 330.42 | 335.34 |
| 加载矩积分 | 62.98 | 1280.46 | 10713.42 | 10740.60 |
| 非负压力拟合 | 46.38 | 50.70 | 143.16 | 144.52 |
| Jᵀ载荷传递 | 9.74 | 13.94 | 70.42 | 71.13 |
| 代理验证 | 51.34 | 60.55 | 370.89 | 369.34 |
| 原加载kernels／空fallback准备 | 162.58 | 170.28 | 272.13 | 271.37 |
| 联合balance／response／stress-output | 485.42 | 1581.16 | 9387.64 | 10811.74 |
| 独立acceptance／state | 29.06 | 28.86 | 38.93 | 89.02 |

原加载kernel在proxy成功时仍含初始化与空任务开销；该时间不代表发生fallback。
同列Q4/planar所有kernel duration和分别为：
B4 max 1.728/1.875 ms。
B1024 max 7.891/4.473 ms。
B8192 max 52.216/26.037 ms。
B8192 full 53.786/27.561 ms。

完整策略另有6份10步trace。下表同为intrusive事件时长/nominal step，非计时器墙钟；单位ms。

| B | 路径 | 联合shell恢复或volume后处理 | rigid约束求解 | 其余GPU kernels（含独立加载） | GPU copies | host CUDA API事件和 | kernels/step |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | Q4 | 0.475 | 1.349 | 2.181 | 0.021 | 0.950 | 324.2 |
| 4 | planar32 | 0.476 | 1.394 | 2.245 | 0.028 | 0.945 | 350.8 |
| 4 | volume | 7.483 | 1.331 | 0.595 | 0.012 | 0.819 | 317.6 |
| 1024 | Q4 | 1.438 | 1.887 | 6.223 | 0.021 | 0.905 | 327.1 |
| 1024 | planar32 | 1.483 | 1.946 | 3.685 | 0.027 | 0.944 | 353.7 |
| 1024 | volume | 61.713 | 1.853 | 0.777 | 0.012 | 0.877 | 320.5 |

rigid约束求解按实际rigid kernel名识别，未混入stress response。Shell列的联合恢复不含其独立加载阶段；volume联合post含rigid后处理／加载／恢复，二列范围不同。host API事件与GPU事件可能重叠，不相加解释latency；完整kernel名和分类规则保存在lossless `final-profile-summary-v19.json.gz`。
<!-- PROFILE_RESULTS_END -->

Device duration之和可能重叠，不能相加当成scene latency。shell `kernel_recover` 将
response、stress/output与部分检查融合；volume `kernel_rigid_stress_post` 联合rigid后处理、
加载与恢复，无法从该trace独立拆出所有子阶段。保留原始kernel名、次数与时间；
不以墙钟减device时间来编造launch/synchronization成本。Policy trace为正式600步之后
额外10步，原始reset计数分别为B4的1与B1024的256（一次控制器reset wave），
其时间不进入正式吞吐量。

| 优化族 | 结论与匹配证据 |
| --- | --- |
| 固定几何、compact maps、offline refinement | 保留QD紧凑cell/grid与J，不复制每环境三角几何；elastic DOF不变 |
| material-aware候选与bounds | 保留实际边连通证书、.005R guard和4×4 blocks；8300覆盖检查及最终bank通过 |
| 跨步witness／support复用 | 保留已有provenance和步内G/H／clip复用；实际moving输入五次快照均不同，未新增未经命中率证明的跨步cache |
| contained解析与正压力 | 保留独立资格分支；真实频率随三角大小／footprint变化，不能假定k32全部contained |
| clipped integration／nonnegative fit | 保留解析radial、角向分段及active-pressure；2D Gauss保留为私有数值对照 |
| empty／zero／fit-scatter复用 | 保留原零任务剔除、每task一次clip与共享G/H；不合并不同source的pressure语义 |
| packing／布局／grouping | 保留source packing、连续task ranges与warp协作；原batched response不改空间 |
| 固定缩小launch grid | W512/2048/4096均通过数值但B1024为26.61/16.09/13.80ms，慢于11.386ms；移除生产增量，源码保留`a26cc3e93` |
| cooperative anchor | 保留CUDA packed warp，CPU／无pool和volume inline anchor仍serial；Q4模块B4由3.938→2.868ms，B1024由9.064→8.264ms |
| batched response／max-full fusion | 保留原Q4 inverse和fused recovery；不以采样点替代global sampled max，不丢full输出 |
| CUDA graphs／device scheduling | 对当前负载有界测量后选择本任务GraphOff；实际开关另存，不声称新proxy kernel已经持久graph capture |
| volume fused scheduling | angle/q128在B4 module/policy分别7.370/9.543ms，unfused+coops为7.743/10.602ms；B1024 module差0.24%单次不足称收益，保留fused |
| capacity／fallback／batch scaling | 128与256速度相近，B1024 native 467.83/586.83MiB；完整B64真实峰63，选128；cap1是proxy与原face同时受限的双重overflow最坏控制 |
| 既有Atlas/H/FP32/screen/history | 保留已交付的有界负面结论与恢复pin；本次不因目标扩展而重跑未变机制 |

组合前后的细分证据见 `hierarchical-tuning-v17`、`graph-policy-control-v18`、
`volume-q128-scheduling-v18` 及相应 `R/meta/*summary*.json`。
cell→blocks在k32 cap128的module B4从2.991→2.164ms、B1024从5.752→4.758ms，
policy B4从5.786→5.111ms、B1024从9.115→8.246ms；这是screening A/B，最终收益以上述
正式配对表为准。GraphOff对照也分别比较同后端、同source与实际执行模式，不把不公平
on/off混用作为最终速度比。

## 保留的反例和边界

- k8完整B4的peak1.085974%超过1%严格目标，且曾有420次fallback／9223 contacts。
  k16完整B64的peak0.323185%和M0.783200%超过0.3%／0.75%；选k32时未改预算。
- 原`.01R` warped guard曾让raw load通过而field0.3900%／peak0.3668%超出0.1%
  fallback等价预算。改进的是geometry admission为`.005R`，旧失败保留。
- 原Q4 q10真实frame585压力拟合失败仍保留；volume edge/q16实际策略tick606同样在
  未改动起点复现失败。最终volume用通过控制的angle/q128，不把失败帧排除来计速。
- body-origin相对矩诊断有B4两次／B64九次旧失败；逐source原作用点门和同fit raw
  重建见证通过。原诊断仍标失败，平移误差用实际lever arm传播解释，未放宽生产门限。
- 旧book使用粗parent连接会向球外材料片加载；open30原反例错误远端力为source的
  4.44–8.37%，新路径该分量严格为0。更极端近闭合book的原K残差负例仍未合格。
- 已取消的两个旧book任务只因同源码压力目标消减错误有独立因果证据及修复资格，
  取消前的30对输出和140份NPZ身份保留。它们没有伪造完成RC，也不充作完整资格。
- shell in-plane stress与volume Cauchy stress分别属于自己的离散模型；volume计时对照
  不证明相同应力场。夹爪继续理想锐边，本文不声称连续体接触峰值或任意非凸几何收敛。

<!-- DELIVERY_RESULTS_START -->
## 证据与复现

[证据包](../evidence/20261027-planar-loading-v1/README.md)在10MiB限制内，
`requirements-ledger.json`逐条记录29项要求、14个优化族、保留的失败及适用边界。
`index.json`是带压缩／解压字节数和SHA256的小型引用；`index.json.gz`无损保留完整540条任务索引与全部字段。
原168份repeat及每环境向量无损保留，完整primary／memory／profile汇总以原始gzip保存；
大型场数组与日志按数据路径、字节数和SHA256保留在外部清单。
独立核验从原始计时重算48个block、16个cohort及其全部配对比值，
并检查实际权重、GraphOff身份、模块冻结接触与原始输入。
`--check-originals`还逐字节检查168份外部worker JSON与包内原始记录。
核验保留此前科学资格，不在此重新计算应力场或将退出码当成空间精度证明。

完整primary原文件为1,164,302,888字节，SHA256为
`96c1a03855499c7690f070789b207d0d4b05c77805d2420a7dbf3d980428c30c`；
原无损gzip为4,305,257字节。核验器按task流式读取，避免一次展开全部大文件。
Memory与profile各有28／14个合格worker，完整无损摘要分别保留。
[最终包核验回执](../evidence/20261027-planar-loading-v1-verification.json)绑定实际包的
index与SHA256SUMS身份；该回执位于包外，避免自引用hash。

以下只需Python标准库，可独立核验包内原始记录。`PLANAR_AUDIT_DIR`须为新的data目录。

```sh
export PLANAR_PACKET="$(git rev-parse --show-toplevel)/research/rigid_stress/evidence/20261027-planar-loading-v1"
export PLANAR_AUDIT_DIR=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/audits/planar-v1-replay
python3 - <<'PY'
import gzip, hashlib, json, os, subprocess, sys
from pathlib import Path
packet = Path(os.environ['PLANAR_PACKET']).resolve()
out = Path(os.environ['PLANAR_AUDIT_DIR']).resolve()
assert not out.is_relative_to(packet)
out.mkdir(parents=True, exist_ok=False)
for line in (packet/'SHA256SUMS').read_text().splitlines():
    digest, name = line.split('  ', 1)
    path = (packet/name).resolve()
    assert path.is_relative_to(packet)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
inventory = [json.loads(line) for line in gzip.decompress((packet/'external-artifacts.jsonl.gz').read_bytes()).splitlines()]
rows = [r for r in inventory if r['path'].endswith('/meta/verify-planar-delivery-packet-v19.py')]
assert len(rows) == 1 and len(rows[0]['retained_payloads']) == 1
row = rows[0]
code = gzip.decompress((packet/row['retained_payloads'][0]).read_bytes())
assert len(code) == row['bytes'] and hashlib.sha256(code).hexdigest() == row['sha256']
verifier = out/'verify.py'
verifier.write_bytes(code)
subprocess.run([sys.executable, str(verifier), '--packet', str(packet), '--output', str(out/'verification.json')], check=True)
PY
```

科学重跑使用index中每task实际module／参数、captured worker SHA与对应engine pin；
在Slurm `rtx-mid` / image `1_26`中选择新的data输出和私有cache，保留原条件fixture。
包内包含两份精确输入：small4 3,754字节和heterogeneous64 25,834字节。
完整worker capture与外部场数组仍在R中，不在login node执行科学模块。
原始packager可从提交 `628a169c298689d2f9913797d22bcb3628e0dda9`
恢复；在交付checkout根目录调用该版本打包器，测量engine worktree使用独立的F19 pin。
最终文档提交的HEAD与测量引擎pin分别记录。

540条归档任务的执行状态为503个RC0、32个历史非零退出、2个有取消记录但无task RC，
另3个仍保留原始“缺少task RC”标签：anchor-contract-v1的两项在提交前被setup检查拦下，
radial-math-v3因任务／源码身份不匹配取消。其原始原因与修复后的v2／v4资格分别保留，
这些旧尝试不充作通过证据，也不改写成终态成功。正式168项全部有RC0与独立资格。

保存检查覆盖12份原有未完成文件、33份旧与31份本阶段冻结源码及36份本任务Python源码，
原内容与冻结身份均保留。未改动无关任务，取消的旧book任务只记录真实取消状态。
原Q4仍为公开默认；完整异质策略推荐显式选择本文k32配置。
<!-- DELIVERY_RESULTS_END -->
