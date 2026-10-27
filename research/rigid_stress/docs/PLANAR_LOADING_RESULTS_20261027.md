# Planar loading DPR：原生实现与测量，2026-10-27

**审阅稿：正式性能与内存 campaign 尚未全部终态，目标仍未关闭。**
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
每环境仅保留有界接触／任务 workspace。CPU 只提供独立 oracle 和整数拓扑准备，
生产浮点准备与运行仍在现有 rigid stress feature 内的 Quadrants 中执行。

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
正式 campaign `primary-paired-v19` 尚未全部终态，当前不发布最终总表或全局资格。
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

## 内存与准备时间

<!-- MEMORY_RESULTS_START -->
独立内存 campaign 仍待全部28个worker终态，当前不发布最终内存资格与总表。
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
完整requirements ledger、紧凑可运行证据包、最终保存审计和远端完成状态将在所有正式
任务终态并核验后写入；本审阅稿不关闭目标。
<!-- DELIVERY_RESULTS_END -->
