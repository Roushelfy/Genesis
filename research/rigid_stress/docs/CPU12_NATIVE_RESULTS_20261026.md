# CPU12 native adaptation：阶段结果，2026-10-26

本文记录 [CPU12 目标](ATLAS_CPU12_GPU_PARITY_GOAL_20261026.md) 第二阶段中已核实的原生空间、轨迹、几何控制、intrusive profile、完整 workflow 和独立内存。**这不是整个目标的完成声明。** Volume 的指定 Franka 轨迹通过；shell 仍有结果量误差反例，非凸颈部远端热点仍未普遍合格。完整 workflow 中 Atlas 比优化 Mother 慢，大批量内存也更高，两后端最终默认推荐 Mother。本批有界数值／代价实验已终态，无需追加科学任务；证据封装、保存审计和整个目标的关闭由主任务处理。

## 来源、存储和适用范围

运行数据根目录记为 `R`：

```text
/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261026-cpu12-gpu-parity
```

| 阶段 | 精确源码 revision | 冻结源码／数据目录 |
|---|---|---|
| 成熟局部空间的起点 | `822ad285497f79d3a24f3ce2f0dd7a58ec6cd699` | 原 5+9 integrated DPR 实现，保留既有数值与生命周期约定 |
| native-v0 | `a60f8ec39eae03737c852a96a939a6bb025ce88e` | `/mnt/home/zhaofeng/workspace/Genesis/.worktrees/cpu12-native-v0`；`R/native-v0` |
| native-v1 / native-core-v1 | `6f2e646090c7dd995ca51fa3ca6cbac76f81482a` | `/mnt/home/zhaofeng/workspace/Genesis/.worktrees/cpu12-native-v1`；`R/native-v1`、`R/native-core-v1` |
| native-v2 / native-final-qualification / workflow-final / workflow-memory / native-final-profile | `94ec782fd0754017074cf3b567b391b1444f7172` | `/mnt/home/zhaofeng/workspace/Genesis/.worktrees/cpu12-native-v2`；`R` 下同名 campaign |

各 campaign 的 `source.json`、`source.patch` 保存引擎身份；上述捕获时 patch 均为空。研究脚本另存于各自 `analysis-source/research/rigid_stress/qipc_comparison/`，文件身份见 `analysis-files.json`，不要假定引擎 revision 同时包含这批研究脚本。`tasks.json`、`job.sh`、`workload-*.sh`、`logs/` 保存实际命令及执行记录。数值执行使用 Slurm `rtx-mid` / image `1_26`；本文整理仅使用已存 JSON 和源码，无新科学执行或数据迁移。

native-v0 在 QD kernel 参数默认值处启动失败，未进入数值实验；保留为启动失败记录，不当作模型精度反例。native-v1 两后端的 `*-spaces/result.json` 数值控制通过；基准 c1h0 的 600 步轨迹 mean field 分别为 shell **19.020%**、volume **15.733%**，均未获空间资格。

原 CPU oracle 为 `a10c5abe3b5bc3c5caa57d3d4d9dadab2b277ade`。这里复用十二方向身份，**没有复用 CPU 的母几何、P/B、材料球冠载荷或 CPU 解**。Shell 采用原生 midsurface 与旋转自由度，volume 采用原生 P1/P2 体网格；每个候选均与自己未改动的母模型比较。CPU/GPU 同空间等价与这里的原生空间资格是不同结论。

## 有界机制和控制范围

- **中心与拓扑：**默认十二个固定 icosahedral 方向。QD 用实际 load-boundary 的 bbox 中心和逐轴半径归一化物理坐标，逐中心选择未占用的真实边界 seed，再沿原材料图做 Dijkstra。保留凸／非凸材料连接关系，不按空间邻近跨接材料片。FPS6 是独立密度控制，不能称为 CPU 十二空间。
- **局部空间与 halo：**令 `rho_i` 为图 Voronoi 区域半径，紧支撑半径 `s_i=1.2*max(rho_i,1e-12)`；初始 local 半径为 `core_scale*s_i + 2*transition + longest_edge`，按完整单元纳入，再加声明的 halo。使用既有 `T_i=[P,B_i]`、小算子、共同 relief/gauge、一次 spin 及一次合并 tensor recovery。
- **选择和宽度：**continuous 使用 Wendland `W(d/s)`；nearest 每 contact 取图距离最近者且并列取首；narrow 再乘 `W((d-d_min)/(fraction*s))`，保留所有正贡献者。positive finite 总权重先精确归一再乘 coverage availability；零／非有限总权重归母空间。保持原 finite-footprint coverage、mother remainder 和数值 fallback 门，不用无条件全母修正掩盖空间误差。
- **有界参数：**已试 core 1.0、1.4、1.6，shell 另试 1.8、最后 2.0；halo 0/1；12 方向与 FPS6；continuous/nearest/narrow；narrow fraction .12/.06。`atlas_transition` 在这些记录中固定为 **0.003 m**，narrow fraction 和 nearest 提供选择过渡宽度控制，不声称扫描过多个米制 transition。此次局部／halo／密度／选择宽度／全局 coarse 控制到此截止，不承诺穷举或继续无界调参。
- **全局 coarse：**研究选项 `--global-enrichment surface` 经既有 `atlas_basis/fixed_nodes` 接口加入全部 shell 三 parent face-center，或由完整 tet 面 incidence=1 推导的全部材料边界 P2 midpoint，涵盖内外边界及非凸拓扑。只在 CPU 推导整数 ID；几何、基函数、算子等浮点准备与运行仍在 QD。不是仅提升选定的外表面，也不改物理网格。`global_enrichment_actual` 如实区分 reduced/full-size feature；全母控制不得称为缩减。

`R/native-core-v1` 的 core1.4h0 continuous/narrow/nearest 三策略全部未合格：shell mean field **6.470/6.624/7.555%**，volume **10.617/10.783/12.454%**。core1.6h1 narrow 的 shell mean **4.325%**，但加权 M/Q 最坏 **15.600/23.552%**；volume mean **2.367%** 且通过。全局 surface coarse 在下述 c1h0 配置也未合格；这些控制不能证明单独提升 coarse 已解决误差。

## native-v2：四条件轨迹

表中 G 为 `R/native-v2/groupG`，其实际子目录名与配置一一对应，例如 G2 为 `volume-none-c1.6-h1-narrow-f0.12-r12`。全部为 level1、FP64、600 步、full 输出、每步更新，含 reset/checkpoint 生命周期；默认 edge/q16。条件文件：

```text
/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/hard-finite-edge-franka-boundary-v2/group0/franka/conditions.npz
```

预算为 area/volume-weighted tensor Frobenius field ≤10%、sampled peak ≤5%、loaded environment-frame mean field ≤5%；shell 的旧 Euclidean 与加权 N/M/Q 均须 ≤10%。对称 tensor 剪切项权重为2，Q 为普通 vector 范数。母空间残差仅是空间诊断，不代替小空间数值门，也不作为修正载荷。

| G／配置 | mean field | worst field | worst peak | 资格 |
|---|---:|---:|---:|---|
| 0 shell surface c1.0h0 narrow .12 r12 | 16.955% | 32.604% | 9.836% | field、peak、N/M/Q、mean 失败 |
| 1 volume surface c1.0h0 narrow .12 r12 | 15.802% | 47.922% | 4.333% | field、mean 失败 |
| 2 volume none c1.6h1 narrow .12 r12 | 2.367% | 5.464% | 0.987% | **通过** |
| 3 shell none c1.8h1 narrow .12 r12 | 2.262% | 8.143% | 0.874% | **M/Q 失败** |
| 4 volume none c1.6h1 nearest .12 r12 | 2.639% | 5.464% | 1.015% | **通过** |
| 5 volume none c1.6h1 narrow .06 r12 | 2.388% | 5.464% | 0.985% | **通过** |
| 6 volume none c1.6h1 narrow .12 FPS6 | 0.391% | 2.050% | 0.0083% | 通过，但缩减弱 |

全部7组完成600步及生命周期断言，各有8538个 loaded contact-env，coverage fallback 为0，grouped-load 相对误差最大 `4.36e-16`。G3 加权 M/Q 最坏 **11.946/20.711%**：M 超标4帧，旧 Q 超标455帧、加权 Q 超标430帧。其 mean、field、peak 通过不能覆盖结果量失败。

| 原生空间 | feature raw scalar DOF | 母 DOF | reduced / full-size feature |
|---|---:|---:|---:|
| G0 shell surface | 912–960 | 1452 | 12 / 0 |
| G1 volume surface | 1533–1632 | 2430 | 12 / 0 |
| G2/G4/G5 volume c1.6h1 | 1911–2244 | 2430 | 12 / 0 |
| G3 shell c1.8h1 | 1032–1200 | 1452 | 12 / 0 |
| G6 volume FPS6 | 2208–2430 | 2430 | **2 / 4** |

FPS6 有4个 feature 已等于母空间，精度改善不能解释为同等缩减下的优势。轨迹 driver 保存 grouped-load、母空间诊断和 feature residual，并执行原运行时 errno/reference-valid 门；没有独立保存每步力矩守恒资格布尔。下面 geometry/profile 提供显式独立守恒检查。

## native-v2：egg／非凸 neck 控制

在上述 G0–G3 配置目录后加 `-egg/unfixed/result.json` 或 `-neck/unfixed/result.json`。下表只统计 `category=="qualification"`；mean 排除释放帧，包含 loaded qualification environment-fields。

| G／geometry | mean field | worst field | worst peak | 实际 false 检查 |
|---|---:|---:|---:|---|
| 0 shell surface egg | 16.896% | 25.218% | 3.418% | spatial：field、N/M/Q；mean；partial footprint |
| 0 shell surface neck | 56.940% | 89.495% | 68.688% | spatial：field、peak、N/M/Q；mean |
| 1 volume surface egg | 17.253% | 19.642% | 2.600% | spatial：field；mean；partial footprint |
| 1 volume surface neck | 22.048% | 53.723% | 19.607% | spatial：field、peak；mean；partial footprint |
| 2 volume c1.6h1 egg | 1.950% | 2.113% | 0.036% | **仅 partial footprint** |
| 2 volume c1.6h1 neck | 2.519% | 40.634% | 27.242% | spatial：field、peak；partial footprint |
| 3 shell c1.8h1 egg | 2.121% | 3.060% | 0.162% | **仅 partial footprint** |
| 3 shell c1.8h1 neck | 23.456% | 161.413% | 125.187% | spatial：field、peak、M/Q；mean；partial footprint |

全部8套的 `numerical`、permutation、release、restore、aggregation、opposed-source sum、continuity、junction、reduction 均通过，所有 neck 的 aggregate remote witness 也通过。没有 numerical fallback；归一化力误差最大 `5.82e-16`、矩误差最大 `2.83e-14`，grouped-load 最大 `1.47e-16`。数值门保留 source-space unscaled residual `max(1e-11,1e-8*rhs_norm)`、state validity、pressure status 和力／矩 `1e-8` 守恒门。

**partial footprint 缺见证与空间失败须分开。** 仅 G0 shell surface neck 找到真实部分覆盖载荷：14次 coverage fallback，probe mother weight 约 `2.213e-4`。其余7套 probe 明确 `locatable=false`，原因是未找到有 nominal feature 的 supported/unsupported 相邻面边界；这是该配置未完成要求的部分覆盖见证，不是观测到丢载或守恒失败。G2 volume egg 和 G3 shell egg 的空间预算全部通过，整体失败仅来自这一见证要求；仍保留整体未合格状态。

**远端热点反例仍然成立。** G2 volume neck 的实际空间失败仅在 `remote_neck_0/1`：field **13.504/40.634%**，peak **3.354/27.242%**。该几何已有10/12个 full-size feature，仅2个缩减，仍不能用小误差均值或多数 full-size 空间覆盖远端峰值失败。G3 shell neck 加权 M/Q 最坏 **161.413/647.773%**，N通过。`remote_neck_witness=true` 只证明母模型颈部存在离接触超过两个半径的热点，不证明候选精度。

已核对捕获的 `check_local_geometry.py:806–839`：最终 `passed=all(qualification_checks.values())`，partial footprint 要求 `probe.locatable && observed_mother_weight>0`。`:1097` 将顶层 `complete=true` 设为流程与保存完成，`passed` 取最后 qualification variant（本批均 `unfixed`）。因此 `complete:true, passed:false` 不等于每套都发生空间误差超预算。

## native-final-qualification：扩展终态

此 campaign 与 native-v2 使用相同引擎 pin，三组均已终态、600步、full 输出并通过生命周期断言；这些结果不替代上面的 egg/neck 控制。异质64条件来自：

```text
/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261025-gpu-throughput/meta/heterogeneous-conditions.npz
```

| `R/native-final-qualification` 下的路径 | 配置 | mean field | worst field / peak | 加权 M / Q 最坏 | 资格 |
|---|---|---:|---:|---:|---|
| `group0/shell-heterogeneous64/result.json` | c1.8h1、12 narrow .12、edge/q16 | 1.748% | 10.236 / 0.941% | 14.840 / 24.239% | field 1帧、M/Q 超预算 |
| `group1/volume-heterogeneous64/result.json` | c1.6h1、12 narrow .12、angle/q128 | 2.024% | 6.515 / 1.462% | 不适用 | **通过指定轨迹** |
| `group2/shell-c2-h1-last-wide-control/result.json` | 四条件、c2h1、12 narrow .12、edge/q16 | 1.868% | 6.664 / 0.779% | 9.806 / 17.396% | **Q 失败** |

异质64条件的 shell/volume 各有88099个 loaded contact-env，最后 shell wide control 有8538个；三组 coverage fallback 均为0，grouped-load 最大 `4.65e-16`。异质 shell 旧／加权 M 超标39/41帧、Q超标478/475帧；最后 c2h1 旧／加权 Q 超标77/74帧，M、N、field、peak、mean 均通过。

三组全部是真实缩减的12 feature：shell c1.8h1 为1032–1200 DOF / 母1452，volume 为1911–2244 / 母2430，shell c2h1为1068–1284 / 母1452。不存在该三组偷偷退化成 full-size feature 的解释。Volume 轨迹通过的载荷积分资格绑定 angle/q128，不能挪给 edge/q16 或任意几何使用；shell 的最后更宽局部空间仍未解决 Q，至此结束这批有界调参。

## Intrusive profile：解释瓶颈，不作吞吐量结论

`R/native-v2/group7/profile-volume/result.json` 和 `group8/profile-shell/result.json` 均为 B64、module/max、warmup130、采样10次，`primary_timing=false`。

| 配置 | 主类别累计 device-duration / recovery | 占已记录 device 总和 | 次要类别 |
|---|---:|---:|---|
| volume c1.6h1、none、12 narrow .12 | solve/response **4.759 ms**；其中 packed kernel **4.283 ms** | 63.67% | acceptance 1.232 ms；pressure/scatter .623 ms |
| shell c1h0、surface、12 narrow .12 | pressure/scatter **3.063 ms** | 43.36% | solve/response1.564 ms；acceptance1.130 ms |

这是10次 trace 的累计 device-duration 除以10；CUDA task 时间可能重叠，不等于端到端延迟或加速比。两组输入复现、数值门、守恒通过；invalid、spin-invalid、coverage/numerical fallback 均为0。最大 residual/gate 分别 `8.31e-4`、`5.53e-5`。Shell profile 对应的空间配置未获精度资格，不能给出同精度速度结论。引擎均实际使用 packed；mother group 未打包，原数值门和失败 fallback 保留。

补充最终候选的 `R/native-final-profile`，两组均 rc0、`complete=true, passed=true`，warmup130、max、10次 intrusive trace：

| group／最终配置 | 主类别 device-duration / call | 次要类别 | 主 kernel |
|---|---|---|---|
| `group0/shell-final-module/result.json`：shell c1.8h1 module B64 | pressure/scatter **3.102 ms，41.03%** | solve/response1.980 ms，26.19%；acceptance1.125 ms，14.88% | packed solve1.506 ms；anchor1.059 ms |
| `group1/volume-final-scene/result.json`：volume c1.6h1 scene B1024 | solve/response **22.014 ms，69.90%** | acceptance2.785 ms，8.84%；pressure/scatter2.290 ms，7.27% | packed solve **20.700 ms** |

这些最终 profile 的 original runtime、力／矩门和输入复现通过，invalid、spin-invalid、coverage/numerical fallback 均为0；归一化力／矩误差均不超过 `4.2e-14`，最大 residual/gate 不超过 `.00130`。Volume 的主要新增成本在打包小空间求解；shell 仍有显著接触／压力成本。两组 scope 和 B 不同，不作跨后端速度比较，也不以 profile 代替下述完整 workflow。

## workflow-final：完整推理 rollout 与 frozen recovery

全部 **32 task** 已终态，逐 task 的 `.returncode.txt` 均为0。精确路径由 `R/meta/workflow-execution-map.json` 的32条 `rows` 映射：`actual_output` 为 **JSON 文件** `R/workflow-final/groupN/<taskname>.json`，不是目录。该 campaign 与 native-v2 使用同一引擎 pin；`source.json`、`analysis-files.json` 和逐任务 `source_sha256` 保留引擎／脚本身份。

完整 policy workflow 包含 observation、policy inference、原生 controller、rigid step、接触／DPR 和所选 max/full stress output。Policy 是固定 seed99173 的随机 MLP `[26,128,128,7]`，FP32、`.eval()`、`torch.inference_mode()`，不是训练、反向传播或优化器更新；residual action 尺度 `1e-4 rad`，warmup 对 FP64 policy 的控制差保留 `<1e-9 rad` 门。Scene/state 为 FP64，dt=.01、substeps1、history0、edge/q16。

每方法各3个重复；policy 每重复600步，warmup130。每个重复从 saved initial state restart，restart 和同步发生在计时开始之前，区间包含最终同步；计时内没有 reset。不同方法按任务块执行，按 repeat index 0/1/2 配对，**不是每个重复交错 A/B**；每组方法的 `gpu_uuid` 完全一致，GPU 为 NVIDIA RTX PRO 6000 Blackwell Server Edition。表中取每方法3个重复的 median：`ms=1000*seconds/steps`；速率为 `transitions_per_second=(B*steps-invalid_environment_steps)/seconds`，是原始数值有效 env-step/s。全部96个重复 invalid 与 scatter overflow 都为0。

优化 Mother 实际导出 `method_selected=inverse`、`surface_inverse_selected=true`、`inverse_precision=64`。Atlas 使用12原生空间，volume c1.6h1、shell c1.8h1，narrow .12、transition .003 m、global none；实际有 `stress.0.features.packed.*` buffer，保留原 factor residual/fallback 门（rtol `1e-8`、atol `1e-11`）。Atlas 顶层 `method_selected=direct` 描述其母模型，feature solver 为 packed，不应据此前者误判没有打包。

各表单元为 **median ms/step / 原始有效 env-step/s**；最后一列为 Atlas/Mother median 时间比，越大表示 Atlas 越慢。

| policy group | pure rigid | optimized Mother | Atlas packed | Atlas/Mother |
|---|---:|---:|---:|---:|
| volume L1 B4 max | 2.709 / 1476.4 | 5.060 / 790.5 | 13.498 / 296.3 | 2.668× |
| volume L1 B4 full | 2.797 / 1430.0 | 5.120 / 781.2 | 13.528 / 295.7 | 2.642× |
| volume L1 B1024 max | 3.365 / 304315.9 | 9.118 / 112307.3 | 34.846 / 29386.7 | 3.822× |
| volume L1 B1024 full | 3.469 / 295211.5 | 9.442 / 108447.0 | 35.618 / 28749.6 | 3.772× |
| shell L1 B4 max | 2.780 / 1439.0 | 6.857 / 583.3 | 12.331 / 324.4 | 1.798× |
| shell L1 B4 full | 2.673 / 1496.6 | 6.637 / 602.7 | 11.836 / 338.0 | 1.783× |
| shell L1 B1024 max | 3.383 / 302679.4 | 13.363 / 76631.2 | 24.171 / 42364.2 | 1.809× |
| shell L1 B1024 full | 3.374 / 303484.5 | 13.568 / 75472.2 | 24.387 / 41989.3 | 1.797× |

依上述顺序对应 group0–7；文件 taskname 为 `policy-{backend}-L1-B{B}-{mode}-{pure_rigid|mother|atlas_packed}.json`。方法块顺序在不同 group 中轮换，精确顺序见 execution map / `tasks.json`，不把这种轮换称为交错重复。

Frozen recovery 为 warmup 生成的真实接触 snapshot 和匹配 pre-integration frame，只计独立 stress-recovery 调用；准备／JIT 在计时外。每重复100次、3重复、B64，不含 policy、controller 或 rigid step，因此没有 pure-rigid recovery 数值。下列速率单位是重复 snapshot 的有效 env-recovery/s，不是动态轨迹速率。

| recovery group | optimized Mother | Atlas packed | Atlas/Mother |
|---|---:|---:|---:|
| volume L1 B64 max | 1.098 / 58303.0 | 9.454 / 6769.9 | 8.612× |
| volume L1 B64 full | 1.102 / 58061.6 | 9.170 / 6979.5 | 8.319× |
| shell L1 B64 max | 3.900 / 16410.6 | 9.351 / 6844.4 | 2.398× |
| shell L1 B64 full | 3.920 / 16328.5 | 9.411 / 6800.8 | 2.401× |

依上述顺序对应 group8–11；文件 taskname 为 `recovery-{backend}-L1-B64-{mode}-{mother|atlas_packed}.json`。每组各方法均使用相同条件文件 SHA256 `ab451473db350152f812f095c64e8595a558a5ea07e4d884ba8b55d7680ff9a8`，与上面的四条件 native-v2 资格文件一致。

**后端决策：本次集成在完整路径上为负结果，推荐优化 Mother。** Atlas 在完整 policy workflow 中 shell 慢1.78–1.81倍、volume 慢2.64–3.82倍；frozen recovery 中 shell 慢约2.40倍、volume慢8.32–8.61倍。单独列出的 native-v2 volume 四条件空间通过、以及 angle/q128 的异质64条件通过，仍保留各自限定资格；不能把后者资格挪给本表 edge/q16，也不能把数值有效速率称为空间合格吞吐量。Execution map 对32项均明确 `spatial_qualified=false`、`qualified_environment_steps_per_second=0`；timing 没有逐帧重跑空间 oracle。

对未知／未合格几何，保守的100% Mother route 由**不启用 experimental atlas**（`atlas_regions=0`）实现；不是依赖当前 selector 自动察觉所有空间误差。已测 bank 的 coverage fallback=0 仅表示其 finite-footprint coverage 门未触发，不能证明一般几何的空间精度或 fallback 资格。保留 shell 结果量与 neck 远端热点反例，不推荐将原生 atlas 打开作为通用提速路径。

## 独立内存、准备和 factor 存储

`R/workflow-memory` 的18组全部 rc0，wrapper `result.json` 的 `complete`、`passed`、全部 `checks` 均通过：采样器在科学模块／GPU import 前就绪，按 worker 自身 PID（保留 namespace 映射）匹配 NVIDIA 进程，占用来自新进程完整构建、JIT、warmup和一次600步 policy rollout。不是32 task计时的重复测量，也不计作性能结果。各任务输出为 `groupN/<taskname>/{result.json,benchmark.json}`；`result.json` 保存 own-PID、task/source身份、原始日志引用及内存检查，`benchmark.json` 保存完整原生 buffer inventory。轮询请求间隔50ms，原始日志保留；采样峰值不保证捕获 allocator 的瞬时最高值。

各内存方法使用独立 Slurm allocation，**不保证方法间 GPU UUID 相同**；wrapper 的 `same_gpu` 只核对本进程采样 GPU 与内部 benchmark GPU。这里是相同GPU型号的独立进程 footprint 对照，不是同GPU配对内存实验；只有上面的 workflow-final 计时在12组内核实了方法间同UUID。采样差值保留为这批进程观察到的 footprint 差异，不作普遍硬件峰值界限。

下表每格为 **采样 NVIDIA GPU峰值 / 进程RSS峰值 / 去重原生 stress buffer，MiB**（MiB=`2^20 bytes`）。GPU/RSS覆盖完整进程，native只覆盖导出的原生 stress buffers；Torch峰值仅覆盖Torch，三者不可相加或互相替代。全部 wrapper 检查通过且 reported invalid environment-step 为0，但 memory wrapper 不建立空间资格。

| backend / B / output | 独立 optimized Mother | 独立 Atlas packed |
|---|---:|---:|
| volume B4 max | 2328 / 3191.6 / 65.434 | 2392 / 3167.3 / 136.234 |
| volume B4 full | 2328 / 3211.2 / 65.844 | 2392 / 3201.1 / 136.644 |
| volume B1024 max | 3550 / 3307.4 / 289.064 | 4798 / 3299.6 / 1547.889 |
| volume B1024 full | 3646 / 3306.7 / 394.064 | 4894 / 3294.0 / 1652.889 |
| shell B4 max | 2328 / 3158.5 / 42.974 | 2296 / 3207.4 / 45.871 |
| shell B4 full | 2328 / 3171.8 / 43.824 | 2296 / 3177.7 / 46.720 |
| shell B1024 max | 3486 / 3275.1 / 237.414 | 4286 / 3304.8 / 1034.770 |
| shell B1024 full | 3710 / 3275.9 / 454.914 | 4510 / 3309.8 / 1252.270 |

路径按 `workflow-memory/tasks.json` 定位：volume B4 max Mother/Atlas 为 group1/2，B4 full 为3/4，B1024 max为7/5，full为8/9；shell B4 max为10/11，full为13/12，B1024 max为14/15，full为16/17。另有 volume max pure-rigid group0(B4)、group6(B1024)，采样GPU/RSS分别 **2214/2993.1 MiB**、**3116/3040.8 MiB**；不把其没有 stress inventory 的情况记作整个原生场景零内存。

Atlas 在 B1024 的采样GPU峰值比 Mother 高：volume max/full均 **1248 MiB**，shell均 **800 MiB**。保留 B4 shell Atlas 采样GPU比 Mother 低32MiB的观察；它没有对应更少的去重 stress buffer，不能推广为一般内存优势。原 factor、packed额外常量与母 fallback 均如实保留，Atlas的 combined驻留没有被当成独立 Mother 峰值。

为区分共享与每环境存储，对同 backend/output/method 的唯一 buffer 总量作两点差分：`per_env=(U1024-U4)/1020`，`shared=U4-4*per_env`。逐 inventory 的非 alias bytes 求和已核对等于 `native_buffer_bytes`；下表是这两个实测 B 的存储模型，**不是硬件峰值、allocator 上界或任意 B 的保证**。

| backend / output | Mother shared MiB / per-env KiB | Atlas shared MiB / per-env KiB |
|---|---:|---:|
| volume max | 64.557 / 224.507 | 130.698 / 1417.191 |
| volume full | 64.557 / 329.507 | 130.698 / 1522.191 |
| shell max | 42.212 / 195.202 | 41.993 / 992.777 |
| shell full | 42.212 / 412.702 | 41.993 / 1210.277 |

新进程 `preparation.setup_seconds`（构建／JIT，warmup另列）在本次各 B/output 中的范围：volume Mother **141.838–145.643 s**、Atlas **224.791–230.015 s**；shell Mother **118.998–122.577 s**、Atlas **131.282–133.561 s**。它们是本次独立进程的准备成本，不能当作稳定每步时延或保证无任何编译缓存的普适 cold-start 时间。

下面取 `native-final-qualification/group0/shell-heterogeneous64/result.json` 和 `group1/volume-heterogeneous64/result.json` 的 `atlas.groups`，报告实际 K/factor 存储。条目均是导出的 **3×3 block存储数**，不是标量 nnz。Feature 行的范围覆盖12个实际小算子，括号为其总和；母行是 Atlas 内保留的 mother/fallback factor，不能用它替代 optimized Mother inverse 的实际内存。

| backend / 空间 | K block entries | factor block entries | `factor_entries/stiffness_entries` 范围 |
|---|---:|---:|---:|
| shell mother | 8648 | 12466 | 1.441 |
| shell 12 features | 7736–9208（102288） | 7460–9292（100316） | .937–1.016 |
| volume mother | 20274 | 66669 | 3.288 |
| volume 12 features | 15085–18314（203924） | 36628–55834（556979） | 2.428–3.049 |

这是显式 block 存储比；K和factor的对称／三角存储约定不同，不把它冒充 `nnz(L)/nnz(tril(K))`。Packed副本及其运行时 scratch另包含在完整 buffer inventory。小空间各自缩减不保证十二套 factor 总存储／求解优于单个母模型；实测大批量代价和内存均支持两后端保持默认 Mother。

## 证据字段与待补交付

所有表格可从对应 `result.json` 复核：

- 轨迹：`complete`、`passed`、`mean_loaded_field_error`、`spatial_passed`；`rows[*].{step,errors,per_environment_errors,coverage_fallbacks,loaded,mother_spatial_residual,feature_numerical_residual_norm}`。
- 实际尺寸：`global_enrichment_actual.{feature_raw_scalar_dofs,mother_raw_scalar_dofs,reduced_feature_groups,full_size_feature_groups,global_coarse_equals_mother}`；全局 coarse 的整数来源与 `basis.npz` 身份见 `global_enrichment`。
- 几何：`qualification_checks`、`partial_footprint_probe`、`mean_qualification_field_error`；`rows[*].{name,category,errors,numerical_passed,source_numerical_checks,maximum_relative_force_error,maximum_relative_moment_error,neck_witness}`。完整 field NPZ 按 `rows[*].frame` 定位。
- Profile：`device_categories[*].{category,sum_duration_us}`、`device_kernels[*].{name,sum_duration_us}`、`warmup_checks`、`final_checks`。`native_buffers` 和进程采样的 scope 不等于独立方法的 allocator 峰值。
- Workflow：`R/meta/workflow-execution-map.json:rows[*].{group_name,method,actual_output,pairing,spatial_qualified}`；输出文件的 `repeats[*].{steps,seconds,transitions_per_second,invalid_environment_steps,scatter_overflow_calls,resets}`、`gpu_uuid`、`policy`、`method_selected`、`surface_inverse_selected`、`stress_configurations`、`native_buffers`，以及相邻 `.returncode.txt`。
- 独立内存：wrapper `result.json` 的 `checks`、`memory.{sampled_peak_device_bytes,raw_log,pid_mapping,matched_poll_count,requested_interval_seconds}`、`peak_process_rss_bytes`、`native_memory`、`preparation`；原始 `benchmark.json` 的 `native_buffer_bytes`、`native_buffers[*].{name,shape,dtype,bytes,alias_of}`（未出现 `alias_of` 代表该条不是 alias），以及 `stress_configurations`。最终 profile 使用 `R/native-final-profile` 同名 profile字段。

本批 workflow、独立内存和最终profile已终态；两后端均为有界 native 集成／调参后的完整路径负结果，默认推荐 Mother，无需继续科学扫参。后续仅由主任务完成证据封装、保存审计和全目标资格审计。指定 volume 轨迹的空间资格和既有反例均保留；本文不自行关闭整个目标，也不将已测通过推广为任意非凸几何或连续体锐边峰值资格。
