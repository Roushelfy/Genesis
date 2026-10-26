# CPU12 native adaptation：阶段结果，2026-10-26

本文记录 [CPU12 目标](ATLAS_CPU12_GPU_PARITY_GOAL_20261026.md) 第二阶段中已核实的原生空间、轨迹、几何控制和 intrusive profile。**这不是整个目标的完成声明。** Volume 的指定 Franka 轨迹通过；shell 仍有结果量误差反例，非凸颈部远端热点仍未普遍合格。完整 workflow 的配对吞吐量、最终内存实验和最终后端决策留待后续补充。

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
| native-v2 / native-final-qualification | `94ec782fd0754017074cf3b567b391b1444f7172` | `/mnt/home/zhaofeng/workspace/Genesis/.worktrees/cpu12-native-v2`；`R/native-v2`、`R/native-final-qualification` |

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

## 证据字段与待补交付

所有表格可从对应 `result.json` 复核：

- 轨迹：`complete`、`passed`、`mean_loaded_field_error`、`spatial_passed`；`rows[*].{step,errors,per_environment_errors,coverage_fallbacks,loaded,mother_spatial_residual,feature_numerical_residual_norm}`。
- 实际尺寸：`global_enrichment_actual.{feature_raw_scalar_dofs,mother_raw_scalar_dofs,reduced_feature_groups,full_size_feature_groups,global_coarse_equals_mother}`；全局 coarse 的整数来源与 `basis.npz` 身份见 `global_enrichment`。
- 几何：`qualification_checks`、`partial_footprint_probe`、`mean_qualification_field_error`；`rows[*].{name,category,errors,numerical_passed,source_numerical_checks,maximum_relative_force_error,maximum_relative_moment_error,neck_witness}`。完整 field NPZ 按 `rows[*].frame` 定位。
- Profile：`device_categories[*].{category,sum_duration_us}`、`device_kernels[*].{name,sum_duration_us}`、`warmup_checks`、`final_checks`。`native_buffers` 和进程采样的 scope 不等于独立方法的 allocator 峰值。

后续由主任务补充：完整 Franka/hollow-egg workflow 的 max/full 配对吞吐量与纯 rigid/优化母模型对照；独立进程 mother/atlas/packed 内存及 buffer inventory；最终精度／代价决策和全目标资格审计。当前只确认指定 volume 轨迹的空间资格和已报告的负项，不关闭整个目标，也不将其推广为任意非凸几何或连续体锐边峰值资格。
