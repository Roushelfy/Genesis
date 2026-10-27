# CPU12 同空间 GPU 对照：交付结果，2026-10-27

原 CPU 十二空间已在原生 Quadrants FP64 路径复现，同空间 CPU/GPU 数值对照通过。原空间本身仍有 shell M/Q 空间误差反例；保持局部支持的 P6 coarse 提升也未消除 Q 失败。Cooperative 和 packed 调度显著降低 factor 成本，但在更强的 auto FP64 surface-inverse 母模型对照下，packed Atlas 的完整模块耗时仍为母模型的 **4.4–4.7 倍**。

本文覆盖 [CPU12 目标](ATLAS_CPU12_GPU_PARITY_GOAL_20261026.md) 的同离散模块、空间控制和独立进程内存证据。原生 shell/volume 几何、DPR、Franka 和完整 policy workflow 见 [native 阶段报告](CPU12_NATIVE_RESULTS_20261026.md)。完整目标的交付与适用范围见 [GOAL.md](../GOAL.md)；[证据包](../evidence/20261026-cpu12-gpu-parity-v1/README.md) 保存十三项审计及原十二个未完成文件、二十三个旧 frozen source 的保留核验。另有本轮十一份冻结源码 inventory 逐文件核验通过。

## 来源和可复现范围

数据根目录记为 `R`：

```text
/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261026-cpu12-gpu-parity
```

| 身份 | 精确 revision | 冻结来源／证据 |
|---|---|---|
| 独立 CPU primary | `a10c5abe3b5bc3c5caa57d3d4d9dadab2b277ade` | `R/oracle-v2/group0/reference/manifest.json` 的 `primary_commit` |
| CPU oracle exporter | `36f8873501d530e0baabf3c87b68ad320cafb54e` | 同 manifest 的 `checkout_commit`；只读 CPU oracle 与 physical/integer input 分开 |
| parity-v4 | `55386eb45febf4f587f5c4f767dc59fbed4a7220` | `R/parity-v4`，stress reconstruction 与 2048 block 边界控制 |
| parity-v5 | `4cb351d172a17f2acf4d760cfdff39760cf92c2e` | `/mnt/home/zhaofeng/workspace/Genesis/.worktrees/cpu12-parity-v5`；`R/parity-v5` |
| parity-v6 / memory-v6 | `3234380188ae2d4e717767989a6d5baefb6f7214` | `/mnt/home/zhaofeng/workspace/Genesis/.worktrees/cpu12-parity-v6`；`R/parity-v6`、`R/memory-v6` |

oracle manifest SHA256 为 `92ea234839d8f5e6056f1df7c05ec8e787ed5b123fb83c0518ced943f4b1e7cb`。各 campaign 的 `source.json`、空 `source.patch`、`analysis-files.json` 和 `analysis-source/` 保存引擎与研究脚本的独立身份；`tasks.json`、`workload-*.sh`、`job.sh`、`logs/` 保存实际参数和返回码。不能仅凭引擎 revision 推断捕获的研究脚本版本。数值执行均在 Slurm `rtx-mid` / image `1_26`；本文只整理已存结果，没有新增科学执行。

匹配母模型为 Q4 MITC4 shell，母空间 **5196 raw / 5190 free scalar DOF**，十二个 `T_i=[P,B_i]` 各 **3522 raw / 3516 free DOF**。四角 Q4 `P_nodes ⊗ I6`、全 incident macro 的 bubble、六个固定 gauge DOF、原母 K、质量及共同 relief 均逐项核对。所有浮点装配、投影、分组、求解、lift 和恢复在 QD；host 只构建整数拓扑。CPU K/T/S、factor 或解只供独立 oracle/readback 检查，未注入生产求解或计时路径。

这里的 **CAP** 是 CPU 指定的材料球冠离散载荷：固定 physical input、材料坐标定位、normal/tangent 分支及 q8/q16/q24 积分。其离散 loaded-node coverage 不是原生 DPR 完整接触足迹资格，也不是由真实碰撞、自动半径或压力拟合生成的 Franka 载荷。锐边接触的连续峰值收敛仍不由本报告认证。

## 数值对照和空间资格

| 已完成控制 | checks | 结果与数据路径（相对 `R`） |
|---|---:|---|
| 四 bank、三 selector 完整同空间对照 | 2332 | 全通过；`parity-v5/group1/all-banks/result.json` |
| displacement-to-stress 原生恢复 S | 172 | 全通过；`parity-v4/group0/stress-operators/result.json` |
| 真实 published43，default / sequential cooperative / packed | 1410 | 全通过；`parity-v5/group0/packed-all-groups/result.json` |
| 2048 block 原生 SPD chain 共享内存边界 | 87 | 全通过；`parity-v4/group1/boundary2048/result.json` |

这些计数属于不同控制，不合并成独立物理样本数。完整对照包含 published43、boundary18、static16/64，保留全部正权重 contributor、共同 relief、原小空间残差门、合并后一次 stress recovery，以及 loaded→zero→reload 和环境隔离。原残差门保持 unscaled `max(1e-11, 1e-8*rhs_norm)`；full-mother residual 是空间诊断，不作为修正 RHS 或放宽小系统门的理由。

独立保存结果审计位于 `spatial-v2b/group0/audit`：`cpu_gpu_equivalence_passed=true`，`spatial_passed=false`，`cpu_spatial_passed=false`。其 `checks.json` 中 `*/cpu_gpu/stress_full9` 最大 relative L2 为 **2.6152e-13**（约 `2.7e-13`）。原 `spatial-v2` 因非 GPU 审计节点上的 `nvidia-smi` 身份查询失败，未完成审计；失败日志保留，以上结论使用修正后的 `spatial-v2b`。

原空间 published43 的 loaded-case 结果如下，均为百分比。CPU 与 GPU 在上述数值误差内重合；表取审计的 GPU 值。field 使用 CPU 原权重，含双表面、四 Gauss 点和 tensor 剪切 multiplicity 2；N/M 使用 Gauss 面积与 tensor 范数，Q 使用加权 vector 范数。预算为 mean field≤5%、worst field≤10%、peak≤5%、N/M/Q 各≤10%。

| selector | mean field | worst field | peak | N | M | Q | 空间资格 |
|---|---:|---:|---:|---:|---:|---:|---|
| continuous | 4.4524 | 7.5895 | 0.9738 | 5.2862 | **10.3244** | **19.9683** | M/Q 失败 |
| nearest | 4.5979 | 9.7410 | 0.9737 | 6.9299 | **12.5722** | **19.9685** | M/Q 失败 |
| narrow | 4.4734 | 7.5896 | 0.9737 | 5.2862 | **10.3245** | **19.9685** | M/Q 失败 |

字段位于 `spatial-v2b/group0/audit/published43/<selector>/summary.json` 的 `native/cpu.maximum_errors`、`mean_loaded_field_error`。static bank 的重复不是独立轨迹；field/peak 通过不能替代 M/Q 资格。

## 保持局部支持的 P6 控制

`parity-v6/group4/held-p6-support` 在同一个物理母模型、原十二中心上，将每个 P3 macro 分成四个 P6 macro。首个控制固定原活跃 macro 覆盖；第二个再加一圈 P6 halo。局部 B 仍只收纳全部 incident macro 都活跃的非 coarse 节点，不以近距离跨接拓扑。

| 空间 | coarse nodes | 每 feature bubbles | raw / free DOF | free / Mother free |
|---|---:|---:|---:|---:|
| 原 CPU P3 | 56 | 531 | 3522 / 3516 | 67.75% |
| P6-heldP3-support | 218 | 428 | 3876 / 3870 | 74.57% |
| P6-heldP3-support+oneP6halo | 218 | 520 | 4428 / 4422 | 85.20% |

两控制十二空间共 **24 项 `contains_P3_space`** 检查得到 `T6 @ H == T3` 的 absolute/relative L2 都为 **0**。`H` 保存于各配置 `operators/feature-XX/P3_embedding.npz`；它是较大空间中的嵌入，不是方阵换基。coarse cardinal 与 bubble identity minor 提供满列秩证据，共同 gauge/mass/relief 同时核对。离线构造的 H 没有进入原生路径。

全部 **330 checks 通过**，所有 selector 的 coverage failure 为0，小空间最大 relative residual≤`1.86e-13`。但下面六组仍未满足全部空间预算：

| 控制／selector | mean field | worst field | peak | M | Q | 失败 |
|---|---:|---:|---:|---:|---:|---|
| held / continuous | 4.1837 | 7.0277 | 0.9145 | 9.9721 | **16.5439** | Q |
| held / nearest | 4.3132 | 9.0031 | 0.9144 | **12.0795** | **16.5440** | M/Q |
| held / narrow | 4.2035 | 7.0277 | 0.9144 | 9.9722 | **16.5440** | Q |
| halo / continuous | 3.4014 | 5.7531 | 0.9228 | 8.2741 | **11.2103** | Q |
| halo / nearest | 3.5058 | 7.7391 | 0.9228 | **10.4136** | **11.2104** | M/Q |
| halo / narrow | 3.4165 | 5.7532 | 0.9228 | 8.2741 | **11.2104** | Q |

N 全部通过。字段为 `result.json.decisions[*]`、`checks.json` 和各配置 `operators/inventory.json`。本控制 `pilot_repeats=0`，没有性能结论；也不能把 larger-space 精度改善当作同等缩减下的收益。

## 调度优化与更强母模型对照

最初串行 factor 完整模块 B64 nearest/max 约为 Mother **128.6 ms**、Atlas **646.6 ms**；其中投影／求解／lift／恢复约643 ms，CAP／selector／coverage 约2.2／0.27／0.89 ms。实际尺寸的 factor-only cooperative 控制曾将 B43 Mother 从112.54降至12.65 ms、单 feature 从57.24降至8.03 ms；这说明三角求解调度是瓶颈，不是完整场景加速证据。

v5 已允许≤2048 block 的默认 cooperative solve；`cooperative=False` 和更大系统保留原路。2048 block 的 FP64 vec3 workspace 正好49152 B，边界控制独立于物理参考。Packed 将十二 feature 的 native factor 常量复制成连续存储，以一个 warp 处理一个 `(feature, environment)`，组间并行；原 factors、原 Ki 残差与外部验收保留。真实 B43 factor-only 五对中，默认和 sequential cooperative 中位数约 **94.625/94.639 ms**，packed 为 **7.197 ms**；计时排除准备、JIT、残差、lift 和恢复。

完整模块的公平对照采用 Mother `auto`，实际选中 **FP64 inverse + surface_inverse**，surface load 生效且无回退；预算512 MiB，factor scaling/cooperative 开启，原容差 `1e-8/1e-11`，允许两次 inverse correction。全部实测 correction 和 fallback 均为0。四组均 nearest、packed Atlas、3次 warmup、7对 repeat，同一 allocation 以保存的 pair/order 比较：

| bank／output | Mother median ms | packed Atlas median ms | paired Atlas/Mother median |
|---|---:|---:|---:|
| static16 / max | 3.757 | 17.215 | **4.430×** |
| static16 / full | 3.969 | 17.920 | **4.532×** |
| static64 / max | 3.979 | 18.527 | **4.665×** |
| static64 / full | 3.996 | 18.375 | **4.624×** |

这是 fresh native CAP→aggregation/selection/coverage→relief→solve→recovery→device numerical gates 的模块耗时，CPU oracle、检查导出和 readback 不计入。所有16/64个环境都数值有效，含1/4个释放零载荷环境；无 invalid source、CAP invalid 或 coverage failure。空间门仍失败，因而不形成空间合格吞吐量，也不是完整 Franka/workflow 结论。

路径为 `parity-v6/group{0..3}/static{16,64}-{max,full}-auto-packed/`。`result.json.mother_recovery` 保存实际 method、precision、surface path、checks；`measurements.json` 保存 pair/order、valid、correction/fallback counts；上表 ratio 取逐 pair 比值中位数，不是两个独立中位数之比。准备约91.5–94.5 s；首次调用含 JIT 的 Mother/Atlas 各约2.74–2.78 / 1.30–1.35 s，单列于 `setup.json`、`first_execution_including_JIT.json`，不混入稳态。

## 八组隔离进程内存

`memory-v6` 以 B64/static64、76 sources、nearest 和三个 fresh current-load calls 分进程测量。Mother-only 不构造十二 feature；Atlas/packed 保留 direct Mother。只有 `mother-auto-*` 两组保留 auto inverse/surface 缓存。因此下面的 isolated Atlas 内存不能冒充 paired auto-baseline 进程的缓存配置。

MB 为十进制，GPU/RSS MiB 为二进制；cold 从 `before_scientific_imports` 到 `after_current_load_call_1`，包含 imports、准备、分配和首次 JIT。

| group／method-output | shared MB | per-env MB | native resident MB | sampled own GPU peak MiB | RSS peak MiB | cold s |
|---|---:|---:|---:|---:|---:|---:|
| 0 mother-max | 21.666 | 0.223107 | 35.945 | 918 | 1267.00 | 37.631 |
| 1 atlas-max | 145.043 | 2.076424 | 277.934 | 1146 | 1296.02 | 108.828 |
| 2 packed-max | 198.736 | 2.414584 | 353.270 | 1210 | 1295.14 | 111.130 |
| 3 mother-auto-max | 453.897 | 0.223107 | 468.176 | 2198 | 1274.57 | 38.142 |
| 4 mother-full | 21.666 | 1.024899 | 87.259 | 950 | 1266.08 | 36.895 |
| 5 atlas-full | 145.043 | 2.878216 | 329.249 | 1178 | 1295.13 | 110.353 |
| 6 packed-full | 198.736 | 3.216376 | 404.585 | 1274 | 1299.01 | 110.472 |
| 7 mother-auto-full | 453.897 | 1.024899 | 519.490 | 2198 | 1274.72 | 38.346 |

全部8组 `passed=true`，每次64 valid、0 rejected/correction/factor fallback，source/input unchanged，constant/nonconstant alias conflict 为空。resident inventory 按 `(device, untyped_storage().data_ptr())` 的实际 allocation 去重；包括暴露的 native tensors，不含编译器、CUDA context、allocator pool 或准备临时量。常量／每环境拆分是 storage 属性，不能用总显存除以 B 代替。

- Packed 额外保留 **75,335,384 B**：常量53,693,144 B、B64 workspace21,642,240 B，与 packed−Atlas resident 差额一致；原 factors 未释放。1174 block feature 使用 bucket1536，共享内存每 block **36,864 B**。
- Mother auto 比 direct 多 **432,230,992 B**，即 full inverse215,987,328 B 加 surface inverse216,243,664 B；paired Atlas 进程也保留这批 Mother cache，已在 benchmark metadata 说明。
- 原始稀疏计数：Mother K stored blocks31,112、factor lower99,502；十二 feature K blocks合计474,768、factor lower619,076（各50,007–53,867）。总 lower718,578；这里只报告实际计数，不把不同存储约定拼成未定义的 fill ratio。

采样 collector 在科学 import 前启动，8组均 `available=true`、无 start/query/parse error、collector rc0，raw trace 只匹配本进程 preferred PID 与一张 GPU。各组 own-PID matched samples为495/1968/1972/522/497/1972/1972/524；覆盖 init、准备、JIT 和 current-load calls。中位采样间隔约50.06 ms，最大间隔157.07 ms。表中 GPU peak 是**采样下界**，不能证明瞬时硬件／allocator 精确峰值；RSS 与 native resident 的统计口径也不同。

每组字段路径：`result.json.resident_inventory.{shared_constant_bytes,nonconstant_bytes_per_environment,unique_native_bytes}`、`process_gpu_sampling.sampled_peak_device_bytes`、`peak_process_rss_bytes`、`stage_events`、`mother_recovery.resident_*_bytes`、`packed_extra_inventory`、`mother_factor_lower_entries`、`feature_factor_lower_entries`。`buffers.json` 提供 K shape，`allocations.json` 提供去重证据，`nvidia-process-memory.jsonl` 保留原始采样，不仅保存最终峰值。

## 复现入口

从对应 campaign 的 `workload-*.sh` 复现，使用新输出目录、同一冻结引擎和其 `analysis-source`；脚本已经明确设置 data 输出和节点本地临时 cache。下列是 v6 实际参数的缩写，须在 Slurm `rtx-mid` / image `1_26` allocation 内执行；`OUT` 必须是尚不存在的新 data 路径：

```bash
R=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261026-cpu12-gpu-parity
SRC=/mnt/home/zhaofeng/workspace/Genesis/.worktrees/cpu12-parity-v6
PY=/mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python
export PYTHONPATH="$R/parity-v6/analysis-source:$SRC"
cd "$SRC"
"$PY" -m research.rigid_stress.qipc_comparison.benchmark_cpu12_gpu \
  --reference "$R/oracle-v2/group0/reference" --bank static64 \
  --output-mode full --strategy nearest --warmup 3 --repeats 7 \
  --packed-solve --mother-method auto --mother-surface-load --output "$OUT"
```

同源 P6 入口为 `tune_cpu12_shell_spaces --reference <reference> --output <fresh-output>`；内存入口为 `measure_cpu12_memory --method mother|atlas|packed --reference <reference> --bank static64 --output-mode max|full --strategy nearest --output <fresh-output>`，auto Mother 另加 `--mother-method auto --mother-surface-load`。对应模块均有 `research.rigid_stress.qipc_comparison.` 前缀。v5 完整数值入口是 `check_cpu12_gpu --stage all --bank all`，packed 控制是 `check_packed_cpu12`；精确完整命令以各 campaign 的捕获 workload 为准。

本阶段支持“原 CPU12 同空间数值复现成功”和“这些已测空间与性能配置不应替代优化 Mother”的结论。它不排除其他缩减构造，也不将 CAP 模块、原生 DPR、物理收敛和完整 policy workflow 的证据互相替代。
