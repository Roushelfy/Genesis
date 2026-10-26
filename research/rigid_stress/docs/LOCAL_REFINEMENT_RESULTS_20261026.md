# 原生局部加密 Atlas：机制、精度与成本

本报告记录 2026-10-26 的真实局部空间实验。
**科学试验、六格决策、生产清理与清理后回归已完成。**
局部空间整体未胜过当前母模型；聚合和一次恢复则有明确内部收益。
新增生产路径已撤回，可运行冻结代码与证据保留。

任务依据为 [GOAL](../GOAL.md)、
[局部加密 review/goal](ATLAS_LOCAL_REFINEMENT_REVIEW_AND_GOAL.md) 和
[执行账本](LOCAL_REFINEMENT_PROGRESS_20261026.md)。
主要算法参考是 [cpu_repro](../atlas_study/cpu_repro/README.md)，
其 [CPU 报告](../atlas_study/cpu_repro/REPORT_ZH.md) 不构成原生 GPU 资格。
此前完整母响应的区域分列实验及 H/FP32/screen/history 负结果继续保留。

已有证据证明两后端确实建立了较小独立坐标、不同的刚度与因子，
并通过共同规范、虚功、原数值门控及 all-bubbles 母解重建控制。
低模完整轨迹的精度前沿很窄：较小特征常失败，达标宽库大部分退化到母维数。
在两套条件库、两种输出和小/大 batch 的匹配计时中，全部局部路径仍较慢。
该结果限定于本文构造和已测规模，不是对所有局部 reduced space 的否定。

## 1. 实际实现的空间

每个后端独立定义物理母模型；shell 和 volume 不共享数值离散。
在同一后端中，几何、材料、厚度、质量、刚体模态及输出采样保持共同。
生产数值准备和运行均在现有 rigid solver 的 Quadrants 中完成。
NumPy 负责整数拓扑与测试输入，SciPy 用于独立研究审计。
生产浮点装配与求解保持在 Quadrants 中。

```text
T_i = [P, B_i]
K_i = T_i^T K_m T_i
f_i^e = sum_c w_ic f_c^e
b_i^e = T_i^T R_m f_i^e
K_i z_i^e = b_i^e
q^e = q_spin^e + q_fallback^e + sum_i T_i z_i^e
sigma^e = S_m q^e
```

`P` 是共同全域粗位移空间；`B_i` 是局部独立细坐标。
`K_i` 由母刚度投影得到，包含粗区与细区的全局弹性耦合。
它不是母逆矩阵的区域列，也不以完整母修正补回每个近似解。
`factor`、`inverse` 与 `response` 是同一已声明特征空间的求解实现。
其中 feature response 由该较小 `K_i` 生成，不能与旧完整母响应分区混称。

主要原生文件为 `stress/feature_basis.py`、`feature_operator.py`、
`feature_regions.py`、`feature_atlas.py` 及其边界响应实现。
这些路径位于 `genesis/engine/solvers/rigid/` 下。
这些文件属于冻结实验实现；决定不继续保留新增生产依赖路径。
原 `4bc4ada25` 基线的精确压力、fallback 与可选角积分优化继续保留。

### 共同质量、规范与惯性卸载

每张特征投影同一个母模型的惯性卸载，而不独立重新平衡。
可等价计算 `T_i^T f - (T_i^T M_m G)(G^T M_m G)^-1 G^T f`。
母 pin 所在物理节点进入共同粗空间；特征继承同一组标量约束。
约束后的 cardinal 子矩阵与互异局部 identity 列给出满列秩证书。
审计另记录 map rank、最小 cardinal pivot、Galerkin 与对称误差。

spin 的六列母响应在准备阶段求解，并经过原母数值门控。
运行时旋转体积载荷只加入一次，不随接触源或特征数重复。
group zero 收集 coverage 分配给母模型的载荷，保留精确模式门控。
近似父状态的 residual 字段汇总 source-space 数值残差；
完整母残差另作空间诊断，不将其冒充 reduced 数值残差。

### Volume 的 P1 加局部 P2

母模型采用 affine tetrahedral 几何和 P2 位移。
粗空间保留原 tetra 顶点，边中点按两个端点各一半插值。
必要 pin 节点与固定热点节点提升到共同 cardinal 集。
`B_i` 补充选中区域的非粗 P2 节点，保留全部三维位移分量。
区域按母单元连通图构造；局部节点的完整单元支撑必须被包含。
这保留 through-thickness 与三维耦合，而不是仅加密外表载荷节点。

### Shell 的几何与转角插值

母 midsurface、Q4 单元和 directors 保持原有坐标。
粗 parent 元数据给每个细物理节点至多三个 cardinal parents。
平移与全局转角分别插值，另用节点相对 parent 平均位置的偏移，
加入转角到平移的刚体修正，使 `omega cross x` 在实际曲面坐标上重现。
不能用一张平面插值网替换弯曲母几何以人为取得刚体一致性。
局部 enrichment 同时补充相邻平移、转角块；保留两侧偏心摩擦力矩。

实验弯曲粗空间进一步加入 director 法向的二次位移修正。
边中点与三角形重心系数分别为 `1/8` 与 `1/18`。
它在平面 patch 重现恒曲率弯曲，但曲面精度须单独测量。

## 2. 选择、有限 footprint 与合并

离线种子与区域使用母材料连通图距离，不使用星形鸡蛋球面射线代替非凸身份。
在线保留原生 DPR 的真实材料位置、自动半径、压力、法向力和摩擦输入。
选择不把接触吸附到代表中心，不改变压力/radius/friction 的原接受条件。
窄过渡允许所有正贡献；三方交点不强行截断为两个候选。

coverage 检查完整有限 footprint 及其形函数节点支撑。
靠近不受支持的面时，局部权重连续衰减，剩余权重进入母 group zero。
特征与母权重之和保留源载荷，不能丢弃越界部分。
surface side、非凸表面身份和实际压力积分仍由母输入路径处理。
共同粗空间全域存在，不意味着任意远端热点已获得足够空间精度。

每个源只将自己的加权载荷交给选中空间。
相同 `(feature, environment)` 先合并，再跨环境使用同一 prepared 因子/响应。
先 lift 并合并位移，再作一次全母采样恢复与 nonlinear criterion。
必须先叠加应力张量；各源 von Mises、各特征峰值不能直接相加。

## 3. 预声明资格与六格决策

特征求解沿用 unscaled reduced operator 的 relative `1e-8`、absolute `1e-11`。
精确母模式与 fallback 的原数值容差不放宽。
共同几何、规范、刚体重现、卸载和虚功目标为 relative `1e-10`，
有量纲检查的 absolute floor 在各控制 JSON 中单列。
all-bubbles 必须恢复母位移及完整应力。

每个资格案例的完整张量误差预算为 10%，全域采样最大值为 5%。
目标平均场误差为 5%；shell 另验 membrane/bending/transverse resultants。
shell 采用面积权重，volume 采用体积权重；对称剪切计两次。
采样覆盖完整物体，几何控制另保存峰值位置。
强抵消同时记录总场相对、绝对及 source-norm 归一误差，保留原总场预算。

以下六格分别作决定；总体 atlas 变慢不能替代组件增量结论。

| 候选家族 | Shell 决定 | Volume 决定 |
| --- | --- | --- |
| 真正 coarse 加 local 空间 | 机制与数值通过；精度/成本前沿不推荐，撤回生产路径 | 含 moderate 完整轨迹与审计；精度/成本前沿不推荐，撤回生产路径 |
| 每特征聚合、多 RHS | 有益内部组件，response D/C 为 3.99×/3.68×；随 atlas 依赖路径撤回 | 有益内部组件，response D/C 为 3.99×/3.82×；随 atlas 依赖路径撤回 |
| 合并位移后一次全域恢复 | 有益内部组件，response B/C 为 1.455×/2.084×；随 atlas 依赖路径撤回 | 有益内部组件，response B/C 为 1.450×/1.816×；随 atlas 依赖路径撤回 |

表中两个数分别对应 B4/B256，且只描述相同局部算子的组件控制。
四个内部组件格存在实测收益，不能写成“六格全部没有收益”。
它们的 parent atlas 完整路径未胜母模型，故不为这些依赖组件单独保留生产复杂度。
数值通过、空间资格和生产保留决定分别记录。

## 4. 曲面 Franka 精度前沿

以下尺寸为 raw scalar coordinates，或明确注明的三分量 blocks。
母与所有特征共同去除六个标量 pin；不能混用 raw/free 维数。
所有误差相对于相同母离散、相同载荷和相同输出采样。

| 配置 | 轨迹/尺寸 | 完整张量/峰值证据 | 判断 |
| --- | --- | --- | --- |
| shell 初始 local | 140 帧；特征 1110/1080/966，母 1452 | 最大 22.4337% / 5.7135% | 空间拒绝 |
| volume 初始 local | 140 帧；真正较小 P2 特征 | 最大 4.9939% / 0.0981% | 短 pilot 通过，不能替代 600 帧 |
| shell core 1.4 | 140 帧；1236–1428，母 1452 | 3.7055% / 0.3172% | 短 pilot 通过 |
| shell core 1.4 | 600 帧 | 12.0542% / 5.1857% | 完整空间拒绝 |
| volume core 1.0 | 600 帧 | 张量最大 17.3490% | 完整空间拒绝 |
| shell level2 线性 coarse | 140 帧；3708–4152，母 5772 | 张量/峰值通过，transverse 13.5694% | resultant 拒绝 |
| volume 六区域窄库 | 140 帧 | 张量最大 20.5437% | 空间拒绝 |

扩大 core 能使低模库通过完整轨迹，但空间节省明显退化。
`atlas-wide-v4` 中 shell core1.6 为 `484/484/452` blocks，对母 `484`；
volume core1.4 为 `810/810/736` blocks，对母 `810`。
两者都只有第三张特征仍小于母模型，不能以 library 布尔标签掩盖。
较窄 transition 没有修复此前失败配置。

`qualified-full-v4` 已通过宽库 lifecycle/heterogeneous。
优化后的 `postwarp-full-v7` 再完成以下六项，各为600帧，均 `complete=true, passed=true`。

| 后端/控制 | 最大张量误差 | 最大峰值误差 | 其他 |
| --- | ---: | ---: | --- |
| shell normal | 3.5837% | 0.5926% | 最大 transverse 8.4744% |
| shell lifecycle | 3.5673% | 0.1666% | 最大 transverse 8.4744% |
| shell heterogeneous | 3.5567% | 0.1619% | 最大 transverse 8.6059% |
| volume normal | 4.7211% | 0.0676% | 原载荷与数值门控通过 |
| volume lifecycle | 3.5165% | 0.0172% | 原载荷与数值门控通过 |
| volume heterogeneous | 4.2985% | 0.0435% | 原载荷与数值门控通过 |

release、partial reset、checkpoint/restore
及环境隔离证据归入对应生命周期记录。
它们资格化该宽库，不资格化此前窄库，也不证明高分辨率物理峰值。

`remaining-controls-v6` 的 volume level2 母为 3210 blocks，
三个特征为 2446/2405/2200 blocks，仍有实际 reduction。
140 帧最大张量 8.7434%、峰值 0.2747%，该短控制通过。
但 `postwarp-full-v7/group6/volume-level2` 完整600帧最大张量为15.6672%，
峰值为0.3954%，因张量预算失败而 `complete=true, passed=false`。
原数值/载荷门控通过；短 pilot 的成功不能覆盖完整轨迹的空间失败。
不同环境 batch 只能证明批量扩展，不能充当不同母模型大小。

## 5. 弯曲 coarse 与非凸 neck 的负结果

`bending-proof-v6` 的 horizontal/tilted 两个平面 patch 均通过。
独立 oracle 使用 parent pairs，与原生逐 parent 系数装配分开。
它检查 rigid、二次 transverse、线性 rotation、常转角修正及 all-bubbles。
这证明构造的平面运动学，没有预先资格化曲面 shell。

`bending-full-v6` 四项均已完成 600 帧，原数值/载荷门控通过，
最终因为既定空间预算失败而返回非零。

| 弯曲 coarse 配置 | 最大张量 | 最大峰值 | 最大 transverse | 判断 |
| --- | ---: | ---: | ---: | --- |
| core1.0，level1 | 26.3744% | 9.5217% | 50.4788% | 拒绝 |
| core1.2，level1 | 18.2499% | 4.5675% | 37.4826% | 拒绝 |
| 六区域，level1 | 29.4144% | 8.5348% | 71.2432% | 拒绝 |
| level2 | 7.3138% | 0.8028% | 19.6479% | resultant 拒绝 |

不能只取最后一行张量/峰值通过而删除 transverse 失败。
这些结果保留为该 coarse 构造的有限调优负证据，不推广为所有 shell reduced space 的不可能性。

`geometry-v5` 对真实 U-neck 比较 contact-only 与共同 fixed-neck enrichment。
shell/volume 分别有 25/47 个固定物理节点进入共同 coarse set。
固定加密后最大完整张量误差仍约 83.74% / 29.79%。
远端 neck 的位置与距 footprint 证据单列；不能以局部接触峰值掩盖。
shell 的第二个 remote witness 本身未通过，保留其失败而不算独立有效证明。
`remaining-controls-v6` 又保留 shell neck 弯曲 coarse 的失败记录。

固定热点“存在”不意味着空间资格自动通过。
受支持近似模式需要验证过的固定加密、拒绝或母回退保护。
本次决定撤回候选并保留失败证据；任意非凸近似不获生产资格。

## 6. 边界控制、联合空间与抵消

`geometry-v5` 的数值、载荷、排列、release/restore、source sum、once-only spin
和递减扰动连续性通过；shell/volume egg 均观察到三方选择贡献。
其中 volume egg 的完整场空间预算通过；shell egg 的 transverse 仍失败。
早期宽库无法定位 partial footprint，明确记录 `locatable=false`。

`remaining-controls-v6` 改用自然较窄的真实库补上实际边界输入。
两后端 `partial_footprint=true`，没有伪造 coverage mask。
shell 定向 partial 样本母权重约 `0.0273883`；
volume 该 footprint 完全回母（权重 `1`），其他输入实际出现约 `0.32984–0.37647` 混合。
载荷保留、数值与连续性通过；整个 variant 仍可因空间误差 `passed=false`。
这关闭部分覆盖机制的正确性缺口，不宣称窄库空间合格。

`joint-timing-v5` 分别对 shell/volume 的 129/221/400/599 帧进行原生独立审计。
coarse、逐源空间相加、联合 enrichment union、mother 使用共同原始载荷。
母重建、投影 K/RHS、source sum、独立应力检查和原数值门控通过。
每源残差对自己的 test space 正交，不保证对其他特征正交。
因此 independent source sum 一般不等于对总载荷的 joint Galerkin 解。

当前两个或三个特征的 union 达到母维数；这是实际饱和结果。
joint 的 `passed` 表示结构/数值审计通过，不表示全部近似场通过预算。
各 joint tuple 的残差只按其自身 `environments` 汇总，避免混入其他环境。
该阶段计时包含投影、prepared 因子、lift、完整恢复与门控，
排除碰撞、压力准备、selector、raw grouping、场景步进及离线因子构建。
它不能替代 module 或 complete-scene 速度。

`moderate-audit-v7` 对 volume level2 的129帧完成独立结构/数值审计。
母9630个raw scalar坐标，三个特征7338/7215/6600；双特征union达到母维数。
四环境加权张量最大误差 coarse/local/joint 分别为92.3286%/8.7175%/8.7175%。
该帧 prepared-factor 阶段中位为 coarse/local/joint/mother
3.384/552.374/721.324/450.534 ms；联合空间不是自动更便宜的解。
该单帧审计 `complete=true, passed=true` 表示数值一致性，不改写600帧空间拒绝。

## 7. 聚合与一次恢复的独立收益

`components-v8` 的两后端 × B4/B256 × factor/response 共八项，
加 volume signed 一项，九项均 `complete=true, passed=true`。
四路径对相同保存源载荷、相同特征空间和相同母采样产生等价完整场。

- A：每源单独求解和恢复，再合并线性场。
- D（`per_contact_common_q`）：每源单独求解，合并位移后恢复一次。
- B：先聚合 RHS，再分别恢复各特征与 spin 后合并张量。
- C：先聚合 RHS，合并位移后恢复一次。

D/C 隔离每特征聚合；B/C 隔离一次恢复；A/B 同时改变恢复次数，不能单独归因为聚合。
每项五次重复、每次100迭代，保留原求解门控、数值有效性和各路径等价检查。
下表取 response；factor 控制的聚合和一次恢复也均有正增量。

| 后端/B | D ms | B ms | C ms | D/C 聚合比 | B/C 一次恢复比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| shell/4 | 11.409 | 4.164 | 2.862 | 3.986× | 1.455× |
| shell/256 | 12.357 | 6.998 | 3.358 | 3.680× | 2.084× |
| volume/4 | 10.858 | 3.946 | 2.722 | 3.989× | 1.450× |
| volume/256 | 11.443 | 5.433 | 2.992 | 3.824× | 1.816× |

这是固定源组件恢复成本，不含碰撞、压力重拟合、selector或刚体步进。
它与原生完整场景母基线的比较见下一节，不能把组件比例换算成场景 FPS。

另有 volume signed raw-load 代数控制，reference cancellation ratio 约 `0.014`。
其总场张量误差约 `10.52%`，source-norm 归一约 `0.147%`，原 10% 总场预算仍失败。
该 signed 控制显式 `physical_contact_qualification=false`；
反向组合不修改原压力输入，也不能作为物理 pressure/friction 接触资格。
signed 的 `passed=true` 仅表示原数值门控和四路径等价通过；空间失败标志保留。

## 8. 完整配对成本、profiling 与内存

下表为已优化 selector、精度通过宽库的 `postwarp-paired-v7`（32份结果）。
数字为各方法 ms/调用中位数；开销按原始重复序号配对后取中位。
module 是冻结真实 contact snapshot 及其匹配 frame 的 recovery 重放，
每项五次、每次100调用；scene 是带 policy inference 的实时600帧，三次。
module 不能称为完整演化场景；scene 不能只保留恢复阶段成本。

| 后端/B/输出 | module 母/local ms | scene 母/local ms | scene 配对开销 |
| --- | ---: | ---: | ---: |
| shell/4/max | 3.923 / 5.517 | 6.307 / 7.790 | +23.38% |
| shell/4/full | 3.904 / 5.427 | 6.251 / 7.981 | +27.75% |
| shell/1024/max | 9.171 / 11.712 | 13.294 / 16.105 | +21.15% |
| shell/1024/full | 9.264 / 11.682 | 13.391 / 15.839 | +18.31% |
| volume/4/max | 1.106 / 2.639 | 5.055 / 6.904 | +36.85% |
| volume/4/full | 1.131 / 2.855 | 5.079 / 6.756 | +33.89% |
| volume/1024/max | 3.715 / 6.211 | 9.409 / 11.942 | +26.92% |
| volume/1024/full | 3.811 / 6.300 | 9.581 / 12.348 | +28.88% |

这轮所有候选重复无 invalid environment steps；完整 JSON 保留范围与元数据。
先前窄库的 `module-paired-v4`/`scene-paired-v4` 也均完成且较慢，
但同时空间不合格，只作为该点的准确率/成本证据。
不能使用窄库速度和宽库误差拼出不存在的配置。

`heterogeneous-paired-v7` 另有32份结果，使用64个不同条件，并按需复制至B1024。
shell保留edge/q16；volume匹配已资格化angle/q128。不得将两者不同积分负担混为算法加速。

| 后端/B/输出 | module 母/local ms | scene 母/local ms | scene 配对开销 |
| --- | ---: | ---: | ---: |
| shell/64/max | 4.518 / 6.047 | 8.766 / 10.346 | +17.96% |
| shell/64/full | 4.691 / 6.253 | 9.008 / 10.599 | +17.62% |
| shell/1024/max | 8.382 / 10.700 | 14.171 / 16.578 | +16.98% |
| shell/1024/full | 8.614 / 10.898 | 14.333 / 16.671 | +16.38% |
| volume/64/max | 7.389 / 8.827 | 10.957 / 12.632 | +15.29% |
| volume/64/full | 7.415 / 8.933 | 11.045 / 13.209 | +19.65% |
| volume/1024/max | 48.430 / 50.904 | 64.385 / 67.093 | +4.21% |
| volume/1024/full | 49.278 / 51.761 | 65.493 / 68.139 | +4.04% |

开销按重复序号配对，取 `local/mother-1` 的中位。
它不等同于两个独立中位的比值。
两套bank全部候选重复的invalid environment steps均为0；所有点均慢于母路径。
原始重复、条件SHA、参数与数值资格分开保存，setup含JIT并排除于稳态计时。

`profile-v5` 指向选择阶段：shell B1024 的 device task sums 约1.86 ms，
其中约1.80 ms来自逐接触串行 footprint-face 扫描。
shared lift 约0.075 ms（shell）/0.116 ms（volume），不支持优先优化 lift 的归因。
这些是 intrusive profiler 诊断，不能与无插桩总延迟直接相加。

`selector-v7` 的四项 B4/B1024 × shell/volume 均完成20种输入控制。
权重差精确为0、fallback决定一致，压力状态与权重守恒检查通过。
完整特征求解的原数值门控由 `postwarp-full-v7` 另行验证。
首个 moving 输入隔离 selector 的 scalar→cooperative 中位数如下。

| 后端/B | scalar ms | cooperative ms |
| --- | ---: | ---: |
| shell/4 | 1.703 | 0.1183 |
| shell/1024 | 1.907 | 0.5242 |
| volume/4 | 0.4427 | 0.0605 |
| volume/1024 | 0.5169 | 0.2600 |

该结果只证明当前 selector 调度优化的等价及隔离成本变化。
其实际完整路径改进已经包含在上述post-warp配对中，仍未超过母基线。

`final-profile-v7` 的10项母/local控制全部完成并通过原门控。
优化后shell B1024的selection约0.378 ms，grouping0.396、lift0.075、stress0.721 ms；
contact/pressure/scatter约7.227 ms，acceptance/state约0.810 ms。
volume异质q128候选的contact/pressure/scatter约46.699 ms，压力仍主导。
这些是插桩device类别和，不替代上表无插桩wall latency，也不从fused母kernel拆分虚构阶段。

以下为full输出、B1024的去重原生常量/每环境工作区及独立进程采样峰值。
MB/KB为十进制；采样峰值使用MiB，含进程CUDA/JIT开销，非连续硬件容量界限。

| 后端/积分 | 共享常量MB 母→local | 非常量KB/env 母→local | 原生总MB 母→local | 进程GPU peak MiB 母→local |
| --- | ---: | ---: | ---: | ---: |
| shell q16 | 44.26→68.21 | 422.61→683.01 | 477.01→767.62 | 3710→4062 |
| volume q16 | 67.69→62.72 | 337.42→684.20 | 413.21→763.33 | 3646→4030 |
| volume异质q128 | 114.92→109.94 | 337.42→684.20 | 460.43→810.56 | 3678→4094 |

volume共享常量略降不能掩盖每环境工作区近翻倍和实测进程峰值上升。
host RSS、Torch peak、别名去重方法和独立采样间隔在各profile JSON单列；彼此不能替代。

## 9. 源版本、证据与复现

运行根目录 `R` 为：
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261026-local-refinement-atlas`。
大数组、日志、缓存及新运行均留在 data 侧；本文不复制 samples/trace。
每项的 `source.json`、`source.patch`、`tasks.json`、workload 和结果 SHA 是复现依据。
部分 benchmark 的 `source_revision="unrecorded"` 不能替代 campaign capture；须关联该捕获记录。

| 捕获源/用途 | source pin | 主要 campaign |
| --- | --- | --- |
| 初始真实空间控制 | `7cad6e3639813b70792875531aa47cf6acc9867e` | `space-controls-v1c` |
| 原生运行、调优与旧selector配对 | `a17fc82b76a78abe4ff3cd76e7a4208a019cb7c2` | `atlas-tuning-v4`、`qualified-full-v4`、`qualified-paired-v4`、`joint-timing-v5` |
| 弯曲 coarse 与补充控制 | `27a9b42435cb70693a1da9cb203fdce3eb979703` | `bending-proof-v6`、`bending-full-v6`、`remaining-controls-v6` |
| cooperative selector与最终engine捕获 | `a971d61d127148401cff2de0d0552daa60e77105` | `selector-v7`、`postwarp-paired-v7`、`postwarp-full-v7`、`heterogeneous-paired-v7`、`final-profile-v7`、`moderate-audit-v7` |
| 完整研究源码与四路径组件worker | `822ad285497f79d3a24f3ce2f0dd7a58ec6cd699` | `components-v8`；实际engine仍按该项source.json捕获为a971 |

研究 worker 还单独捕获 `analysis-source`；其 SHA 与 engine source pin 分开记录。
例如 `joint-timing-v5` 使用 frozen v4 engine 和新审计 worker，不能混写为同一提交。
早期编译/接口失败、探针顺序修正和未完成快照保留，不删除为“干净结果”。
复现在独立 worktree 检出完整研究 pin `822ad2854` 或该 campaign 的 engine pin，
再匹配 `R/<campaign>/analysis-source` 和捕获的 workload；生产清理不改写这些源。

入口位于 `research/rigid_stress/qipc_comparison/`：
`check_local_feature_spaces.py`、`check_local_atlas.py`、`check_local_geometry.py`、
`audit_local_keyframes.py`、`check_shell_feature_bending.py`、
`check_feature_selection.py`、`measure_local_components.py`、`profile_local_atlas.py`。
精确参数以每项 workload 为准，不从默认参数推测历史配置。
数值任务只在 Slurm `rtx-mid`、镜像 `1_26` 执行；科学 CPU oracle 同样走该环境。
复现须新建独立输出和缓存，不覆盖 frozen campaign，也不热改运行源。

科学证据已完成：真实两后端空间、原数值/载荷、完整精度前沿、有限调优、
组件独立收益、两bank的max/full模块与场景成本、最终profiling及内存。
清理提交 `36f887350` 的 `genesis`/`examples` 与 `4bc4ada25` 内容完全相同。
Slurm 751411 对该冻结清理源执行20项GPU回归：shell 9项、volume 11项，
全部通过，无失败、错误或跳过。原12个未完成文件和15个原冻结版本均未改变；
连同本轮实验与清理源共23个冻结engine通过保护核查。
详见[最终证据包](../evidence/20261026-local-refinement-v1/README.md)、
[六格需求审计](../evidence/20261026-local-refinement-v1/metadata/requirements-final.json)
和[汇总索引](../evidence/20261026-local-refinement-v1/metadata/index.json)。
推送提交与远端一致性另记录在运行根目录的 `meta/delivery-push.json`。
本稿不将相同离散的一致性解释为连续体或独立接触 ground truth。
shell 缺失的三维接触正应力、尖边峰值和高分辨率物理收敛仍在本阶段范围外。
任何负结论限于已测试的空间构造、尺寸、载荷、输出和硬件；
任何正结论也须绑定完全相同的资格配置。
