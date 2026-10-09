#!/usr/bin/env python3
"""Build the scoped Chinese report from the recorded CPU validation data."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE/name).read_text())


def main():
    checks = read('arbitrary_contact_results.json')
    bench = read('arbitrary_contact_benchmark_L2.json')
    solves = read('arbitrary_contact_solve_results.json')
    tractions = read('arbitrary_traction_validation.json')
    rows = [row for case in checks['cases'].values() for row in case['rows']]
    residual = max(row['checks']['direct']['full_equilibrium_relative_nonzero'] or 0. for row in rows)
    incremental_error = max(row['checks']['incremental']['peak_relative_error_1Pa_floor'] for row in rows)
    post = bench['postprocessing']['records']
    post_table = '\n'.join(f"  {r['mode']}, threads={r.get('threads',1)}: {r['timing']['mean_ms']:.6f} ms" for r in post)
    pipeline_table = '\n'.join(
        f"  {name}: {case['pipeline']['shared_map_solve_numpy_peak']['mean_ms']:.6f} -> "
        f"{case['pipeline']['indexed_map_solve_fused_peak']['mean_ms']:.6f} ms"
        for name, case in bench['cases'].items())
    mapping_table = '\n'.join(
        f"  {name}: {case['mapping_shared']['mean_ms']:.6f} -> {case['mapping_indexed']['mean_ms']:.6f} ms"
        for name, case in bench['cases'].items())
    solve_table = '\n'.join(
        f"  {r['ordering']}, B={r['batch']}: total={r['timing']['mean_ms']:.6f} ms, "
        f"{r['timing']['env_solves_per_second']:.2f} RHS/s, LU nnz={r['factor_nnz']}"
        for r in solves['records'])
    refinement = solves['FP32_refinement']['records']
    refinement_mean = sum(r['elapsed_ms'] for r in refinement)/len(refinement)
    text = f"""只固定形状、任意接触下的刚体线弹性应力恢复：修订算法与CPU验证
日期：2026-10-09

1. 结论与适用范围

接触的位置、方向、数量、面积和是否对称，全部是每帧输入。使用完整物体，
每次从实际载荷构造完整右端项，不使用固定夹持位置、对称边界、三载荷基或固定热点。
之前三维载荷空间中的“786432个应力样本减为9个”只在该载荷空间内成立，
其微秒查询速度和据此推测的RTX6000/RL吞吐量不能迁移到当前任务。

“形状固定”只保证几何算子及矩阵稀疏结构可复用。数值刚度K还取决于材料，
质量M还取决于密度。本次材料是同一个物体的已知模型输入：
E=10 GPa、nu=0.3、rho=2000 kg/m3；不是对接触的约束。
若材料或密度可变，必须更新相应算子；不能仅凭形状固定复用旧数值分解。
若只有全局E缩放且nu不变，可以利用K按比例缩放，复用参考分解并相应缩放解。
接触力作为Neumann载荷改变右端项，不改变K；若引入随接触变化的位移约束、
接触刚度、损伤或非线性几何，固定K路线需另作处理。

这条路线是准静态线弹性应力恢复：计算由刚体当前外载和惯性产生的弹性响应，
没有推进弹性振动。刚体可以运动，弹性位移和应变需处在线性模型适用范围。

2. 离线预处理与每帧算法

离线：生成完整体积网格；组装K、M；保留六个刚体模态R；选择六个独立标量
gauge去除刚体零空间；分解缩减SPD系统A；缓存表面求积几何、空间索引、
P2连接表和重心梯度；四个角点的形函数梯度可选择预存或现算。
本代码用SuperLU，不声称它是GPU Cholesky实现。
六个gauge只选定弹性位移的坐标规范，不代表抓手给物体添加了固定支撑。

每帧，先将世界坐标的接触位置、力和刚体运动量变换到参考几何的材料坐标系。
对所有实际接触区域的向量traction进行一致有限元积分：
  f_ext = integral_surface N(x)^T t(x) dS + f_body。
法向力和摩擦切向力均保留。P2表面使用完整六节点形函数，不能把顶点负权重裁掉。
每帧接触区域、力和分布可以改变。点力在弹性连续体中的峰值可能奇异；
若需要物理最大应力，接触面积/traction分布必须由当前接触模型给出，
不能仅由几何、接触点和合力唯一推导。没有固定接触面积的假设。

自由刚体的惯性释放，采用质量一致的六模态平衡。节点相对COM位置为r_i：
  R_i = [I, -skew(r_i)]
  c_i = omega cross (omega cross r_i)
  H = R^T M R
  eta = H^(-1) R^T (f_ext - M c)
  f_eff = f_ext - M c - M R eta
所以 R^T f_eff = 0，再求解 A y = f_eff_free 并恢复完整u。
本测试由载荷反算平动和角加速度。接入Genesis时也可使用实际a和alpha；
必须检查总力/力矩与质量/惯量的一致性，不应把不平衡误差静默投影掉。
全部残差检查应包含gauge对应的原始方程；只检查free rows会漏掉假支撑反力。

求最大von Mises应力：每个直边P2四面体的应变是空间仿射函数；当单元内
本构常量时，应力也仿射，而von Mises是其偏应力的凸范数，因此未平均应力
在这个离散单元内的最大值可由四个角点获得。这是每个单元的4点，
并不是全物体固定4点或9点。本次扫描所有1920个单元的7680个角点。
曲边/isoparametric P2、单元内连续变材料或平滑后的应力不直接适用此结论。

当前均匀各向同性实现直接计算：
  VM = mu sqrt(2[(exx-eyy)^2+(eyy-ezz)^2+(ezz-exx)^2]
                  +3[gxy^2+gyz^2+gxz^2])。
这里g为engineering shear，mu=E/[2(1+nu)]。体积应力项lambda不影响VM；
仍必须在求位移时使用完整本构。内核融合梯度、VM平方和max归约，
不写出全体应力分布，只对最终最大值开方，并逐单元减去一个节点位移
以减轻大刚体平移下的数值抵消。当前代码使用单一mu；非均匀或各向异性
本构需扩展内核，但仍可在满足上述单元条件时扫描四角点。

3. 在任意接触下仍成立的优化

已实现并验证：
  • 缓存表面几何；用固定空间索引查询当前接触区域，再做一致载荷散射。
    当前cKDTree只索引几何求积点，没有预存接触/应力载荷基。
  • 固定数值矩阵的分解复用；优化消元排序，批量多右端项求解。
  • 遍历全域的融合最大应力内核；避免应力张量大数组及中间结果。
  • 完全相同的有效RHS可精确复用上一结果。比较的是完整f_eff，
    包括姿态变换、全部外载、角速度和惯性，不能仅比较夹爪状态。

同一A下，增量公式 y_t = y_(t-1) + A^(-1)(b_t-b_(t-1))、u_t=V y_t 是精确的；
其中b是缩减右端项，V将自由DOF嵌入完整位移空间。
本次验证了载荷突变和非对称情况。但一次增量回代通常仍需遍历整个因子；
小/稀疏Delta b并不自动让弹性影响局部化，也不保证减少回代耗时。
需要另行证明消元树可达性或采用经过残差检验的迭代校正，才能跳过相应工作。

未经验证不能承诺的优化：全表面ROM/Green函数/低秩近似、上一帧热点候选、
warm-start迭代、空间应力上界剔除。它们可以不限制接触，但应从完整当前RHS
检查误差；任意接触不保证响应低秩、接触变化小或热点不跳变。
在缩减系统中 r=b-A y_hat，可用能量误差 sqrt(r^T A^(-1)r) 加每点算子界，
构造当前FEM离散解的应力峰值上下界后做自适应筛选，
不包含未知接触分布或网格离散误差。精确计算该能量一般又需要一次校正求解；
要省成本，需有经过证明的预条件器/强制性下界，不能把经验残差当误差证书。

4. 新的完整物体验证

CPU：{checks['CPU']}，容器CPU quota为8核；BLAS单线程。
完整鸡蛋形空壳，约44×44×60 mm，厚0.5 mm；直边二阶四面体；
3210节点、9630DOF、1920四面体、7680角点应力样本。
该网格刻意较粗，未完成任意接触下的网格收敛，不作为真实峰值准确度结论。

52个变化输入快照：连续移动接触中心/方向/半径；位置与数量突变；随机1–4个
非对称接触；单侧载荷；零载荷；无接触旋转。没有使用对称子域。
这些是独立的代数快照，omega和反算alpha没有要求时间差分一致，
因此不是实际刚体动力学轨迹或弹性动力学验证。

全部检查通过：
  最大完整平衡相对残差：{residual:.3e}。
  增量法与参考峰值的最大差异（1 Pa floor）：{incremental_error:.3e}。
  全域融合峰值与原始完整strain/stress路径差异（1 Pa floor）：
    {bench['full_original_peak_error_1Pa_floor']:.3e}。
  两个非对称加载更换gauge后应力一致；能量一致性、六模态合力/力矩也通过。
  求积阶数10→14只检查两个加载，峰值变化分别约0.0024%和0.00067%；
    这是载荷积分敏感性检查，不是网格收敛。

任意向量traction接口也通过独立载荷组装与合力/力矩检查：
  每个表面求积点具有位置变化的三个traction分量；257个随机稀疏样本；
  重复样本累加；零输入。非零完整平衡残差最大
    {max(r['full_equilibrium_relative_nonzero'] or 0. for r in tractions['records']):.3e}。
它不限高斯压力、恒定方向或对称分布，但这里只覆盖存储的外表面求积点。
任意连续分布的积分精度、很小接触斑、内表面加载需另行离散/收敛检查。

测试中平滑patch保持指定合力，其力矩由实际积分载荷质心决定；
该质心通常不等于名义center_m。若从Genesis的点力扩展成有限patch，
应明确所用载荷分布或额外匹配原接触wrench，不能暗中改变力矩。

参考direct/incremental路径共享数值因子，主要验证不同路径一致；
完整残差、独立能量积分、替换gauge和独立应力内核提供额外检查。
另对73个合成P2单元、随机/二次弯曲/仿射/静水/刚体/零场验证10种内核配置；
非零应力相对差异不超过约7.85e-16，零场舍入应力小于1e-5 Pa。

5. CPU实测成本：时间范围必须区分

下列是独立、串行安排的性能测量，未让其它大型CPU测试并行争抢资源。
均为上述粗网格；预处理和JIT不计入逐帧时间；没有GPU、Genesis集成、策略
推理或RL训练。主算法正确性测试文件中的初步timings不作为主要性能数据，
应使用arbitrary_contact_benchmark_L2.json和arbitrary_contact_solve_results.json。

完整流程（表面载荷映射 + 离心惯性 + 六模态惯性释放 + 分解回代 + 全域最大应力）：
前者是缓存几何的完整NumPy strain/stress；后者为空间索引+串行融合峰值。
{pipeline_table}

随机接触19.292860→9.905151 ms，约1.95倍加速，约101次恢复/秒。
该数值是单CPU完整恢复调用的吞吐量，不是RL环境FPS或细网格性能。
本次完整流程沿用MMD排序；下列ND排序结果是独立测量，未混合成另一条实测流程。

接触映射单独时间（缓存几何全扫描 -> 空间索引当前区域）：
{mapping_table}

最大应力后处理单独时间（任意完整位移场，所有单元四角点）：
{post_table}
因此8线程0.0664 ms的优势仅是峰值后处理；完整流程仍由回代主导。
缓存全部P2角点梯度比现算多960 bytes/element，在本测量中没有优势。
内核线程测试为1/2/4/8；小模型不应把并行后处理收益直接当成总流程收益。

分解求解单独时间（不含载荷映射、惯性释放、最大应力）：
{solve_table}
B=32用16个不同随机载荷列循环扩展到32列，衡量同因子的块回代成本；
不等于运行32个完整RL环境。ND利用本例挤出薄壳网格结构做排序，
不依赖接触或对称性，但其它形状需要适合其图结构的ordering。

FP32分解+FP64残差迭代修正：8个任意加载都通过精度检查，各需4次修正；
平均求解时间{refinement_mean:.3f} ms，比FP64单次约7–8 ms更慢，当前不推荐。
这是薄壳病态系统上实际精度/成本结果，不能从“FP32带宽更小”推出必然加速。

已组装RHS上的上一帧缓存测试：相同RHS阶段有13/16命中，才能大幅降低平均成本；
连续移动和独立随机接触无命中。增量回代通常约7.8–8.1 ms，没有可靠省时。
这些排除了载荷映射和惯性释放，不能和上面的完整流程混为一谈。

6. GPU与下一步性能决策

当前环境没有CUDA GPU，本次未测试RTX6000，未给出新的GPU/RL吞吐量结论。
在该约束下GPU基础路线应为：所有环境共享完整几何/因子，动态组装B，
块多RHS求解，融合全域P2角点最大应力归约；不复制每个环境的分解。
预估必须先固定实际收敛网格、RTX6000型号/显存、批量大小和求解策略，
测量或标定该完整矩阵的factor存储/块回代，再计入Genesis、接触映射与策略成本。
全载荷空间稠密应力传递表可能耗费巨大存储，不能沿用三列应力响应表的成本。

当前最明确的优先级是：完成代表性任意接触的网格/载荷积分收敛；
随后优化完整矩阵的批量回代或残差控制的迭代解。峰值后处理已很便宜，
继续追求固定热点集合不是通用方案。若改用近似，必须注明容限及失效回退。

7. 复现

解压fixed_shape_arbitrary_contact_cpu.zip，在解压目录运行：
  python arbitrary_contact_cpu/arbitrary_contact_test.py
  python arbitrary_contact_cpu/traction_validation.py
  python arbitrary_contact_cpu/pipeline_benchmark.py
  python arbitrary_contact_cpu/solve_benchmark.py
  python arbitrary_contact_cpu/build_report.py
不要同时运行多个性能测试。结果写到arbitrary_contact_cpu/。
旧源文件仅作为网格/FEM/排序依赖，未修改；不要运行旧quarter/三载荷基流程
来代替本报告的完整物体验证。requirements.txt记录本环境所用依赖版本。

8. 可核对的相关原始资料

Altair官方Inertia Relief说明（自由结构外载与平动/转动惯性平衡，参考约束）：
https://2025.help.altair.com/2025/hwsolvers/altair_help/topics/solvers/os/inertia_relief_r.htm
Zheng & James, Rigid-Body Fracture Sound with Precomputed Soundbanks, 2010
（刚体加载与预计算准静态弹性恢复的相关路线）：
https://www.cs.cornell.edu/projects/FractureSound/files/fractureSound_comp.pdf
Yano, 2017, elasticity residual certification相关论文：
https://arrow.utias.utoronto.ca/~myano/papers/yano_2017_ecrb_elasticity.pdf

本报告的具体数值来自附带CPU实验；P2角点/VM融合及缓存适用范围的论证
是针对本离散模型的推导，不把上述资料作为本实验的验证证据。
"""
    (HERE/'fixed_shape_arbitrary_contact_report.txt').write_text(text, encoding='utf-8')
    print('Report built from recorded validation and benchmark results.')


if __name__ == '__main__':
    main()
