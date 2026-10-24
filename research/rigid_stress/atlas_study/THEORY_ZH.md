# 局部加密特征网格库：自包含理论推导

本文给出本次 CPU 实验所依据的数学模型、成立条件与误差界。完整英文证明、几何与积分误差项见 [THEORY.md](THEORY.md)。下述推导不预设网格库一定比粗网格准确或更快；这两点需要单独实测。

## 1. 固定物理问题与共同惯性释放

在固定连通物体域 \(\Omega\) 上，线性弹性满足
\[
\epsilon(u)=\operatorname{sym}\nabla u,\quad
\sigma(u)=C:\epsilon(u),\quad
a(u,v)=\int_\Omega\epsilon(v):C:\epsilon(u)\,dx.
\]
材料张量 \(C\) 在对称应变上正定。施加足够物理位移约束时使用相应的 \(H^1\) 空间；自由刚体使用商空间 \(V=H^1(\Omega;\mathbb R^d)/\mathcal R\)，其中 \(\mathcal R\) 是刚体运动空间。Korn 不等式保证 \(\|v\|_a=\sqrt{a(v,v)}\) 是商空间上的范数。数值 gauge 仅选择刚体运动代表，不能添加真实夹持。

单个接触通常不平衡。令 \(r_i\) 为刚体模态、\(m\) 为共同的物理质量双线性型，定义
\[
G_{ij}=m(r_i,r_j),\quad b_i(\ell)=\ell(r_i),\quad
\alpha=G^{-1}b,\quad
(P\ell)(v)=\ell(v)-\sum_j\alpha_jm(r_j,v).
\]
于是 \((P\ell)(r_i)=0\)，且 \(P\) 线性，因此 \(P\sum_c\ell_c=\sum_cP\ell_c\)。不同网格必须使用同一物理质量／惯性释放定义；各自的质量近似不同会产生额外载荷误差，不能当作 gauge 差异忽略。平衡总载荷的各单源释放应相互抵消。

下文把 \(\ell_c\) 记为已按同一规则处理的载荷泛函。共同体力不能每个接触重复加入。载荷须属于 \(V'\)：有限面积压力可满足该条件，理想点载荷则可能没有有限能量解，不能直接套用后面的界。

## 2. 响应叠加成立，混合网格解仍是近似

定义连续解算子 \(T\)：
\[
a(T\ell,v)=\ell(v)\quad(\forall v\in V).
\]
固定线性算子下，\(T\) 与应力映射 \(S:u\mapsto C:\epsilon(u)\) 都线性，故
\[
u=T\sum_c\ell_c=\sum_cT\ell_c,\qquad
\sigma=\sum_cST\ell_c. \tag{1}
\]
这里先固定每个接触的**实际分布牵引**；并没有假设接触区域预测、压力拟合或耦合变形接触求解也是线性的。

为每个局部加密网格建立同一物理域上的相容空间 \(V_j\)，定义 Galerkin 解 \(T_j\ell\in V_j\)：
\[
a(T_j\ell,v_j)=\ell(v_j)\quad(\forall v_j\in V_j).
\]
对接触 \(c\) 选取覆盖其真实压力盘的网格 \(j(c)\)，网格库近似为
\[
\widehat u=\sum_cT_{j(c)}\ell_c,\qquad
\widehat\sigma(x)=\sum_cST_{j(c)}\ell_c(x). \tag{2}
\]
每一项是该接触的**全物体响应**。若若干接触选择同一网格，可以先合并载荷再回代一次。不能让每张网格承受全部载荷后相加，否则重复加载。

**命题：**在上述共同物理域、共同双线性型、精确载荷积分和相容空间假设下，
\[
\left\|T\sum_c\ell_c-\widehat u\right\|_a
\le\sum_c\inf_{v_j\in V_{j(c)}}\|T\ell_c-v_j\|_a. \tag{3}
\]
**证明：**Galerkin 正交性给出 \(a(T\ell_c-T_j\ell_c,v_j)=0\)。因而 \(T_j\ell_c\) 是 \(T\ell_c\) 的能量正交投影，误差等于该空间的最佳逼近误差。对式 (1) 与 (2) 相减，再用三角不等式即得式 (3)。证毕。

这说明准确的单源响应可以组合，但不保证总误差单调下降。若单源相对误差不超过 \(\varepsilon\)，总相对误差只满足
\[
\frac{\|u-\widehat u\|_a}{\|u\|_a}
\le\varepsilon\frac{\sum_c\|T\ell_c\|_a}{\|T\sum_c\ell_c\|_a}. \tag{4}
\]
分式是响应相消放大因子；接近相消时应报告绝对误差及归一化尺度。**弹性场相消不等于总合力为零**：平衡的对向夹持仍可产生显著压应力场。

混合网格的每项只在自己的测试空间中平衡，通常不在联合加密空间中平衡。例如
\[
K=\begin{bmatrix}2&1\\1&2\end{bmatrix},\quad
f_1=(1,0)^T,\quad f_2=(0,1)^T,
\]
分别仅在 \(\operatorname{span}(e_1)\)、\(\operatorname{span}(e_2)\) 求解，得到的和是 \((1/2,1/2)^T\)，完整解却为 \((1/3,1/3)^T\)。因此“线性响应可叠加”不能被写成“多网格近似等于联合加密解”。

在共同有限维参考空间中，令 \(r=f-K\widehat u\)，有精确恒等式
\[
\|u-\widehat u\|_K^2=r^TK^{-1}r. \tag{5}
\]
只检查粗空间残差可能漏掉缺失的细尺度分量。曲面几何、压力映射或积分不同还会引入一致性误差，详见英文式 (6)，不能全部归为“网格加密误差”。

## 3. 粗解加细化改变量：严格代数等价

令 \(T_0\) 为共享粗网格算子。线性性直接给出
\[
\widehat\sigma
=ST_0\!\left(\sum_c\ell_c\right)
+\sum_cS(T_{j(c)}-T_0)\ell_c. \tag{6}
\]
保留全域改变量时，式 (6) 与式 (2) 完全等价；它避免把粗响应重复计入，并允许离线存储或压缩改变量。这是同一 FEM 问题的数值细化差值，不是另加解析 Hertz 应力。

若只保留接触附近的 \(\chi(x)\delta\sigma(x)\)，便引入新的截断误差与平衡残差：
\[
\operatorname{div}(\chi\delta\sigma)
=\chi\operatorname{div}\delta\sigma+\delta\sigma\nabla\chi.
\]
若截断的是位移，正确应变则为
\[
\epsilon(\chi\delta u)
=\chi\epsilon(\delta u)+\operatorname{sym}(\delta u\otimes\nabla\chi).
\]
不能忽略这些过渡项，也不能预先断言薄壳、U 型或细颈结构的改变量远场很小。相容局部基函数加共同变分求解，是更一致的局部化方式。

## 4. 接触位置混合与空间混合是两件事

代表点只负责选择细化区；实际压力仍放在真实位置。对同一接触参数 \(p\)，可取非负权重 \(w_j(p)\)、\(\sum_jw_j=1\)，定义
\[
\widehat u(p)=\sum_jw_j(p)T_j\ell(p).
\]
若第 \(j\) 项能量误差界为 \(E_j(p)\)，则
\[
\|T\ell(p)-\widehat u(p)\|_a\le\sum_jw_j(p)E_j(p). \tag{7}
\]
连续权重及连续载荷给出能量意义的连续响应；硬切换时，两张网格在同一载荷下的跳变不超过 \(E_i+E_j\)。这只是连续性／误差界，不能保证峰值已准确。

上述权重随**接触参数**变化，在单个接触的一次响应中不随观测位置 \(x\) 变化，故没有空间 \(\nabla w\) 项。对物体不同区域用 \(\chi_j(x)\) 拼接响应，才会出现上一节的附加项。

只插值不同锚点的应力快照，还近似了载荷位置本身，可能抹平或分裂尖峰。保存较大片细化区的分解，在其中积分真实压力盘，比把接触吸附到锚点更可靠。

## 5. 半径、方向与摩擦的条件

Hertz 半径满足 \(a=[3F_n/(4E^*\bar\kappa)]^{1/3}\)。改变力通常也改变载荷形状，因此一条“单位接触力”快照不能直接按力缩放。原 DPR 的锚点与非负压力拟合还可能随合力方向改变。

正确顺序是：先生成当前真实牵引／节点广义载荷，再应用固定线性恢复算子。只有载荷的空间形状固定且对力分量线性时，三个方向的单位响应才足够。预存节点响应列或分解不需要假设接触位置或半径固定。

薄壳面侧 \(s=\pm1\) 上的摩擦必须保留一致偏心力矩：
\[
f_i=\int N_i t\,dA,\qquad
m_i=\int N_i(s h d_i/2)\times t\,dA.
\]
不同网格的应力须在同一物理点计算，并转到共同坐标系后相加；内外表面也必须对应。数组下标、各自 Gauss 点最大值或不同的节点应力平均不能直接对比。

## 6. 静态凝聚及共享粗结构

将一张选定 FE 系统分成接口／粗自由度 \(\Gamma\) 与细区域内部自由度 \(I\)。消去内部未知量得到
\[
S_\Gamma=K_{\Gamma\Gamma}-K_{\Gamma I}K_{II}^{-1}K_{I\Gamma},\quad
g_\Gamma=f_\Gamma-K_{\Gamma I}K_{II}^{-1}f_I,
\]
\[
S_\Gamma u_\Gamma=g_\Gamma,\qquad
u_I=K_{II}^{-1}(f_I-K_{I\Gamma}u_\Gamma). \tag{8}
\]
这是该离散系统的精确消元。内部因子、接口映射和应力恢复算子可预计算，但仍须保留接口耦合。重叠细化区不能重复加入材料刚度。

进一步在共同相容空间中写 \(u=V_0q+W_Az\)，其中 \(V_0\) 是共享粗基、\(W_A\) 是活动局部基。消去粗变量得到
\[
(K_{AA}-K_{A0}K_{00}^{-1}K_{0A})z
=b_A-K_{A0}K_{00}^{-1}b_0. \tag{9}
\]
可离线保存 \(K_{00}\) 的分解及候选交叉块，在线只求小活动系统。即便局部支撑互不相交，经过全局粗逆后的交叉耦合仍可能非零，不能默认只保留对角块。这种共同 Galerkin 模型与独立单源响应求和的近似不同。

## 7. 最大应力误差需要额外条件

对对称张量 \(A,B\)，由偏应力投影与 Weyl 不等式，
\[
|\mathrm{VM}(A)-\mathrm{VM}(B)|\le\sqrt{3/2}\|A-B\|_F,
\quad |\lambda_{\max}(A)-\lambda_{\max}(B)|\le\|A-B\|_F.
\]
故在**共同有限采样集** \(X\) 上，
\[
|\max_Xq(\widehat\sigma)-\max_Xq(\sigma)|
\le L_q\max_X\|\widehat\sigma-\sigma\|_F. \tag{10}
\]
必须先叠加张量再计算 VM／主应力／最大值。不能相加各源最大值，也不能只扫描各源自己的接触区；远处细颈可能才是全局最大处。

能量误差仅控制柔度加权的 \(L^2\) 应力误差，不推出统一的 \(L^\infty\) 应力界。连续最大值还需要场的正则性与采样分辨率。尖角、突变牵引或理想点力可使应力奇异；有限压力盘也不能消除所有几何奇异性。纯壳加密不能补回壳理论缺失的三维法向应力。

## 8. 理论与 CPU 证据如何对应

以下是对应关系，不代表每条理论条件都已在任意几何上验证；实际范围、失败和参考不确定性以各 JSON 及最终总报告为准。

| 要检查的命题 | 本轮 CPU 入口／记录 |
|---|---|
| 同矩阵线性、混合空间反例、完整改变量、相消、峰值界 | [algebra.py](algebra.py)／[algebra_results.json](algebra_results.json) |
| 相容离散和基础能量／载荷行为 | [affine_control.py](affine_control.py)／[affine_control.json](affine_control.json) |
| 共同质量、自由刚体与惯性释放 | [freebody_control.py](freebody_control.py)／[freebody_control.json](freebody_control.json) |
| 消元等价及共同离散的局部接口 | [condensation_check.py](performance/condensation_check.py)／[condensation.json](performance/condensation.json) |
| 真实三维移动接触及源参数混合 | [volume/study.py](volume/study.py)／[switch_control.py](volume/switch_control.py) |
| 壳面内响应、曲面鸡蛋补充控制 | [plate_atlas.py](shell/plate_atlas.py)／[egg_supplement.py](shell/egg_supplement.py) |
| 非凸细颈及远离接触的应力 | [nonconvex/study.py](nonconvex/study.py)／[结果](nonconvex/results.json) |

需要分别报告：实现数值一致性、网格库近似误差、参考离散不确定性、载荷模型误差、采样误差，以及 setup／在线成本和共享存储。平板、实体块和二维细颈控制不能自动认证自由三维鸡蛋；未运行的空间截断或连续全域峰值检查不能写成已通过。

## 9. 相关一手工作

预计算多分辨率线弹性响应可参考 [James–Pai 的 Multiresolution Green's Function Methods](https://www.cs.cornell.edu/~djames/papers/JamesPai02TOG.pdf)；相容局部空间可参考 [Babuska–Melenk 的 Partition of Unity Method](https://ivo100.oden.utexas.edu/assets/papers/ijnme97.pdf)；离线组件库和在线接口耦合可参考 [Huynh–Knezevic–Patera 的 Static Condensation Reduced Basis Element Method](https://www.numdam.org/item/M2AN_2013__47_1_213_0/)，DOI 10.1051/m2an/2012022。这些工作支持方法基础，但不替代本项目的应力峰值与吞吐验证。
