"""Build the self-contained CPU shell-DPR study report from measured results.

Run from the research package parent: python -m shell_cpu.make_report.
Large generated field NPZs are reproducible inputs, not committed benchmark data.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'


def read(name):
    return json.loads((ROOT / name).read_text())


def stress_figure():
    OUT.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(13, 8), layout='constrained')
    cases = [('automatic_normal_n8', 'Normal, n=8 / 2,316 DOF'),
             ('automatic_normal_n24', 'Normal, n=24 / 20,748 DOF'),
             ('automatic_friction_n24', 'Friction, n=24 / 20,748 DOF'),
             ('nonconvex_torus', 'Nonconvex inner contact / 1,728 DOF')]
    for i, (name, title) in enumerate(cases):
        d = np.load(ROOT / 'results/egg_final' / (name + '.npz'))
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        v, q = d['vertices'] * 1000, d['quads']
        values = d['vm'].max(axis=(0, 2)) / 1000
        limit = 130 if i < 3 else 31
        norm = Normalize(0, limit)
        faces = Poly3DCollection(v[q], facecolors=plt.cm.inferno(norm(values)),
                                 edgecolor='none', linewidth=0, rasterized=True)
        ax.add_collection3d(faces)
        lo, hi = v.min(axis=0), v.max(axis=0)
        mid, span = (lo + hi) / 2, max(hi - lo) * .53
        ax.set(xlim=(mid[0]-span, mid[0]+span),
               ylim=(mid[1]-span, mid[1]+span),
               zlim=(mid[2]-span, mid[2]+span),
               xlabel='x (mm)', ylabel='y (mm)', zlabel='z (mm)')
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.view_init(elev=16 if i < 3 else 52, azim=-64 if i < 3 else -60)
        ax.set_title(title + f'\nsampled peak = {values.max():.2f} kPa', fontsize=11)
        mapper = plt.cm.ScalarMappable(norm=norm, cmap='inferno')
        fig.colorbar(mapper, ax=ax, shrink=.64, pad=.015, label='shell VM (kPa)')
    fig.suptitle('Actual CPU fields: top/bottom in-plane stress envelope\n'
                 'Each face shows its largest four-point sample; no 3D contact stress', fontsize=14)
    path = OUT / 'shell_stress_fields.png'
    fig.savefig(path, dpi=155)
    plt.close(fig)
    return path


def convergence_figure():
    rows = read('results/egg_final/summary.json')['rows']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout='constrained')
    names = [('automatic_normal', 'automatic / normal'),
             ('automatic_friction', 'automatic / friction'),
             ('automatic_asymmetric', 'automatic / asymmetric'),
             ('manual_2mm_normal', 'manual 2 mm / normal')]
    for key, label in names:
        r = [x for x in rows if x['case'] == key and x['valid']]
        axes[0].plot([x['dofs'] for x in r], [x['peak_vm_pa']/1000 for x in r],
                     'o-', label=label, ms=4)
        axes[1].plot([x['dofs'] for x in r[1:]],
                     [100*x['adjacent_energy_change'] for x in r[1:]], 'o-', label=label, ms=4)
    axes[0].set(xscale='log', xlabel='shell DOF', ylabel='sampled VM peak (kPa)',
                title='Egg peaks remain mesh dependent')
    axes[1].set(xscale='log', xlabel='shell DOF', ylabel='adjacent change / finer value (%)',
                title='Strain energy is also not converged')
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(OUT / 'egg_convergence.png', dpi=155)
    plt.close(fig)
    return OUT / 'egg_convergence.png'


def embedded(path, caption):
    src = base64.b64encode(path.read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{src}" alt="{html.escape(caption)}"><figcaption>{caption}</figcaption></figure>'


def table(headers, rows):
    return '<div class="table"><table><thead><tr>' + ''.join(f'<th>{x}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{x}</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def build(destination, revision='rigid-stress-recovery'):
    stress = stress_figure()
    conv = convergence_figure()
    egg = read('results/egg_final/summary.json')
    perf = read('results/performance_confirmation.json')
    bem = read('contact_reference_results/refinement1280/results.json')
    meshes = sorted({x['n'] for x in egg['rows']})
    egg_rows = []
    for n in meshes:
        r = {x['case']: x for x in egg['rows'] if x['n'] == n}
        egg_rows.append([n, f"{r['automatic_normal']['dofs']:,}"] +
                        [f"{r[k]['peak_vm_pa']/1000:.3f}" for k in
                         ['automatic_normal','automatic_friction','manual_2mm_normal','automatic_asymmetric']])
    methods = [('reused_lu_full_local','预分解 LU＋完整局部场'),
               ('column_loop_lu_full_local','逐 RHS 循环 LU＋完整局部场'),
               ('compact_surface_full_local','紧凑精确响应算子＋完整局部场'),
               ('scalar_hinted_fast_transfer','标量 witness 快速载荷转移'),
               ('batched_witness_fast_transfer','批量 witness 快速载荷转移'),
               ('batched_transfer_plus_lu_full_local','批量转移＋LU＋完整局部场'),
               ('batched_transfer_plus_surface_full_local','批量转移＋紧凑响应＋完整局部场'),
               ('batched_transfer_plus_surface_peak','批量转移＋紧凑响应＋VM 最大值')]
    perf_rows=[]
    for key, title in methods:
        a,b = [x['methods'][key] for x in perf['rows']]
        perf_rows.append([title, f"{a['median_ms']:.3f}", f"{b['median_ms']:.3f}",
                          '–'.join(f'{x:.3f}' for x in b['range_ms'])])
    bem_rows = [[f"{r['force_N']:.3f}", f"{r['hertz_radius_m']*1e6:.3f}",
                 f"{r['radius_relative_difference']*100:.4f}%",
                 f"{r['pressure_relative_continuous_L2']*100:.3f}%",
                 f"{r['indentation_relative_difference']*100:.5f}%"] for r in bem['rows']]
    precision_path = ROOT/'review_results/gauge_checks.json'
    precision_extra = ''
    if precision_path.exists():
        precision_extra = '<details><summary>补充 gauge / FP32 检查原始结果</summary><pre>' + html.escape(precision_path.read_text()) + '</pre></details>'
    body = '''
<header><p class="eyebrow">CPU RESEARCH · FIXED LINEAR SHELL + AUTOMATIC RADIUS + DPR</p>
<h1>薄壳应力恢复：算法、实测与适用边界</h1>
<p>2026-10-24 · Genesis rigid-stress-recovery 分支研究原型 · CPU 实测，无 GPU 外推</p></header>
<section class="lead"><h2>结论</h2>
<p>固定形状的线性薄壳可以沿用“刚体动力学 → 自动接触半径 → 有限压力载荷 → 预分解 FEM”的架构。CPU 原型通过了刚体模态、膜、弯曲、薄板、球壳及曲面弯曲控制；与独立开源壳 FEM 的误差随细化减小。批量转移与精确响应算子均有实际提速。</p>
<p><strong>但是，这还不能把夹蛋的三维最大应力交给薄壳代替。</strong>实际 Hertz 半径约 45 μm，厚度 500 μm：接触局部尺度小于厚度，也远小于当前面片。鸡蛋峰值与能量仍未收敛，薄壳只输出面内膜＋弯曲应力，不能给出局部法向压应力或三维接触最大值。</p></section>
<nav><a href="#algorithm">算法</a><a href="#checks">力学验证</a><a href="#contact">独立接触参考</a><a href="#egg">鸡蛋结果</a><a href="#performance">CPU 优化</a><a href="#precision">FP32</a><a href="#integration">迁移建议</a><a href="#reproduce">复现</a></nav>
<section id="algorithm"><h2>1. 固定线性壳＋DPR 的实现</h2>
<p>每节点采用三个位移和三个微小转角，使用含曲率项的双线性曲面 MITC4 类单元。刚度由膜刚度 hC、弯曲刚度 h³C/12、横剪 5Gh/6 与弱一致 drilling 稳定项组成。参考曲面、厚度、材料固定时，K、应力映射 S、刚体模态和质量投影都只建立一次。</p>
<pre>X(ξ,η,z) = Σ Nᵢ xᵢ + z Σ Nᵢ dᵢ
u(ξ,η,z) = Σ Nᵢ uᵢ + z Σ Nᵢ (θᵢ × dᵢ)
G₀ = D₀ A₀⁻¹ ; G₁ = (D₁ − G₀ A₁) A₀⁻¹
σ± = N/h ± 6M/h²</pre>
<p>G₁ 中的曲率修正使任意弯曲 Q4 单元上的刚体旋转保持零应变。MITC 横剪采用四个边中点的协变剪应变插值。返回四个面内 Gauss 点的内外表面应力张量，以及 N、M、Q；Q 是横剪合力，并未混入三维接触应力。</p>
<h3>每次更新</h3>
<ol><li>把刚体碰撞器给出的点、法向、合力和几何 witness 转入物体参考坐标。只依赖固定形状，不假设固定接触点、对称或仅法向力。</li>
<li>利用预计算主曲率及 Fn 估算圆形 Hertz 半径：a=[3Fn/(4E*κ̄)]¹ᐟ³，刚性对面时 E*=E/(1−ν²)。κ̄ 是两个组合主曲率的平均值。这是圆形近似；未实现一般椭圆 Hertz 解。</li>
<li>将接触法向平面内的有限 Hertz 压力盘沿合力方向投影到真实偏置壳面。通过形函数积分得到等效节点力及力偶。切向力沿相同压力权重分配；只恢复已给定摩擦合力，不重新求解粘滑。</li>
<li>对非平衡载荷作明确的六模态惯性释放，求 Kq=P f；已平衡的双侧载荷保持原样。摩擦上提及非对称案例的惯性释放系数均写入数据。它是瞬时准静态恢复，不是弹性波传播或耦合变形接触求解。</li>
<li>经已缓存 LU 多 RHS 回代，或应用精确离散响应 H=SK⁻¹P，得到完整场；最大值模式再作归约。</li></ol>
<h3>摩擦的偏心力矩不能丢</h3>
<pre>δu_surface = Σ Nᵢ [δuᵢ + z(δθᵢ × dᵢ)]
fᵢ = ∫ Nᵢ t dA ; mᵢ = ∫ Nᵢ z(dᵢ × t) dA</pre>
<p>内外面 z=±h/2。该映射已通过合力、力矩及虚功检查。表面力偶可以精确折入响应算子，使每一侧只需 3N 个力输入列，而不是 6N 个通用力／力偶列；同时有内外接触时用两套映射。</p>
<div class="note"><strong>模型差异：</strong>这里实现的是 projected-Hertz DPR。现有 Genesis DPR 在曲面上使用 surface-ball 支撑与拟合，两者在平面控制上吻合，但在一般曲面上不是同一个载荷离散。CPU 结果不能宣称与当前 native DPR 逐点数值等价；迁移时必须明确选择并建立匹配测试。</div>
<h3>额外验证：保留原生 DPR 的压力载荷律</h3>
<p>另写 <code>native_dpr_control.py</code>，直接导入分支未修改的 <code>wrench.py::PadPressureFit</code>，保留 surface-ball Hertz 权重、实际曲面面积、非负仿射拟合与原 wrench 预算，再接一致的壳节点力／力偶。平板及 n4/n8 鸡蛋、纯法向／Ft/Fn=0.3、自动／手工 2 mm 共 12 案例全部有效。</p>
<p>与主 projected-Hertz 原型的完整壳应力差：平板 ≤2.50×10⁻¹²，自动小半径 ≤4.63×10⁻⁸，2 mm 宽盘的 n4/n8 分别 ≤8.09×10⁻⁵／2.664×10⁻⁴（0.02664%）。原生载荷律 q24→q48 的应力变化最大约 1.03×10⁻⁶。这是同一网格所有内外表面采样张量的未加权 Frobenius 相对范数。</p>
<p>因此可以保持成熟 DPR 的载荷律、只替换壳恢复后端。该补充控制仅扩展到所选 Q4 壳面侧：没有认证生产 Quadrants/P2 的逐位一致性，也没有认证体网格全部内外边界同时进入支撑球的语义。自动半径采用同样给定曲率 50/70 m⁻¹、Fn=0.05 N，a=38.456 μm；它是隔离载荷律差异的控制，不是主鸡蛋几何曲率或独立接触真值。</p>
<h3>快慢路径与非凸支持</h3>
<p>完整落在单一仿射 Q4 内的压力盘用精确零阶／二阶矩。非仿射完整盘采用嵌入 16/36 点规则，并检查差异；夹边或多面片的压力盘使用分片极坐标积分、角端点变换和有限的 16→32→64 自适应。自适应只增加积分精度，不扩大物理半径。超过力矩／积分有效性预算即明确失败。</p>
<p>当前 witness 需重新验证坐标、法向、接触面侧及盘包含性；失效后重新查找并限制在局部连通材料片，防止非凸模型加载到对侧。U 型双壁载荷转移、环面内侧和混合内外面已验证。对象本身凹并不等于 Hertz 无效：组合曲率仍可为正；若出现非正组合曲率，则自动 Hertz 路径明确拒绝，尚无物理认证的廉价通用半径估计。</p>
</section>
<section id="checks"><h2>2. 力学与独立 FEM 验证</h2>
<p>控制检查共 13 项，全部选定控制通过。初次测试发现夹边积分失准，保留失败数据并修正端点积分；所有受影响接触检查及最终完整控制套件均在冻结后的源码上重新运行。没有以更大 patch 掩盖载荷转移问题。</p>
'''
    body += table(['检查','实测结果','能说明什么'], [
        ['刚体与能量','恰好 6 个零模；无额外负模；刚体旋转算子误差 1.31×10⁻¹⁵；膜／弯曲能量误差约 10⁻¹⁶／10⁻¹²','离散运动学、刚度基本一致'],
        ['Navier 薄板挠度','h/L=0.001，n=4/8/16：2.30% → 0.514% → 0.125%','未发现明显横剪锁定'],
        ['曲圆柱二阶弯曲','h/R=0.005，n=16/32/64：4.66% → 1.13% → 0.279%','曲面弯曲控制随细化收敛'],
        ['球壳均压解析','独立套件 n=16，平均膜应力误差 0.317%','平滑壳载荷通过'],
        ['独立 pyfe3d 0.10.0 Quad4','球壳 n=16 完整表面张量 L² 差 1.60%；另组鸡蛋 n=24 为 8.08%','不同壳离散趋近；不是接触真值'],
        ['独立解析应力场','Navier n=4/8/16/32：完整张量 L² 18.61% → 9.38% → 4.71% → 2.38%','验证场而非只验证一个最大值'],
        ['接触／摩擦／非凸','内外面偏心、虚功、旋转不变性、卸载、跨边、顶点、U 型内壁、失效 witness','不依赖夹持对称性']])
    body += embedded(ROOT/'review_results/shell_validation.png','解析与不变性测试；不同控制使用不同几何和误差定义。')
    body += embedded(ROOT/'reference_results/independent_validation.png','独立 pyfe3d 壳 FEM 比较。该图的鸡蛋控制为 18/18/26 mm、taper=+0.15、Fn=0.2 N，与下节主鸡蛋案例不同，数值不能交叉混用。')
    body += '''<p>独立对照使用实际运行的 <a href="https://saullocastro.github.io/pyfe3d/quad4.html">pyfe3d 0.10.0 Quad4</a>（Hughes 混合积分、独立 Cython 刚度装配），没有复用本原型的 MITC4 刚度。<a href="https://mooseframework.inl.gov/moose/modules/solid_mechanics/ShellElements.html">MOOSE 壳文档</a>用于核对壳运动学与平面应力假设。本轮环境没有 Code_Aster；OpenSees 的导入在 MPI 初始化失败，因此没有把二者列为已运行参考。</p></section>
<section id="contact"><h2>3. 自动半径的独立接触控制</h2>
<p>为避免把 Hertz 公式同自身比较，另写了轴对称弹性半空间边界元。输入抛物面间隙 r²/(2R)、材料及总力，通过 Green 核和接触互补主动集联合解出非负压力、接触支撑及压入量；求解阶段不输入 Hertz 半径或压力曲线。R=20 mm、E=10 GPa、ν=0.3，最细 N=1280。</p>'''
    body += table(['Fn (N)','Hertz a (μm)','半径差','连续压力 L² 差','压入量差'],bem_rows)
    body += '''<p>压力误差为常值环单元内连续积分，较只在环中点比较更严格；N=640→1280 时三组压力误差分别从 2.498/1.052/0.618% 降到 1.270/0.631/0.326%。离散合力和互补残差约 10⁻¹⁵，最细网格的节点间最大穿透／压入量为 7.26×10⁻⁵。扩大域及提高积分阶次也已检查。</p><div class="note">这验证了半空间适用范围内的 Hertz 估算。它没有验证薄壳整体柔度参与接触后的真实鸡蛋接触区域，更没有证明一般凹面或强椭圆接触适用。</div>'''
    body += embedded(ROOT/'contact_reference_results/halfspace_contact_control.png','独立主动集半空间接触解；误差、域截断与积分误差分开记录。')
    body += '''</section><section id="egg"><h2>4. 夹蛋、摩擦和非对称载荷</h2>
<p>主案例：外形半轴 22/22/30 mm，taper=−0.18；沿法向内偏 h/2 得到中面；h=0.5 mm，E=10 GPa，ν=0.3。两侧各 0.05 N，正常夹持的自动半径 45.472 μm（a/h=0.09094）；摩擦案例 Ft/Fn=0.25；非对称案例有三个不同位置、力和切向分量。2 mm 是人工宽载荷控制，不能当作自动 Hertz 结果。</p>'''
    body += table(['面细分 n','DOF','自动法向 kPa','自动摩擦 kPa','手工 2 mm kPa','自动非对称 kPa'],egg_rows)
    body += '''<p>24 个主案例载荷转移均通过数值有效性检查，解的残差约 10⁻¹²。可是 n=16→24 时自动法向峰值仍从 77.77 升至 128.84 kPa，增加 65.7%（以上一级为分母；以细网格为分母则 39.6%），应变能变化约 27.8%（以细网格为分母）。<strong>不能宣布鸡蛋应力收敛。</strong></p>
<p>四点采样的最大值还不是每个单元内的连续最大值。一个完整且居中的压力盘落在正交平面双线性 Q4 单元内时，形函数的压力平均值恰等于盘中心值，因此许多不同的微小半径产生同样的等效节点载荷。这是低阶离散分辨率限制，不能解释为半径无需准确或接触已经解析。</p>
<p>还有独立的模型误差：按该 Hertz 近似计算，接触中心压力 p₀=3Fn/(2πa²)≈11.55 MPa，而壳的面内输出把 σnn 设为零。p₀ 与 VM／最大拉应力不是同一个指标，不能直接算误差倍数；但它说明缺失的三维分量具有实际量级。即使面内壳解收敛，也不等于完整三维接触应力已正确。</p>'''
    body += embedded(stress,'直接从 CPU 求解保存的完整场绘图。颜色为每面片四个采样点、内外表面的 VM 包络，未显示三维 Hertz 接触应力。前三图使用相同色标。')
    body += embedded(conv,'不是所有曲线都已进入渐近收敛区；宽载荷控制也未被当作收敛真值。')
    body += '''<h3>三维 P2 体 FEM 交叉比较</h3>
<p>调用分支现有的 P2 四面体 CPU oracle，与壳使用同一组球面顶点及厚度。曲面 Q4／director 与直边四面体的几何插值仍不同，不能把任一粗解当真值。均匀内压的 3D Lamé 控制、单／双厚度层，以及局部有限压力均已运行。</p>'''
    body += table(['载荷','细化结果','解释'],[
        ['自动半径约 41 μm，a/h≈0.082','n=8：公共面内 Voigt 应力 L² 差 61.2%；中面位移差约 19%','微小局部载荷没有通过壳／体一致性验证'],
        ['手工宽盘 5 mm，a/h=10','n=8/12/16：应力差 34.2/22.0/15.7%；位移差 9.40/4.63/2.53%','大尺度响应逐渐接近，仍非收敛参考'],
        ['宽盘 n=16 的采样峰值','壳 30.335 kPa、体 31.303 kPa，差 3.09%','峰值接近不代表完整场合格：场差仍 15.7%']])
    body += '''<p>此处应力指标是面积加权 (xx,yy,xy) 等权 Voigt 向量 L²，非完整张量 Frobenius 范数，也不是现有体 DPR 报告中其他接触误差的同一指标。所有比较保持原有限压力载荷，无解析应力叠加。</p>
<p>另做了相同球面顶点、共同面内应力输出的 CPU 成本对照：n=8 时壳 2316 DOF、P2 体 13842 DOF，B32 缓存求解＋公共应力采样分别 40.71／99.10 ms（约 2.43×）；n=4 为 3.75／18.30 ms。该比较不含载荷转移或全体积三维输出，几何插值与误差也不同，不能称为同精度加速倍数。壳的 Python 装配初始化在这些小案例中更慢，初始化成本已单独记录。</p></section>
<section id="performance"><h2>5. CPU 效率：实际跑过的优化</h2>
<p>EPYC 9V74、NumPy 2.3.5、SciPy 1.17.0、OpenBLAS／OMP 单线程。主计时使用 588 DOF / 96 个 Q4、7 次正反交错测量，其他重验证作业结束后运行。下表单位是每批毫秒；不包括建模／分解、刚体、Franka、策略或渲染，不是 Genesis FPS 或 GPU 吞吐。</p>'''
    body += table(['操作','B1 ms','B128 ms','B128 min–max ms'],perf_rows)
    body += '''<p>批量 witness 载荷转移相对标量约 17.0×；精确紧凑响应恢复相对批量 LU 约 2.52×；包括转移后的整体恢复模块约 1.54×。完整场差 8.55×10⁻¹²，批量／标量载荷差 2.38×10⁻¹⁶。跨边、顶点、卸载、内面和失效 witness 的混合测试实际触发回退，并与标量一致。</p>
<p>紧凑响应矩阵每侧存储量在 336/588/1308 DOF 分别为 1.74/5.42/27.12 MB，构建约 13/26.7/254 ms，多个同形环境共享，不应每环境复制。它仍随模型大小近似二次增长；大网格、接触列并集大或 B 很小时，列缓存和稠密响应可能不划算，完整扫描保留了这些负例。</p>
<p><strong>只求最大值在本 CPU 原型中没有确认提速。</strong>同一离散最大值仍需计算候选点，应力后的 VM 归约反而增加开销。分块最大值可以减输出存储，但 Python tile 循环不是 GPU 融合 kernel，不能据此推算 GPU 性能。</p>
</section><section id="precision"><h2>6. FP32：先修约束与缩放，再决定精度</h2>
<p>原六个 QR gauge 会把三个旋转约束集中在一个节点，导致对角平衡后的条件数估计仍约 7.77×10⁸（588 DOF）和 1.47×10¹⁰（2316 DOF）。旧 gauge 下，后者的直接 FP32 应力误差 60.4%，平衡 FP32 仍 16.1%；朴素混合精度迭代修正在该例发散。GMRES 修正能恢复精度但远慢于 FP64，未作为速度优化推荐。</p>
<p><strong>实际找到的优化：</strong>在按物体尺度归一化的刚体模态上，仅从分散的平移自由度选择六个约束，最终复验的条件数估计降至 1.27×10⁴／3.54×10⁴。保持相同载荷时 FP64 应力与旧 gauge 的相对差仅约 3×10⁻¹¹／4×10⁻¹⁰。FP32 因子加一次 FP64 残差修正后，完整场差分别为 5.35×10⁻¹¹／1.03×10⁻¹⁰。不能把原来的 FP32 失败简单归因于 E=10 GPa；统一缩放 E 本身不会改变条件数。</p>
<p>速度归因也做了对照：2316 DOF、8 RHS，原单位 FP64 求解＋应力为 11.97 ms，对角平衡 FP64 为 <strong>3.14 ms</strong>，一次／两次 FP32 因子修正为 5.55／8.46 ms。LU 非零数从 3.68M 降至 0.664M；此处最有效的是 gauge／缩放和减少 fill，不能把收益归于 FP32。本段是单独的补充试验，未把增益乘到上一节旧 gauge 的计时上。</p>
<p>另一条经过验证的路径是 FP64 构建响应，FP32 存储和乘积：588 DOF 的完整场差约 1.14×10⁻⁷；FP32 存储、选列 FP64 累加约 2.19×10⁻⁸。这只认证应力响应误差，不自动认证位移或平衡残差。没有运行 GPU TF32，也未把几何、曲率与接触全部改为 FP32。</p>'''
    body += precision_extra
    body += '''</section><section id="integration"><h2>7. 怎样接回 Genesis／Quadrants</h2>
<p>建议把壳作为 rigid stress recovery 的可选恢复后端，保留刚体动力学和 DPR 管线。当前 CPU 代码用于验证，不是新的生产动力学 solver。原体 FEM 保留用于厚区、局部三维应力和对照。这里不改变 native 默认配置。</p>
<ol><li><strong>静态共享：</strong>每个物体形状预计算曲面几何、曲率／局部曲率表、拓扑邻接、应力算子、约束与分解。相同形状跨环境共享，运行时数组仅保留环境状态、载荷和必要输出。</li>
<li><strong>先移植有效优化：</strong>Quadrants 内复用碰撞 witness，在连续接触中验证并更新，批量执行完整盘规则；跨面／无效环境压缩到独立慢路径，保持最终合力、力矩与接触片身份检查。</li>
<li><strong>小模型优先响应算子：</strong>紧凑表面响应将摩擦力偶折入列，按当前活跃节点的列分组；输出点×环境分块，完整模式写局部张量，最大模式融合主应力／VM 与归约，避免完整张量往返。需要全局张量时再转坐标。</li>
<li><strong>大模型保留分解：</strong>响应矩阵的二次存储会失去优势；改用多 RHS 分解回代和稀疏应力应用。两种执行后端共享相同的载荷与验收算子，基于实测规模选择。</li>
<li><strong>精度从良好 gauge 开始：</strong>优先分散平移 gauge＋对角平衡，再比较 FP32、FP32 响应表和混合精度。把误差与峰值、完整场、力学残差同时记录，不能直接把 FP64 结论套到 TF32。</li>
<li><strong>历史只缓存不改变问题：</strong>复用上一帧的 witness、局部 sheet、投影初值和已验证积分类型；卸载／reset 清空。半径、位置、切向力持续更新，超阈值时重算。按需响应列是固定离散算子的精确缓存，不依赖固定夹持位置。</li></ol>
<p>这些迁移项中，固定分解、多 RHS、精确响应、单侧紧凑映射、完整盘批量执行及混合回退已在 CPU 实现验证；GPU 融合、图捕获、压缩调度和真实完整场景收益仍是待测项。</p>
<div class="note">建议的实际用途是大尺度膜／弯曲应力和低成本风险信号。当前 tiny-Hertz 夹蛋不能据此宣称“精确最大应力”。下一步应固定应力定义与误差预算，先证明合适尺度上壳后端有用，再投入 native GPU；不要以加大接触盘冒充准确，也不需要回到昂贵自动接触求解或解析叠加路线。</div>
</section><section id="reproduce"><h2>8. 源码与复现</h2>
<p>研究源码及精简结果位于分支 <code>research/rigid_stress/shell_cpu/</code>。保留初始失败、参考可用性、源码散列、随机／确定性载荷及计时范围。主验证运行使用 legacy gauge；新增 translation gauge 为显式可选项，附加测试单独记录，历史计时不重贴为新 gauge。</p>
<pre>python -m pip install numpy scipy matplotlib threadpoolctl pyfe3d==0.10.0
cd research/rigid_stress
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -m shell_cpu.review_checks
python -m shell_cpu.reference --suite plate --levels 4,8,16,32
python -m shell_cpu.reference --suite sphere --levels 2,4,8,16
python -m shell_cpu.study --help
python -m shell_cpu.performance
python shell_cpu/results/performance_confirm.py
python -m shell_cpu.precision_study
PYTHONPATH=. python shell_cpu/review_results/check_translation_gauge.py
python -m shell_cpu.contact_reference --help
python -m shell_cpu.solid_comparison --help</pre>
<p>最细鸡蛋 n=24 的一次分解约 203 秒，属于线下成本；无需为 CPU 性能微调反复运行此级别。大 NPZ 场通过 study.py 重建，小 JSON 与图随研究目录保存。可先使用 n=3/5/8 完成实现一致性测试。</p>
<p>本轮基线分支代码为 <code>58a61834cb8e2c536a52887f032dce9898adaafd</code>。所有本报告数字均来自本轮 CPU 数据；没有将旧 native 体 DPR 的 GPU 数据套用到壳后端。</p>
</section><footer>CPU shell-DPR study · generated from recorded results · 2026-10-24</footer>'''
    source_url='https://github.com/Roushelfy/Genesis/tree/'+revision+'/research/rigid_stress/shell_cpu'
    body=body.replace('<footer>', '<section><p>研究源码与结果：<a href="'+html.escape(source_url)+'">Genesis / shell_cpu · '+html.escape(revision)+'</a></p></section><footer>')
    css='''body{font:16px/1.75 system-ui,-apple-system,"Noto Sans CJK SC",sans-serif;color:#1d2939;background:#f1f5f9;margin:0}header,section,nav,footer{max-width:1100px;margin:24px auto;padding:28px 36px;background:white;border-radius:12px}header{background:#102a43;color:white}h1{font-size:34px;line-height:1.25}h2{font-size:25px;color:#16324f}header p{color:#cbd5e1}.eyebrow{font-size:12px;letter-spacing:1.6px}.lead{border-left:6px solid #14856c}nav{display:flex;gap:20px;flex-wrap:wrap;padding-top:16px;padding-bottom:16px}a{color:#126c9c}pre{padding:18px;background:#f4f7fb;overflow:auto;font:13px/1.6 ui-monospace,monospace}code{font-size:14px;background:#f1f5f9;padding:2px 5px}.table{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px}th{background:#eaf1f7;text-align:left}td,th{padding:10px 12px;border-bottom:1px solid #dce5ed;vertical-align:top}.note{padding:16px 20px;background:#fff7e5;border-left:4px solid #d9a12a;margin:24px 0}img{width:100%;height:auto}figure{margin:28px 0}figcaption{font-size:13px;color:#536174}footer{font-size:12px;color:#67768a}li{padding:5px 0}details pre{max-height:500px}@media(max-width:700px){header,section,nav,footer{margin:12px;padding:20px}h1{font-size:27px}body{font-size:15px}}@media print{body{background:white}section,header{break-inside:auto;margin:0}figure{break-inside:avoid}nav{display:none}}'''
    document='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>薄壳自动接触半径＋DPR：CPU 验证</title><style>'+css+'</style><body>'+body+'</body></html>'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(document)
    return destination


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('deliverables/shell_dpr_cpu_report.html'))
    p.add_argument('--source-revision',default='rigid-stress-recovery')
    args=p.parse_args()
    print(build(args.output,args.source_revision))
