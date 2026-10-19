# Normal physical-pad reference and conditional error comparison

The declared normal physical-pad case now passes its predeclared reference
change targets. This establishes a useful, metric-specific comparison for
three normal loads. It does not complete the
[active goal](QUALIFIED_STRESS_RANKING_GOAL.md): independent sustained-friction
spatial qualification, complete matched sparse/batched costs, memory and
field/traction maps are still required. The later
[common-peak and batch checkpoint](COMMON_PEAK_BATCH_RESULTS_20261019.md)
adds common-peak convergence, time/contact controls, actual B4 conformance,
admitted B1 every-step full/max timing and the force/closure response figure.
The subsequent [spatial/cost update](NORMAL_SPATIAL_COST_20261019.md) completes
normal field/traction maps and B1 sparse timing; fine friction and B4/memory
remain outstanding.

The physical case remains the full free hollow egg,500um wall,E=10GPa,nu=0.3,
rho=2000kg/m3, with the explicitly declared1e9N/m3 compliant pad. Native DPR
and SCR use all actual contact sources with the fixed2mm reconstruction radius.
The compared Shell is the repaired250um polar-resolution contact/elastic
variant, with its unchanged Shell constitutive model. The original hard-plate
case and original Shell failures remain separate; this report does not replace
or qualify them. See [the pad declaration](COMPLIANT_PAD_DECLARATION.md).

## Reference admission and uncertainty

Job728139 completes the125um polar/three-wall P2 reference with contact orders6
and10 at0.1,0.3 and1N. All six equilibrium cases pass the actual pad, complete
equation, free-body wrench, nodal transfer and omitted-boundary checks. The
largest equation residual is1.005e-10 and normal-law residual3.217e-12.
The verified long-index CVXOPT factorization takes1448.98s; reference solution
time is1709.42s. The preceding32-bit CHOLMOD failure remains retained.

Independent surface and wall changes decrease along the retained refinement
sequences. At the final joint resolution, the125um two-to-three-wall tangent
changes are1.1315/1.0872/1.0135%; the250-to125um three-wall changes are
1.0457/0.9755/0.8907%. The250um/two-wall to125um/three-wall diagonal changes
are1.5663/1.4843/1.3706% using25 angular samples per fine face. All complete
tangent, membrane, bending, inner and outer field changes pass2%; all raw-peak
changes pass5%. Contact-order6-to10 and angular integration controls are
materially smaller. Physical-volume weights include all tetrahedron partitions.

`pad-normal-qualification-v2` (728217) checks those receipts, all native
source/history acceptance, every Shell equilibrium window and all1000 Shell
history steps. No Shell solver or pad-law failure occurs; the largest full
history equation residual is1.826e-6, below its existing1e-3 equation target.
The reference's stricter force-equation criterion is not silently imposed on
an older Shell run. Every check in this qualification artifact passes.

For each metric/load, delta is the largest observed final wall, surface or
joint change plus contact and metric quadrature changes. The maximum field
delta is1.5671%. Conditional ranges are
`[max(e-delta,0)/(1+delta), (e+delta)/(1-delta)]`. This triangle-inequality
propagation is valid **if the observed envelope bounds the reference error**.
It is not a certified continuum bound or a statistical confidence interval.
Radius sensitivity, constitutive choices and contact-model choices remain
separate from this numerical envelope.

## Complete fields and retained raw peaks

All entries are percentage relative errors against125um/three-wall,
contact-order10 P2. Raw peaks retain each method's complete native sampling;
they are not silently relabeled as common-location tangent-plane peaks.

| Method | Load N | Tangent | Membrane | Bending | Outer | Inner | Raw Mises | Raw tension |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DPR |0.1|7.934|4.682|10.303|7.558|9.877|2.871|5.617|
| DPR |0.3|7.947|4.528|10.899|7.919|10.518|0.302|0.784|
| DPR |1.0|7.429|4.314|10.517|7.450|9.894|0.956|1.700|
| SCR |0.1|7.977|4.810|10.280|7.507|9.925|1.248|5.080|
| SCR |0.3|7.924|4.670|10.719|7.679|10.502|1.277|0.158|
| SCR |1.0|7.444|4.577|10.289|7.188|9.879|2.420|2.365|
| Shell |0.1|25.980|5.748|42.039|31.985|35.209|0.241|14.846|
| Shell |0.3|25.200|5.640|41.911|31.325|34.647|0.129|14.962|
| Shell |1.0|23.985|5.455|41.725|30.282|33.723|0.213|15.661|

The conditional ranges separate native and repaired-Shell tangent, bending,
inner and outer errors at all three loads. Membrane comparisons overlap;
DPR/SCR field differences also remain unresolved. Raw-peak ordering depends
on metric and load. A small Shell Mises-peak error does not imply an accurate
complete field. The predeclared1.5/2.5mm radius controls change tangent fields
by2.89--4.21% and bending by4.50--6.44%; the primary radius remains2mm.

The editable normal-field error figure and its caption/QA are in the
[checkpoint packet](../evidence/20261019-qualified-ranking/normal-pad-reference-v1/README.md).
It shows individual load cases and conditional ranges, without statistical
aggregation. The retained CSV includes all seven metrics above.

`--common-peaks` additionally evaluates tangent-plane Mises and nonnegative
tensile peaks at identical angular/material columns, including both one-sided
tetrahedron traces and both wall boundaries. It separately integrates true3D
normal/transverse shear components and assigns Shell no unsupported transverse
error rank. Its analytic invariant, rotation and component tests pass. The
9/25-point sampling and complete common-peak spatial-uncertainty campaign now
pass, as documented in the later checkpoint. Raw and common Mises rankings
differ; retain both definitions and their conditional uncertainty.

## Objective transient reference and measured cost preflight

The full-mass StVK reference (728171) completes400 steps, both equilibrium
windows and the current-position moment audit. The sustained-shear endpoint
moment residual is7.258e-10, resolving the retained linear model's5.208e-6
failure. At0.1N normal load the objective extension changes tangent/membrane/
bending by0.0185/0.0115/0.0256% and raw peaks by at most0.0418%; this control
does not establish the same bound at other loads. The unchanged dense-solver
half-timestep trajectory remains a separate qualification run.

The modal compliance split retains the complete operator and converges its
unrelaxed force, point-velocity and momentum defects. Its first undamped
manufactured control fails; the retained Aitken-damped version passes all24
controls and then the full400-step StVK trajectory (728205), including
sustained shear and current moment3.325e-10. Its dynamics take1154.54s versus
1167.30s for the dense baseline: this is not a large measured speedup. It
avoids the dense, potentially singular high-order contact compliance and
therefore enables the next integration/refinement controls. The warm
matrix-free StVK trajectory also passes400 steps (728194), taking1647.58s.
The normal/shear full-field solver controls (728224) agree with the dense
baseline within7.759e-10 in tangent/membrane/bending and4.384e-10 in raw peaks.
The earlier cold linear matrix-free run728169 times out after344 steps;
that is an incomplete execution, not an admitted comparison or a solver-cap
success. All failure records remain retained.

The matched B1 every-step preflight (728186) completes400 steps per method
and output mode, with the first200 excluded as warmup. All histories and
stress observations pass. Instrumented segmented latencies are:

| Method | Full ms/step | Max ms/step |
|---|---:|---:|
| DPR |67.17|68.01|
| SCR |1365.57|1418.03|
| Shell |434.58|436.56|

These segments include mechanics and requested recovery/host output, but
exclude extra mechanical auditing and frame/journal I/O. They are not primary
uninstrumented throughput, repeated timing or batching results. Full fields
agree with the original first equilibria within1.122e-9 in all tensor values.
The output-mode audit728218 nevertheless fails its stricter1e-8 cross-run
peak comparison for Shell (1.520e-7), mostly at first contact; native arms
pass. A same-frozen-state paired observation control isolates observer mode
from independently accumulated floating sums. The failed audit is retained
without changing its threshold. Sparse20-step preflights remain separately
labeled and cannot be substituted for every-step cost.

All runtime arrays, logs, captures and environments remain under the established
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/` root.
Only small source, documentation and selected publication evidence enter Git.
