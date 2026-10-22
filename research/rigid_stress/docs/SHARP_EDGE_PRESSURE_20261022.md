# Slender sharp-edge pressure with transverse shear

An unfitted local pressure estimate now covers the two previously rejected
sharp-box contacts and their reversed transverse loads in the declared
half-space reference problem. Grid72 cell-pressure errors are3.40--5.75%,
positive-depth sampled tensor errors1.32--1.61% and sampled von-Mises peak
errors1.85--2.13%. Native Quadrants scale and pressure evaluation pass CPU/CUDA
checks in32/64-bit precision. This supplies the sharp-edge pressure component;
actual contact binding, complete curved-shell recovery and accepted full-grasp
max/full throughput remain open. The [active goal](../GOAL.md) is unchanged.

## Independent normal/shear reference

The existing normal-only rectangle compliance is augmented by the normal
displacement produced by tangential surface force. For reduced modulus
`E*=E/(1-nu^2)`, inward-positive displacement has kernels

\[
G_{zz}=\frac{1}{\pi E^* r},\qquad
G_{zi}=\frac{1-2\nu}{2(1-\nu)\pi E^*}\frac{r_i}{r^2}.
\]

These coefficients follow the isotropic entries of Eq.86 and Table4 in
[Dean (2023)](https://doi.org/10.1680/jgere.22.00063). Exact rectangle integration
of the second kernel gives a skew matrix on a uniform grid. Prescribing
`q_i=(Ft_i/Fn)*p` therefore produces a nonsymmetric normal-contact equation.
The offline active-set solver uses a general linear solve for this equation
and checks load balance, nonnegative force/gap and complete complementarity.
Its original positive-definite normal path remains the default and reproduces
the archived48/72-grid normal forces exactly.

Independent cell quadrature differs by2.43e-15 relative. A manufactured
nonnegative force/dual pair is recovered within1.95e-16 absolute. Zero self
integral, dimensional scaling, the incompressible coupling limit, skew
reciprocity and positive symmetric part pass. These checks establish the
discrete reference calculation for prescribed proportional traction; they
do not determine a stick/slip history law.

On the captured actual P2 gaps, the observed transverse/normal force ratios
are+0.646 and-0.796; along-edge ratios are+0.00804 and-0.00775. At grid72,
including shear-induced normal displacement changes pressure by18.80/27.45%
and sampled stress by0.850/1.109%. Reversed loads give the same qualitative
conclusion. Full-resultant moment discrepancies remain1.449/7.054`Fn*a` for
the observed directions. Normal/shear displacement coupling does not explain
that captured-wrench discrepancy.

## Local estimate without pressure fitting

Let`x>=0` run inward from the sharp edge and`y` run along it. The material gap
near its edge minimum is approximated by`s*x+k*y^2/2`, with`s,k>0`. Set

\[
\beta=\frac{1-2\nu}{2(1-\nu)},\qquad
\lambda=\frac12+\frac{\arctan(\beta F_{t,x}/F_n)}{\pi}.
\]

The strip singular-integral equation admits

\[
p(x,y)=C\left[\frac{w(1-y^2/b^2)-x}{x}\right]^\lambda,
\quad C=\tfrac12E^*s\sin(\pi\lambda),
\quad 0<x<w(1-y^2/b^2).
\]

This expression is derived from the stated surface kernels. Independent
principal-value quadrature verifies its strip equation within1.13e-12 for
both shear signs and zero shear. Integrating the profile gives
`Fn=2*pi*E* s lambda w b/3`. Center/end surface compatibility in the slender
limit gives the larger root of

\[
b^3=\frac{3F_n}{\pi E^* k}
\left[\log\frac{2b}{w}-\psi(1-\lambda)+\psi(2)
-\frac32-\frac{\pi\beta}{2}\frac{F_{t,x}}{F_n}\right].
\]

Here`psi` is the digamma function. The independent CPU implementation uses
SciPy digamma and a bracketed scalar root. The native implementation uses a
shifted asymptotic digamma evaluation and12 safeguarded scalar Newton steps,
with a precision-scaled residual check. It admits the declared slender regime
`w/b<0.05`. It requires no fitted support width, reference-pressure data,
matrix assembly or PCG solve at runtime.

Independent quadrature of the trial's center/end displacement difference
measures at most0.100% discrepancy from the logarithmic approximation on these four cases. This
checks the scale derivation separately from comparison with the independent
contact reference. Earlier trials that retained energy-fitted normal widths
are also preserved; the final unfitted law replaces that fitting requirement.

| Captured contact / shear | Exponent | Width w (micrometres) | Half-length b (micrometres) | Pressure L2 error | Sample tensor error | Sample VM peak error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / observed | 0.55810 | 1.1228 | 99.854 | 3.402% | 1.320% | 1.850% |
| 1 / reversed | 0.44190 | 1.4390 | 98.398 | 5.648% | 1.458% | 1.923% |
| 2 / observed | 0.42884 | 2.1401 | 92.970 | 5.752% | 1.608% | 2.129% |
| 2 / reversed | 0.57116 | 1.5757 | 94.805 | 3.783% | 1.394% | 1.976% |

The unchanged targets are15% cell-pressure L2,10% sampled tensor and15%
sampled peak. Coupled-reference48-to72 changes are0.136--0.395% for tensors
and0.074--0.199% for sampled peaks. Samples lie at depths0.25--2 times the
original Hertz major radius. The reference retains curved gaps with flat
half-space compliance, constant proportional shear and a finite half-plane.
Broader aspect ratios, significant along-edge shear, changing friction history,
other edges/corners and complete hollow-shell errors need further evidence.

## Native checks and measured component cost

Engine[`88d9e891a`](https://github.com/Roushelfy/Genesis/commit/88d9e891aa321010cbd0dd50777e978b7d84fcc0)
adds`rigid/stress/edge.py`. Each CPU/CUDA32/64-bit control covers270 parameter
combinations plus the four captured physical inputs. Maximum scale relative
differences are2.34e-13 in64-bit and6.74e-7 in32-bit. Tests also cover positive,
zero and unbounded pressure locations; unloaded/negative/NaN inputs; unsuitable
edge regimes; unchanged neighboring regions; and restoration after changes.

| Precision | 64 regions | 2,048 regions | 65,536 regions |
| --- | ---: | ---: | ---: |
| 64-bit | 0.0298ms | 0.0320ms | 0.0789ms |
| 32-bit | 0.0200ms | 0.0202ms | 0.0197ms |

These are warmed synchronized inclusive dispatch medians for resident **scale
updates** on RTX PRO6000 Blackwell Server GPUs. The precision runs use separate
GPU allocations, so their difference is not an isolated precision speedup.
Imports, initialization, source geometry, pressure/stress recovery, transfers,
friction history and scene simulation are outside this timing. No accepted
max/full scene-throughput claim follows from these component measurements.

## Remaining physical and integration limits

The edge pressure behaves as`x^(-lambda)`. Its pointwise loaded-edge limit is
unbounded, and native pressure evaluation explicitly returns infinity there.
Refined cell-average maxima continue to rise, as expected for this singular
model. A finite cell average or positive-depth sample peak is not a converged
surface/global stress peak; retain sampling scales with every such value.

The pressure law has not been connected to actual scene source binding or
complete curved-shell particular/correction fields. The earlier Franka
rejections and moment mismatch remain evidence. Before changing their handling,
quantify the source correction and its complete stress response under the
existing engineering-error protocol. Preserve strict source diagnostics and
the original refused frames. Native max/full integration and matched complete
pure-rigid/max/full throughput remain required.

The [evidence packet](../evidence/20261019-qualified-ranking/hard-edge-law-v1/README.md)
retains both coupled-reference stages, the unfitted-law comparison, native
CPU/CUDA controls and timings, exact sources and externally hashed arrays.
