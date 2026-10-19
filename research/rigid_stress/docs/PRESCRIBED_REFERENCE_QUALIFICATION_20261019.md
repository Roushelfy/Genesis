# Qualified prescribed-load reference and signed error decomposition

The independent prescribed-C2 reference now passes the unchanged reference-change
targets. Job728896 passes all25 checks and rehashes31 inputs after all four
field groups of728860 finish. The [packet](../evidence/20261019-qualified-ranking/prescribed-reference-controls-v1/README.md)
retains the terminal evidence and exact sources. This is the0.1N-per-pole,
fixed2mm-radius pressure `max(1-rho^2/a^2,0)^3` on the free full hollow egg,
with E10GPa, nu0.3 and500um wall. It is a separately prescribed traction case;
it does not solve or qualify the physical-pad contact history.

The final125um/three-wall P2 reference has independently admitted q8/q10 elastic
equations. The verified long-index CVXOPT factor adapter agrees with the
retained same-mesh reference to at most3.130e-13 across field/raw/common peaks.
Factorization takes1443.67s; these factor times are separate from postprocessing
and from compared-method throughput.

## Spatial and integration evidence

All table entries are percentage changes with q25 common surface integration,
physical volume weighting and complete material-column thickness integration.
The five final field metrics remain below2%; raw and common peaks remain below5%.
No pressure profile, radius, polar domain or target was adjusted to pass.

| Control | Tangent % | Membrane % | Bending % | Outer % | Inner % |
| --- | ---: | ---: | ---: | ---: | ---: |
| 125um wall2→3 | 1.361407 | 0.327054 | 0.3531605 | 0.2061074 | 0.4224326 |
| 250→125um, three walls | 1.248164 | 1.033006 | 0.4943475 | 0.5115825 | 0.734714 |
| 250um/two→125um/three | 1.903291 | 1.408207 | 0.8518662 | 0.7369621 | 1.016687 |
| Final load q8→10 | 2.512164e-08 | 9.383068e-09 | 3.268347e-08 | 3.091253e-08 | 2.996258e-08 |

| Control | Raw Mises % | Raw tension % | Common Mises % | Common tension % |
| --- | ---: | ---: | ---: | ---: |
| 125um wall2→3 | 0.4560095 | 0.2781456 | 0.1474708 | 0.2733945 |
| 250→125um, three walls | 0.155284 | 0.1275349 | 0.1141651 | 0.1313412 |
| 250um/two→125um/three | 0.4635106 | 0.1259349 | 0.1619861 | 0.126108 |
| Final load q8→10 | 2.387711e-08 | 3.666819e-08 | 2.507028e-08 | 3.686222e-08 |

The retained two-wall surface sequence decreases from3.14818% to2.08330%
tangent change. The latter still fails2%; it is not rounded into acceptance.
At three wall layers the final surface change is1.24816%, with a separate
1.36141% wall change and1.90329% joint change. All five field metrics decrease
along the retained two-wall surface sequence and the250-to125um wall-refinement
controls. This resolves the previous prescribed-case spatial gate at the declared
final configuration, rather than discarding the failing two-wall control.

Across the seven q9/q25 sampling controls, the largest absolute change in a
relative field metric is1.56259e-4 (0.015626 percentage points); the largest
change in an actual/reference common peak is0.118647%. Both pass the existing
0.005 field-metric and0.01 peak sampling bands. Load q8/q10 changes are below
3.69e-10 for every field/raw/common peak. These are separate measured controls,
not a force-equation tolerance relabeled as stress accuracy.

Observed reference-change envelopes include final wall/surface/joint changes,
load quadrature and common sampling. The tangent envelope is1.90533%, membrane
1.40825% and bending0.862419%. They are conditional resolution estimates, not
certified continuum bounds or statistical confidence intervals. Method error
ranges must retain that qualification and their own sampling effects.

## Direct tensor decomposition

The next native comparison uses three complete fields on identical material
columns: (0) DPR or SCR, (1) independent prescribed-C2 P2 on the same coarse
mesh, and (2) the qualified fine prescribed-C2 reference. Native inputs are the
original two-pole-source reconstruction with fixed2mm radius; they are distinct
from the later full-source physical-pad runs. Same-mesh native core agreement
under the actual independent nodal load is already verified to3.13e-12.

`contact_fem_exact_columns --decompose-three-fields --physical-volume` computes
`A=S0-S1`, `B=S1-S2` and `T=S0-S2` at every quadrature point, then directly
integrates `||A||^2`, `||B||^2`, `<A,B>` and `||T||^2` on one union of all three
tetrahedron partitions. The retained identity is
`||T||^2 = ||A||^2 + ||B||^2 + 2<A,B>`.
Membrane, bending and both boundary tensors use the same signed construction.
A negative cross term exposes cancellation; RMS contributions are not added
or subtracted as independent causal error budgets. All relative contributions
use the same fine-reference norm, with explicit null relative values at an
exactly zero denominator and no arbitrary floor.

Seven manufactured controls (728884) pass independent analytic volume,
membrane, bending and boundary integrals, differing material partitions,
reinforcement/cancellation,24-decade stress scaling and exact-zero handling.
The complete native q9/q25 decomposition is running as728893 at this packet's
cutoff; its results and method ranks are not inferred from these analytic tests.
The comparator now also records distinct raw Mises and raw tensile positions,
while preserving the historical `raw_peak_position_m` Mises alias. This metadata
addition does not change either peak magnitude or its sampling.

## Matching the Shell static control and repairing allocation

The historical prescribed-load Shell replay used a coarser configuration.
The new replay uses the same250um polar mesh,500um material,12 coarse patches,
PCG1e-4/velocity1e-10 and8000-iteration cap as the primary normal Shell setting,
with the pinned repaired Shell source. It takes the independently integrated
250um C2 point tractions and transfers their force/couple to the Shell surface.
Velocity clearing remains explicit static relaxation, not a continuous contact
history or a replacement for the physical-pad dynamics.

The first attempt728882 exhausts its32GB host allocation during mapping,
before solver initialization. It has57,158 source points. Each retained
`bary[index]` view kept the entire candidate-face array alive, accumulating
storage proportional to source count times face count. The correction copies
only the three selected barycentric weights; no geometry, force, tolerance or
numerical operator changes. The failed run and exact old source are retained.

The measured before/after control728885 uses5000 faces and200 source points.
Mapping arrays and assembled node forces are bitwise identical. Traced peak
allocation falls from25,308,379 to1,400,691 bytes; arbitrary rigid virtual work
is preserved to1.066e-14. This is a scoped allocation/control result, not a
whole-process or stress-accuracy claim. The full250um replay retries as728888
in a fresh directory with the same32GB allocation and physical parameters.
The reference/control packet retains only its submitted inputs. The subsequent
[terminal companion](../evidence/20261019-qualified-ranking/prescribed-shell250-terminal-v1/README.md)
now verifies completion in4m52s, all3 iterations without solver flags and final
static force residual2.10101472e-8. Maximum step RSS is31,636,028KiB under the
same32GB request. Its source/frame/rest identities are retained. This full-run
completion supplements the scoped allocation control without turning its traced
memory ratio into a whole-process memory reduction claim.

The matching Shell three-field comparison is now728918: static Shell250um,
independent prescribed-C2 solid250um and qualified125um/three-wall solid.
The first difference combines thin-shell constitutive/kinematic and load-map
effects; it is not isolated contact error. The second is the solid reference's
spatial change. Native and Shell decompositions retain their distinct physical
interpretations and require completed q9/q25 comparisons before final ranks.

## Reproduction and remaining work

`qualify_prescribed_load --data-root RUN_ROOT --output NEW_DATA_PATH` rechecks
the fixed targets, exact inputs and all25 gates after terminal field comparisons.
Use the captured task files and the existing Slurm/container workflow. CPU field
comparisons, signed integration controls and metadata audits use `submit
--cpu-only`; Shell static replay retains its GPU allocation. All arrays, full
fields and runtime logs remain under the established data root.

The reference/control packet freezes209 payloads totaling2,776,889 bytes; the
terminal Shell companion adds25 payloads totaling455,148 bytes. Active native/
Shell field workers contribute immutable submitted inputs only; their outputs
and all live friction writers are excluded. Same-load method comparisons, final
friction reference/ranks and complete friction figures remain required. This
reference qualification does not complete the [active goal](QUALIFIED_STRESS_RANKING_GOAL.md).
