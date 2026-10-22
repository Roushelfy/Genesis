# Native hollow-egg admission and geometry gap

Native Hertz recovery now includes the omitted cavity boundary and passes
four facet-centered normal shell controls against independent complete-boundary
fields. The original selected-surface passes and their omission remain below
as evidence. The actual Franka/hollow-egg rollout still rejects its first
loaded step because its smooth contact normal differs from the material facet
normal. The full
[hard-contact goal](AUTOMATIC_CONTACT_AND_THROUGHPUT_GOAL_20261020.md) remains active.

## Actual Franka rollout

Slurm739120 uses uncompressed source795805290, four independently initialized
environments, level1 geometry, float64 and the existing moving controller.
The controller's manual pressure radius input is disabled for the Hertz route.
Steps0–9 are unloaded. The first loaded frame, step10, fails in all four
environments. Input and friction processing pass, but field status2 rejects
the source geometry before elastic recovery. No loaded update is admitted.

The retained frame and read-only inspection locate the cause. Each resultant
acts at a coarse mesh vertex. Its smooth estimated inward normal differs from
the witnessed triangle's inward normal by22.63569 degrees. The frame is
orthonormal to9.23e-16, the center lies on the material boundary to6.82e-19m,
and the entire material remains in its inward half-space. The existing
`1e-10` normal-alignment gate therefore rejects a real geometry mismatch.
Changing equilibrium or friction solver tolerances cannot fix this mismatch.
The original rejection, source identities and complete frame arrays remain.

The captured-frame gap study739190 finds that64 of89 native tangent-frame
probes lie outside the faceted material. At the Hertz footprint, the maximum
facet depth is21.35 micrometres, while the quadratic curvature gap reaches only
0.04598 micrometres and predicted Hertz indentation is0.09102 micrometres.
Merely relaxing normal alignment would therefore leave invalid material probes.

A small independent unilateral-contact control keeps half-space compliance
fixed and replaces the quadratic rigid-obstacle gap with the actual projected
facet gap. Its32/48 grids change positive-depth sampled tensors by0.335%,
the sampled von-Mises peak by0.0189% and force second moments by1.37%.
The Hertz footprint area is234–247 times the faceted-gap support area; the
sampled Hertz tensor differs by98.34% relative to that control. This demonstrates
gap-model sensitivity, not the stress error of Hertz on the physical hollow
egg: the control uses a rigid obstacle and half-space compliance, excludes
surface/global peaks and does not solve the faceted shell's elasticity.
These results favor resolving the smooth-rest-surface interpretation before
changing production geometry gates.

## Selected-surface component control

The component control places four separate normal loads at actual outer-facet
centers on the same level1 volume, with forces0.20/0.12/0.18/0.10N and declared
relative principal curvatures50/80 per metre. Its finite-radius law, all element
corners and89 moving probes run in native Quadrants. This isolates recovery
from the Franka source association and smooth-contact approximation.

| Boundary rule | Maximum wrench error | Native accepted environments | Independent complete field check |
| --- | ---: | ---: | --- |
| Order8 | 0.0284832 | 0/4 | Refused before a usable field |
| Order16 | 0.0000405193 | 4/4 | Pass on CPU and CUDA |

The unchanged wrench gate is`1e-4`. With order16, independent P2 derivatives
and consistent-mass inertial relief, combined with the same prescribed Hertz
particulars, differ by at most`3.59e-11` at element corners and`2.22e-9` over
the complete probe bank. Both remain below the original`1e-7` field gate.
This is a same-load numerical control. It supplies no independent pressure,
curved-contact, frictional shell error or matched scene-throughput claim.

The subsequent complete-boundary audit739157 finds80 selected outer faces
versus160 actual material-boundary faces. Independent integration over the
selected80 reproduces the native order16 load within`1.56e-8` relatively.
Including all160 faces changes the sampled corner tensor bank by98.95–101.12%
relative to the complete-boundary result. The force/moment checks cannot
detect this omission because the cavity's resultant is zero while its local
traction correction matters. These are differences within the same point-
traction regular-response model, not newly qualified physical stress errors.

The all-boundary reference concentrates integration around each face's closest
point to the source. Order24/32 changes the sampled corner fields by at most
`2.56e-9`. The first lower-order trial evaluated moment about the source center.
Applying the production gate about the authored model origin rejects that
trial. Both moment definitions and original outcomes remain recorded.

## Complete-boundary native repair

Source[`2e39f5f75`](https://github.com/Roushelfy/Genesis/commit/2e39f5f752136de5d7616b3436063f1bf480814b)
now constructs every material-boundary face for Hertz recovery, preserving
supplied face indices and appending omitted faces. The normal-control asset
itself is unchanged. Cavity faces participate in clearance checks, traction
correction and the shared surface operator.

Near faces use three triangles around the closest point, a rational radial
map resolving the source-to-face distance, and two intervals in each integral
coordinate. Remote faces use the supplied uniform rule. With order16, all four
shell cases pass on CPU and CUDA. Maximum authored-frame wrench error is
`4.91e-7`, within the unchanged`1e-4` gate. The original unsplit native attempt
fails three of the four cases and remains retained.

The independent higher-order all-boundary integral and P2 readout give maximum
relative differences`9.71e-9` in nodal load, `2.58e-9` in the complete corner
tensor bank and`2.22e-9` in the probe bank. The existing`1e-7` complete-field
gate passes. These four cases use43,520–54,784 point-tensor evaluations each.
That count describes quadrature work, not measured module or scene throughput.
Order16 is the tested setting for these facet-centered normal shell controls.

Slurm739154/739157/739163 preserve the independent integration studies, including
the unconverged uniform-rule attempt and moment-definition correction.
Native739161/739164 and inspection739169 retain the original rejection and the
passing complete-boundary fields. The earlier native attempt also passes the
existing rotated normal-component CPU/GPU and actual planar-friction CUDA
controls. The final refinement's new all-boundary normal path is tested on both
backends. See the [complete-boundary packet](../evidence/20261019-qualified-ranking/hard-native-hertz-complete-boundary-v1/README.md).

This repairs boundary coverage and numerical integration under the declared
normal traction model. Moving friction on the shell, physical pressure errors,
curved-source compatibility and accepted Franka throughput remain required.

## Parallel boundary integration and measured module cost

The first complete-boundary timing739183 attributes most cost to boundary
integration. Measured source[`61c121720`](https://github.com/Roushelfy/Genesis/commit/61c1217206d57269caabf45480469b2bc65233c4)
assigns one angular Gauss knot to each lane;
each lane retains all radial rules and atomically accumulates its nodal loads.
Quadrature points, weights, coverage, pressure law and acceptance gates remain
the same. CPU/GPU shell and rotated normal controls pass. Independent complete
corner/probe fields remain within2.58e-9/2.22e-9 relatively, and the actual
planar-friction control preserves history exactly and complete fields within
1.07e-18 relative to the retained baseline.
Its formatted descendant[`32c1f3ecf`](https://github.com/Roushelfy/Genesis/commit/32c1f3ecff20813a7567e03f4ed725686b57f21d)
has an identical Python syntax tree; both exact sources and the formatting
patch remain retained.

The same-allocation739197 comparison on an RTX PRO6000 Blackwell uses the unchanged level1 shell,160
boundary faces, order16, one supplied region per environment, changing
facet-interior centers, rotating curvature, changing loads and release.
Primary timing includes resident input generation, Hertz lookup, boundary
traction, regular recovery, moving probes and max/full reduction. Research
exports and inspection are outside timing. It includes no rigid scene,
collision association or shell friction.

| Environments | Baseline max / full (ms) | Parallel max / full (ms) |
| ---: | ---: | ---: |
| 1 | 23.139 / 23.097 | 2.845 / 2.713 |
| 64 | 27.140 / 27.168 | 3.964 / 4.018 |
| 256 | 50.287 / 50.203 | 8.370 / 8.413 |
| 1024 | 62.036 / 62.435 | 25.787 / 25.852 |

All24 audited frames per environment pass, including unloaded frames. Loaded
counts per mode are22,1418,5672 and22686 respectively. Input/admission histories
match exactly; sampled peak differences are below2.55e-16 relative. Separate
fixed-input diagnostics put the parallel input/boundary phase at1.88ms forB1
and18.72ms forB1024 full output. These diagnostics are not additive inclusive
phase totals. The measured module improvement is2.41–8.51 times for this
declared normal workload; it is not a Franka throughput result.

The separate739203 paired sphere/flat-link scene includes actual collision
association, normal/friction input, physical history, two rigid substeps,
elastic recovery and max/full output. Both variants use separate processes
and private compiler caches on the same allocation; the baseline runs first.
All24 audited loaded steps per environment pass, trajectories agree within
1e-12m, and the final739207 reader verifies every saved peak and status.

| Environments | Baseline rigid / max / full (ms) | Parallel rigid / max / full (ms) |
| ---: | ---: | ---: |
| 1 | 1.798 / 15.980 / 16.063 | 1.787 / 14.245 / 14.163 |
| 64 | 1.936 / 15.461 / 15.534 | 1.933 / 14.051 / 14.120 |
| 256 | 1.930 / 24.840 / 25.049 | 1.929 / 22.034 / 22.268 |
| 1024 | 2.113 / 49.003 / 49.032 | 2.058 / 48.308 / 48.432 |

Full-scene wall time falls9.10–11.83% forB1/B64/B256, with separated three-repeat
ranges. The B1024 full-scene change is only1.22%, while the matched pure-rigid
baseline also changes2.64%; a general large-batch scene speedup is not established.
The optimized B1024 full scene measures21,143 admitted environment scene-steps/s
on this supported planar-friction workload. It does not admit the Franka egg.
Retain the parallel integration for the measured boundary-limited path; use
order16 for the tested facet-centered shell controls. Neither performance
result changes the curved-source or physical-pressure limits.
The [parallel-boundary packet](../evidence/20261019-qualified-ranking/hard-native-parallel-boundary-v1/README.md)
contains the actual sources, numerical controls, raw repeated costs and gap
study. Use its captured tasks with the same Slurm runner and explicit fresh
data outputs; full arrays remain hashed on data.

The original order16 reader round-tripped surface probes through world
coordinates. At contact edges, coordinate changes below`3.30e-18m` changed
its reference field by up to`1.36e-7` relatively. The corrected reader uses
the probes' declared exact local coordinates, matching the existing production
definition, and separately records that round-trip sensitivity. The original
field assertion failures remain preserved.

## Repair and reproduction

Source[`d26b5b3d4`](https://github.com/Roushelfy/Genesis/commit/d26b5b3d46efe614ed715fef293f8cc49bcdf0c7)
repairs a standalone normal-component initialization bug introduced by the
optional shear container: a zero-length scalar Quadrants array was invalid.
Unused scalar-shaped placeholders follow the existing inactive-buffer
convention. Actual friction routes already supply their own history buffers.
Existing rotated normal-component/max/full controls pass on CPU and CUDA.
The Hertz option documentation now describes the implemented vector history.

The first shell worker also retained an array-axis error before reaching the
engine. Campaigns`hard-native-hertz-shell-admission-v1` through`v4` preserve
that failure, the initialization defect, the order8 rejection and both versions
of the independent reader. Slurm739126/739131/739135/739143 record their actual
terminal states. Inspection of the Franka frame completes within739131 even
though that group's later shell tasks fail.

Use the captured tasks in the
[evidence packet](../evidence/20261019-qualified-ranking/hard-native-hertz-shell-admission-v1/README.md)
with `shell_comparison/submit.py`, the stated immutable source, explicit fresh
data outputs and node-local caches. Scientific execution uses Slurm`rtx-mid`,
image`1_26` and the existing data-side physical environment. Full arrays stay
on data. `check_live_hertz_franka`, `check_native_hertz_shell` and
`inspect_hertz_franka_admission` are the three new research entrypoints.

Remaining work is a useful curved-source approximation with matched physical
error, moving shell friction, and accepted moving Franka max/full throughput.
The coarse-facet versus smooth-rest-surface mismatch remains unresolved.
