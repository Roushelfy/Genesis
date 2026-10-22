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

## Curvature correction and curved rest-geometry control

The first loaded Franka frame also exposes a pressure-estimation error before
the separate geometry refusal. With the captured0.062275N load and fixed
half-space compliance, an independent unilateral contact solve on the analytic
smooth egg gives6.264% pressure-cell error for the original native Hertz law.
Analytic egg curvature reduces that error to1.365%. The32/48-cell reference
changes0.051% in sampled positive-depth stress and0.064% in pressure second
moment. These local controls exclude finite-shell compliance and global peaks.

Source[`e7e49b738`](https://github.com/Roushelfy/Genesis/commit/e7e49b738e425f9a25227929262122bcd1f35340)
fits an implicit local quadric to adjacent positions and normal directions
once per shared shape, entirely in Quadrants. A scaled six-coefficient fit
updates the cached tangent curvature. Rank-deficient fits retain the original
face-derivative estimate and expose a false`vertices_quadric_valid` flag.
Current-frame queries, source identity, pressure law and recovery acceptance
are unchanged. No per-step fit or contact solve is added.

The independent comparison uses the imported mesh's actual normals. At the
first contact, the quadric candidate gives1.683% pressure and0.494% sampled
tensor error, versus6.264%/2.412% for face averaging. A direct vertex-normal
derivative fit gives2.150%/0.749%; the quadric fit is retained for its broader
shape improvement. Whole-surface vertex-curvature RMS remains resolution
dependent:

| Shape and mesh | Face averaging | Quadric fit |
| --- | ---: | ---: |
| Ellipsoid,80 faces |18.309%|5.507%|
| Egg,80 faces |17.356%|9.138%|
| Egg,320 faces |6.120%|3.659%|
| Egg,1280 faces |2.159%|1.709%|

Thus coarse vertex curvature is improved but does not meet the5% RMS target
everywhere. Interpolation between vertices contributes additional error:
the existing native face-centroid control still measures5.741% on the80-face
sphere and14.193% on the80-face ellipsoid. The local first-contact result
does not qualify arbitrary moving contacts or an entire shell.

CPU/GPU fits agree with an independent least-squares implementation below
`2.26e-14` relative over six closed mesh controls. Normal reversal preserves
signed curvature within`1.08e-14`; exact planar input uses the explicit
rank-deficient fallback. The2406-query disconnected-surface control preserves
all selected faces in three rigid frames. Existing region splitting, movement,
release, invalid-input and rotated-frame controls pass on both backends.

For672 vertices, five warmed inclusive constructor measurements give median
CPU2.017→4.965ms and GPU1.436→2.102ms. The additional persistent flag storage
is672bytes; transient fit buffers total233,856bytes. These measurements include
allocation and synchronization and quantify preprocessing only. They are not
module or scene throughput measurements. The actual four-environment Franka
rollout still rejects all four loaded environments at step10 with recovery
status2 and valid contact inputs. No rejected step counts as a useful update.
The actual native snapshot comparison confirms the1.683%/0.494% local
pressure/tensor result, with15 rigid/contact/frame/status arrays exactly equal
to the baseline. Its sampled VM/tensile-peak errors are0.965%/0.0181%.

A separate quadratic rest-surface candidate preserves every original vertex
and uses shared edge nodes constrained by endpoint tangent planes. On the
80-face egg, its sampled cell Jacobians remain positive, but analytic-normal
curvature error is13.715% and the maximum cross-edge normal jump is6.244°.
Area-weighted normals give14.811% and6.236°. Its local half-space reference
differs3.257%/3.778% in sampled tensor from the analytic egg; neither curvature
nor tangent continuity is generally qualified at this resolution. At1280faces,
the analytic-normal reconstruction meets the declared geometry checks.

The curved candidate nevertheless supports a consistent independent P2
elasticity discretization. Both normal choices pass the all-boundary uniform
traction test at quadrature orders8/9: uniform stress error is below`8e-12`,
load consistency below`2.67e-13`, and mass changes below`4.3e-15`. This tests
curved Jacobians, stiffness, consistent mass and rigid modes. All89 contact
probes can be shifted onto the curved boundary and located inside curved
material; the maximum surface shift is46.64nm and inverse-map error is below
`1.42e-17m`. The corresponding unmodified tangent probes remain invalid in
the original faceted domain. These are operator/geometry controls, not contact
stress or pressure validation, and no production material geometry is changed.

The [curvature and curved-geometry packet](../evidence/20261019-qualified-ranking/hard-native-quadric-curvature-v1/README.md)
retains all actual successes and failures. Failed attempts include a misplaced
test option, an imported flat mesh that was thickened, thin-cell global inverse
mapping, a JSON scalar and an unavailable order10 tetrahedron rule. The final
boundary integration uses each face's known local chart and supported orders8/9.
Reproduce with captured tasks, immutable sources, Slurm`rtx-mid`, image`1_26`
and fresh explicit data outputs. Continue with consistent curved-source
traction and native material geometry, matched stress error and accepted
Franka max/full throughput; the complete goal remains open.

## Native curved material and source integration

Experimental engine [`aa9cc8298`](https://github.com/Roushelfy/Genesis/commit/aa9cc8298)
now accepts optional `edge_positions` in the stress mesh asset. Coordinates follow
lexicographically sorted unique tetrahedral edges. Shared quadratic geometry is
used consistently for volume stiffness, consistent mass, boundary positions,
normals and weights, and corner/query stress gradients. Positive sampled geometry
Jacobians are checked before assembly. Assets without edge coordinates retain the
affine path; curved geometry currently requires the Hertz route. Source association
and transfer are not yet adapted to it, so this is an isolated material implementation.

The 810-node/480-cell shell passes independent same-load checks with both analytic
and estimated geometry on CPU and CUDA. Maximum relative differences are:

| Quantity | Difference |
| --- | ---: |
| Stiffness matrix | `5.32e-15` |
| Consistent mass matrix | `4.37e-15` |
| Uniform-traction load consistency | `8.14e-13` |
| Complete corner stress | `7.77e-11` |
| 1440 interior material-query stresses | `5.27e-11` |
| Response change from 6³ to 7³ volume quadrature | `4.22e-8` |

The controls include uniform and asymmetric loads, centrifugal inertia relief,
zero load, max/full output, cached/uncached gradients and a deliberately inverted
cell. Existing affine four-source shell checks, rotated normal components and the
live friction/history/checkpoint/partial-reset scene pass. The original decimal
corner-cancellation test passes all four CPU/CUDA and cached/uncached cases.
Five warm B3 regular-core samples are retained, but include 1440 extra material
queries and no active contact particular. They are not contact-module or Franka
throughput measurements. Initial failures from static tensor-shape inspection
and a default kernel argument remain in the v1/v2 campaigns.

A separate independent elasticity study uses the actual first-loaded Franka normal
source on the same curved shell. It compares the inexpensive point-traction
boundary correction with distributed prescribed-Hertz traction cancellation.
Both use the complete material boundary and consistent curved operators; contact
pressure is prescribed, not independently predicted by this comparison.

A rational radial map applied over the entire source face misses its broad curved
traction: the original order16 wrench error is about1.4%. Surface tangency cancels
the point singularity against a source-centered fan's radial Jacobian. Linear
radial integration at that source, with the existing focused rule on nearby remote
walls, reduces the order16 error without changing its50176 sample count:

| Geometry normals | Point16 wrench error | Corner-field difference | Probe-field difference |
| --- | ---: | ---: | ---: |
| Analytic | `6.61e-7` | `2.33e-7` | `1.23e-7` |
| Estimated | `7.06e-7` | `2.99e-7` | `1.61e-7` |

Differences are relative to the refined distributed-Hertz correction. Peak
VM/tensile differences are below`1.41e-7` in this control. The reference separates
the broad point term from the localized distributed-minus-point correction and
integrates physical pressure together with its cancelling traction. Order24→32
field/peak changes are below`2.38e-7`; final reference wrench errors are
`1.45e-5`/`2.67e-5`, below the independently retained`1e-4` gate. The estimated
order24 reference still has`1.30e-4` wrench error, which remains reported rather
than treated as an admitted result. The first two quadrature attempts are retained.

These very small decomposition differences do not measure shell discretization,
unknown-pressure error or arbitrary curved-source accuracy. They support porting
this source-centered integration into Quadrants. Native curved source provenance,
material-following probes, moving friction and accepted Franka max/full throughput
remain required. The whole goal stays active. The
[material/source evidence packet](../evidence/20261019-qualified-ranking/hard-native-curved-material-v1/README.md)
retains seven terminal campaigns, their original failures, exact sources and
complete fields at hashed data-side paths. Reuse the captured tasks with immutable
sources, Slurm`rtx-mid`, image`1_26`, fresh explicit data outputs and private caches.

## Native curved Hertz sources and moving material probes

Experimental engine [`ce57a7fb2`](https://github.com/Roushelfy/Genesis/commit/ce57a7fb2af5a235a2f77360afe0ad9490cbecb7)
connects quadratic material geometry to source traction and all89 moving probes.
Sources are checked against the actual curved position and tangent. Probes follow
the curved boundary, then invert the owning material cell through shared-face
adjacency. Complete-boundary correction uses pointwise curved normals and areas,
with linear radial integration on a source-centered fan. Original affine,
source/wrench, material-containment and field acceptance gates remain in force.

Both analytic-normal and estimated-normal assets pass independent CPU/CUDA
same-prescribed-Hertz checks. Maximum relative differences across them are:

| Quantity | Difference |
| --- | ---: |
| Consistent nodal load versus refined point reference | `6.57e-7` |
| Complete corner tensor versus distributed reference | `3.16e-7` |
| Moving-probe tensor versus distributed reference | `2.23e-5` |
| Sampled von Mises peak | `3.98e-8` |
| Sampled tensile peak | `4.36e-4` (0.0436%) |
| Complete-boundary wrench | `4.92e-9` |
| Maximum material inverse-map error | `1.39e-17m` |

Max/full agreement, released states, restoration, and isolation of a deliberately
misplaced source pass. Existing affine-shell and actual planar-friction controls
also pass. The initial two-radial-subdivision rule failed the declared nodal-load
`1e-5` gate with `1.43e-5`. Three radial subdivisions pass without changing that
gate. Both attempts and their complete fields remain retained. These comparisons
verify native same-load recovery on the selected quadratic geometry. Independent
pressure accuracy, shell discretization and continuous global-maximum accuracy
retain their separate requirements.

Three warm B3 normal-module samples give approximately11.49ms on CUDA and19.9ms
on CPU. They include scatter, regular recovery, moving probes and synchronization
for two loaded environments plus one release. They exclude automatic collision
inputs, friction and full scenes, so they are component costs only.

The actual original Franka controller/collision scene now accepts the first30
steps for both curved assets, with111 valid loaded environment-steps per asset
(B4, full output, default quadrature10). A220-step estimated-geometry attempt
first becomes invalid at step108 as the gripper makes contact, and the production
exception is raised at110. Environment0 reports region geometry status2, while
the other environments remain valid. The accepted prefix covers ground contact
and approach. It does not qualify gripper closure/lift throughput. Every frame,
including the first invalid result, remains available.

A separate read-only CPU inspection identifies the finite-box gate:
the normal root residual is`3.20e-17`, with positive principal curvatures
24.779/39.940 per metre. Its original2.052um excess used the box pose after
integration; the same-substep capture below corrects that excess to1.817um.
The half-extents are3/2/3.5mm. Source geometry is therefore outside the currently
supported interior Hertz region even with a converged normal root. Relaxing the
finite-face gate or tightening Newton/PCG tolerances would not validate pressure
near that edge. Reuse the existing [straight-edge physical controls](HARD_CONTACT_ESTIMATES_20261020.md#finite-straight-edge-pressure-candidates)
for an inexpensive finite-edge model. Curved moving-source/friction binding and
matched accepted max/full throughput also remain active work. The diagnostic
worker's first cache-owner attribute error remains in its original failed run.

The [curved-source packet](../evidence/20261019-qualified-ranking/hard-native-curved-source-v1/README.md)
binds the executed native, affine regression and actual scene checks to their
source commits. Reproduce its captured tasks with Slurm`rtx-mid`, image`1_26`, the
existing data-side environment, fresh data outputs and private node caches.

## Same-substep finite-face boundaries and edge pressure

Experimental engine [`660c85e98`](https://github.com/Roushelfy/Genesis/commit/660c85e9859d54adaf3cf4e69e2a06025d392dff)
retains the four finite-box half-planes in each Hertz region's tangent frame at
the actual contact evaluation pose. Release and partial reset clear them.
The existing root, curvature, finite-face and wrench gates are unchanged.
CPU/CUDA sphere/box controls pass interior,2um exterior, rotated, released and
restored cases. Independent box corners verify each recorded inequality.

The actual B4 Franka run still first rejects at step108. Recorded face distances
are4.842132,1.157868,-0.001816555 and7.001817mm: the converged normal root lies
**1.816555um outside** the selected face. This supersedes the earlier2.051902um
diagnostic based on the post-integration box pose. The old capture remains
immutable. Comparing all109 saved frames preserves exact positions, rigid loads,
history and acceptance. Maximum relative stress differences are below9.13e-16;
curvature differs by at most7.11e-15 per metre and normalized wrench error by
1.09e-15 absolute. The original exact-curvature comparison failure is retained;
the corrected reader declares a1e-13 curvature roundoff budget while leaving
production gates and the existing field budgets unchanged.

A compact circular lookup uses nine dimensionless edge offsets from-1 to2,
piecewise-linear support scale and a regular/edge pressure mixture. Training
minimizes complementary energy, without using reference pressure as a fit target.
Independent unilateral square-cell contact solves choose their own pressure and
support. Held-out offsets-0.625,0.125 and0.75 give5.41–8.10% cell-pressure error,
1.46–3.13% sampled tensor error and0.51–2.51% sampled von Mises peak error.
These pass the declared15%/10%/15% engineering budgets.

Circular parameters do not generalize sufficiently to flat ellipses. For axis
ratio0.399887, the original lookup gives15.31%/17.25% pressure errors at0/45deg,
so its campaigns remain FAILED. Fitting support scale, mixture and a tangential
shift independently by complementary energy improves these selected cases:

| Control | Pressure error | Sampled tensor error | Sampled VM peak error |
| --- | ---: | ---: | ---: |
| Captured root geometry, ratio0.727659 | 6.40% | 1.57% | 1.07% |
| Flattened ellipse,0deg | 9.73% | 2.44% | 2.81% |
| Flattened ellipse,45deg | 10.92% | 3.43% | 0.82% |

The rotated flattened trial shifts3.1545um along the edge. These are offline
fits (up to24.65s here), not runtime timings or a tested multidimensional table.
For the captured-root control, reference48-to72 refinement changes the sampled
tensor by1.05% and VM peak by1.96%; quadrature20-to40 changes trial cell forces
by less than0.012%. Observations lie at positive depths0.25–2 major radii.
The sharp-edge pressure is singular, and these samples do not qualify continuous
surface/global peaks, shell fields or friction.

The root-based pressure control also does not resolve actual source placement.
Material contact positions and forces must be rotated into the world frame
before comparing them with region geometry. The original lookup-v1 diagnostic
mixed these frames; its actual-contact diagnostic is superseded byv2/v3, while
its independent circular training and held-out controls remain valid. Correct
conversion places the captured force line approximately6mm from the normal
root. Native inputs already move the recovery source along the resultant force
line to retain its wrench; a pressure estimate at the geometric root cannot be
substituted for that source without a separate load/geometry assessment.

An independent intersection with the captured quadratic material surface
preserves the single-force wrench to3.37e-19Nm and locates the incoming outer
face46. That point is197.2um from the collision witness, and its local normal
differs14.53deg from the geometric root normal. It lies29.05um **outside a
different box edge**; the local Hertz width toward that edge is45.14um. Thus a
root-centered finite-edge correction alone cannot qualify this observed source.
The collision/material surface correspondence and the load-compatible finite
support need assessment before native pressure integration. The first inspector
failed because it read transposed surface connectivity incorrectly; its corrected
version and the original failure are both retained. This inspection establishes
geometry/wrench facts, not physical pressure accuracy.

The [finite-face packet](../evidence/20261019-qualified-ranking/hard-finite-edge-v1/README.md)
retains original failures, corrected readers, exact native/worker sources and
data-side array identities. Reuse its captured Slurm tasks with fresh explicit
data outputs, image1_26 and the existing environment/private caches. Actual
gripper admission and matched accepted pure-rigid/max/full throughput remain
required. The whole goal stays active.

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
