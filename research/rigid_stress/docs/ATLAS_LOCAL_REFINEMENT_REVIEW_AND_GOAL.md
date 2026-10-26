# Local-refinement atlas: review and remaining native goal

**Active user direction, 2026-10-25.** Complete the original locally refined
feature-mesh route for both midsurface shell and three-dimensional volume FEM.
The previous response-partition experiments remain valid evidence for their
implemented scope. Their negative timings do not close this spatial
approximation route.

## Primary executable CPU reference

Use the new [cpu_repro implementation](../atlas_study/cpu_repro/README.md)
and its [measured report](../atlas_study/cpu_repro/REPORT_ZH.md) as the primary
algorithm and discrete numerical reference for this follow-up. The source
snapshot is commit
[`a10c5abe`](https://github.com/Roushelfy/Genesis/tree/a10c5abe3b5bc3c5caa57d3d4d9dadab2b277ade/research/rigid_stress/atlas_study/cpu_repro).
The new executable reconstruction is named `cpu_repro`; it reconstructs the
old `cpu_next` report, whose original source remains unavailable.

Start with [model.py](../atlas_study/cpu_repro/model.py) for the genuine
`T_i=[P,B_i]` feature spaces and smaller `K_i`,
[selection.py](../atlas_study/cpu_repro/selection.py) for source-wise nearest
and narrow-transition selection, and
[pipeline.py](../atlas_study/cpu_repro/pipeline.py) for same-feature RHS
aggregation, batched solves and merged displacement/stress recovery.
The earlier independent-mesh examples in the
[reference guide](ATLAS_CPU_REFERENCE_GUIDE.md) are supplementary controls.

This reconstruction qualifies a 5,196-DOF shell CPU model with prescribed
material-cap loads. It does not already qualify volume FEM, arbitrary
nonconvex mapping or native GPU throughput. Adapt its construction to both
native backends while preserving the existing pressure/radius/friction
semantics. Use the new measured evidence, including the conditional/negative
history results, rather than treating the old timing claims as guaranteed
gains. Start with focused low-model numerical controls; rerunning the whole
CPU study or a physical-convergence campaign is not a prerequisite.

## Review judgment

Review of delivery `2d70f353466a9a31888c6254bd3e5d7ba9e0716e` confirms
substantial native implementation, numerical and end-to-end tuning for H,
FP32 response storage/accumulation, global screening and sustained history.
Retain those scoped negative results and the optional angular pressure repair.
They do not need an unchanged rerun for this follow-up.

The measured atlas, however, uses regional load injections into one complete
mother response:

    U_i = K_g^-1 R E_i
    sum_i U_i g_i = K_g^-1 R sum_i E_i g_i

In experimental source
[`atlas.py`](https://github.com/Roushelfy/Genesis/blob/d5641351a127f674b4a19bf3d921cc3be0200789/genesis/engine/solvers/rigid/stress/atlas.py),
`self.response = model.surface_inverse.info` reuses the same full response.
The independent regional displacement arrays still span all mother nodes.
The direct variant saves those arrays but uses the same mother response.
The 18-direction coarse variant in
[`atlas_coarse.py`](https://github.com/Roushelfy/Genesis/blob/d5641351a127f674b4a19bf3d921cc3be0200789/genesis/engine/solvers/rigid/stress/atlas_coarse.py)
retains the full exact complementary response.

This tests exact response partitioning, with extra selection, grouping and
merge costs. It does not create meshes/spaces that are fine near a
representative contact region and coarse elsewhere. The original accuracy
versus cost mechanism is therefore still untested natively. The final report
itself states that no deliberate spatial approximation was used and that its
results do not exclude approximate local refinements.

The current status is:

| Candidate family | Review status |
| --- | --- |
| Locally refined atlas, its per-mesh aggregation and merged recovery | Exact same-mother partition trial completed; true local-refinement route remains open in both backends |
| Complete H, FP32, global screening and sustained prediction | Native trials and tuned negative results accepted within their recorded models/workloads |
| Prior pressure/fallback optimizations and angular repair | Preserve the maintained implementation and its qualifications |

Keep the previous reports, source pins and audit JSONs as historical records.
Their fourteen-cell completion labels describe the tested implementations;
the active completion status follows this review and [GOAL.md](../GOAL.md).

## Required work

Build a real offline library of locally refined discretizations or
Galerkin-compatible reduced feature spaces. Geometry, material and thickness
are fixed; actual contact locations and loads remain arbitrary.
Each feature must retain global elastic coupling while using fewer
independent coordinates/work than the appropriate uniformly or jointly
refined mother reference. A regional view of an unchanged full response
alone is not such a feature space.

A useful common-mother construction is:

    T_i = [P, B_i]
    K_i = T_i.T @ K_m @ T_i
    b_i = T_i.T @ R @ f_i
    K_i z_i = b_i
    q = sum_i T_i @ z_i
    stress = S_m @ q

Here P is a shared coarse space, B_i is genuinely local fine enrichment,
K_m/S_m belong to a fixed physical mother discretization, and R is its
common load-relief map. Build compatible geometry/directors, constraints and
prolongations; preserve partition continuity, rigid modes, work and force/
moment transfer. Another validated reduced construction is welcome.

Within a backend, project the same mother mass, gauge and inertia relief;
do not balance each feature independently. Shell and volume each have their
own discretization, maps, output and numerical reference. Preserve shell
rotations, bending and side-dependent eccentric moments. Volume refinement
must cover necessary through-thickness behavior and all 3D coupling.

Document mesh/space sizes, rank of T_i, K_i dimensions/nonzeros, factor fill,
local/core/halo and transition sizes, preparation/storage, active groups,
and fallback frequency. Demonstrate actual spatial reduction rather than
only relabeling complete response columns. Retain a global coarse space
and add fixed refinement or validated safeguards for remote hot regions,
including nonconvex thin necks.

Map the original native finite-pressure/friction load at its real material
position. Select the nearest feature per contact with a narrow continuous
overlap, allow all necessary junction contributors, merge equal
(feature, environment) RHSs, and sum tensors before stress criteria.
Check the whole finite footprint, surface-side and nonconvex identity.
Measure covered-feature, union or mother fallbacks without dropping loads.
No online remeshing/factorization is required for a fixed prepared library.

Integrate into the existing rigid stress feature, with numerical preparation
and runtime in Quadrants. Reuse validated selection, coverage, grouping,
lifecycle and response machinery where useful. Start with small planar/box
controls for debugging, then run curved hollow-egg Franka and nonconvex controls
for both backends. A planar-only demonstration does not complete this goal.

## Numerical solve accuracy and spatial approximation

The new atlas is an explicitly identified spatial approximation. Preserve
existing exact-mode assertions and the numerical solve tolerances for each
declared feature discretization. Check the unscaled reduced operator as well
as any scaled solve.

For assigned loads f_i, feature numerical residual is

    K_i @ z_i - T_i.T @ R @ f_i

The complete mother residual is

    R @ sum_i f_i - K_m @ sum_i (T_i @ z_i)

That mother residual generally includes spatial approximation error.
Requiring the approximate result to pass the mother's exact-solve tolerance
and then always doing a full mother correction would erase the intended
reduction. It is a diagnostic or a basis for a validated error estimator,
not automatically the numerical acceptance test for the new reduced model.
Keep original exact acceptance when exact/mother fallback is actually used.

Qualify the approximate mode separately: feature-system numerical residuals,
common gauge/relief, load work and wrench preservation, plus declared
full-field/sample-maximum/task error budgets against matching mother or
joint-refined references. Current native pressure/radius semantics and
physical acceptance remain unchanged. This distinction does not authorize
silently weakening an existing exact-mode check or calling approximation
numerical equivalence.

A sum of independently solved source spaces is generally not the joint-space
Galerkin solution. Each source residual is orthogonal to its own test space;
that does not imply orthogonality to every other feature space. Validate the
combined field for the actual total load, including cancellation, distant
contacts and remote maxima. Independent single-source error checks alone
cannot certify the combined answer.

## Performance and completion

Measure the accuracy/time/memory tradeoff. Include the current coarse native
baseline, the actual feature library, a joint/local-refinement control and a
suitable uniformly refined reference where useful. Comparing only against
the cheapest coarse mesh can miss the intended accuracy benefit. Do not
claim a speedup by comparing different loads, output sampling or error budgets.

Use low/moderate models that reveal a real coarse-versus-local DOF difference,
not high-resolution physical convergence campaigns. Sweep a small meaningful
range of region sizes, library sizes and transition widths. Test arbitrary
moving/asymmetric contacts, friction, junctions, both material sides, release,
partial resets and checkpoints. Validate complete fields and sampled maxima
over the object, separately for both backends.

Profile grouping, projection, selected solves/response products, lifting,
stress output, acceptance, history and fallback. Measure module and complete
Franka scene latency, useful batch throughput and shared/per-environment/
peak memory; account for construction separately. Optimize the bottleneck
and promising combinations instead of repeating unchanged negative H/history
campaigns. Any new use of those methods on genuinely smaller feature
operators needs its own incremental evidence.

Completion requires real locally refined/reduced feature implementations in
both backends and numerical plus tuned end-to-end accuracy/cost decisions.
A measured negative result is acceptable after testing this actual mechanism.
The previous full-response partition's negative result cannot substitute for it.

Start from [cpu_repro](../atlas_study/cpu_repro/README.md) and its
[new measured results](../atlas_study/cpu_repro/REPORT_ZH.md). Use the
[CPU reference guide](ATLAS_CPU_REFERENCE_GUIDE.md) for the earlier
supplementary controls and provenance. Preserve all earlier positive/negative
evidence, source identities and storage policies.
