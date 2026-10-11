# Explicit shared finite support for a plane contact pair

This follows [the squared-fan checkpoint](NATIVE_SQUARED_FAN_PROGRESS_20261011.md),
source `e1ab516321884034e73da93bbb7dac3e5e8a37c9`. It evaluates a different,
explicitly declared load-support model on the saved tick710 and tick1160 plane
inputs. It is not selected production behavior, live acceptance, a throughput
measurement or completion of the active nonconvex goal.

## Candidate law and original-wrench acceptance

Each saved plane/egg pair has two actual witnessed contacts, not one. The
previous per-contact law cannot represent one of their forces on its own
declared intrinsic disk. This does not prove that the entire multi-contact
load is physically incompatible. The independent per-disk certificate remains
correct for that specific declared support law.

The candidate builds each original disk with its original anchor, physical
radius, Q10 squared-coordinate finite fans, eight intrinsic edge subdivisions,
owner-oriented normals and positive area/profile weights. It forms an explicit
union of those disks for this identified collision pair. Every exported sample
retains a verified path from its constituent witness within that witness's
unchanged radius. No disk radius, original force, original application point,
original friction or free moment is changed. No contact or traction sample is
discarded to make a fit pass.

The two disks can be disconnected. Therefore this is a model for partitioning
the pair's load across identified actual contact islands, not a geodesic
shortcut connecting sheets or a claim that each disk grew. A production API
must declare this scope and preserve all source identities. Arbitrary contacts
from different geom pairs, bodies or ambiguous branches must not be grouped
silently. Destination friction must remain attached to the actual source
surface; the two contacts of each saved pair have the same measured mu.

The candidate deliberately checks more than the aggregate wrench: it fits
each original input wrench separately on the shared finite support. Each fit
preserves `F_i` and `M_i + (p_i-a) cross F_i`, with the original radius also
retained as that input's moment-conditioning scale and acceptance budget.
Summing the component tractions then preserves the complete original pair
wrench. Positive component compression and local Coulomb feasibility also
hold for their sum when the source friction is common.

There is also a separate aggregate-only comparison. It preserves the pair's
six total components but does not establish each original input's wrench.
Its smaller iteration count is not a reason to substitute it for the
componentwise checks or silently change their acceptance.

## Numerical evidence

Independent inscribed 128-sided cone LPs construct each component and aggregate
wrench on the union, with true circular-cone feasibility checked afterwards.
The Quadrants candidate uses the existing FP64 dual objective/projection,
`9e-9` internal stopping, 128 evaluations and final `1e-8` normalized original
force/moment checks. The one-anchor Euclidean-radius check is replaced in
this offline candidate by verification of every constituent intrinsic disk;
the radius value itself is unchanged.

| Saved frame | Wrench | Native evaluations | Normalized force / moment error, CPU initial full check |
|---|---|---:|---|
| tick710, mu=0.6008812824796964 | original contact0 | 34 | 3.94e-11 / 7.10e-11 |
| tick710 | original contact1 | 10 | 6.22e-16 / 8.46e-16 |
| tick710 | aggregate comparison | 4 | 2.40e-12 / 9.14e-13 |
| tick1160, mu=0.6531024606382921 | original contact0 | 8 | 1.23e-11 / 2.84e-11 |
| tick1160 | original contact1 | 17 | 3.08e-12 / 1.06e-12 |
| tick1160 | aggregate comparison | 5 | 4.54e-9 / 3.74e-10 |

Six initial CPU and six initial CUDA cases pass sample wrench/cones, signed P2
scatter, independent nodal loads, complete equilibrium including all gauge
rows, every stress tensor/von-Mises value and the global unaveraged peak.
Each uses the same inexpensive full shell: 42 collision vertices, 810 P2
nodes and 480 tetrahedra. No high-mesh or material quadrature study is run.
The original residual budget remains `max(1e-11 N, 1e-8 ||rhs||)` and stress
remains `rtol=1e-4`, `atol=1e-3 Pa`. Initial per-component full residuals are
at most about `1.03e-12 N`; the aggregate tick1160 comparison is about
`2.02e-10 N`, inside its original relative budget.

The independent CPU oracle first minimizes the convex dual with SciPy
`trust-exact`, without receiving native multipliers or native fitted forces.
Large dual values can make energy comparison lose the precision needed for
its final wrench check. A separate analytic-Jacobian `hybr` root polish starts
from that independent convex solve and is accepted only by the same final
force/moment budgets. Optimizer success flags alone are not acceptance.
The older unpolished oracle failure is preserved.

The initial CPU snapshot writer retained views of reusable force buffers.
Its runtime assertions were completed before reuse, but the later saved
per-case arrays were overwritten. Cross-backend archive validation caught
that error. Both initial cross-backend failures are retained, and the final
writer explicitly copies forces, nodal fields, displacement, full stress and
von Mises before proceeding to another case. Final CPU/CUDA reruns and direct
cross-backend checks are recorded in the complete evidence capsule.

The final copied-snapshot rerun passes all six CPU and six CUDA cases. Direct
canonical nodal differences are at most `8.99e-16 N`, direct native peak
differences at most `1.02e-7 Pa`, and independently recomputed complete tensor
differences at most `1.37e-8 Pa`. Every native full tensor and von Mises field
also passes the unchanged comparison budgets across backends.

The [immutable evidence manifest](../evidence/20261011-nonconvex/shared-support-checkpoint-v1/manifest.json)
records 330 artifacts, 6,780,855 bytes and SHA256
`aa54fb5f5f6204b614018ce378d5c74d9d8d018554fd98725e5abe6daffc1871`.
It includes all six candidate-driver revisions, their frozen source, original
failed oracle/allocation/snapshot comparisons and the final successful rerun.
Its production source base remains `e1ab5163`.

The hardware remains RTX PRO 6000 Blackwell Server Edition, driver595.71.05,
Xeon Platinum8562Y+, container `1_26`, Genesis1.4.3, Quadrants1.3.3, FP64.

## Production implementation still required

1. Expose a declared pair-union support scope while preserving the existing
   point-support API and reproducible model metadata.
2. Keep complete pair membership together during contact streaming. A block
   boundary must not truncate a union. Account for source/geom identities,
   changing contact counts, valid zero-load contacts, destination friction,
   enabled free moments and source failure propagation.
3. Reuse the existing per-source packed profiles/finite geometry. Iterate
   shared source samples for each target component instead of replicating
   candidate arrays for every component. Preserve each component's original
   force/moment acceptance in commit and nodal scatter.
4. Verify independent full/max fields, disjoint source pairs, concave sheets,
   friction changes, release/birth/death, rotations, partial reset/state restore
   and the existing no-feedback rigid trajectory contract.
5. Run longer live trajectories on several seeds. Rebuild the four-scope
   baseline only after model/source robustness, then measure cooperative fit,
   scatter/commit fusion, shared layout, lifecycle costs and large batches.

An applicable candidate now has constructive numerical evidence. It remains
actionable engineering work, so the active goal must continue.
