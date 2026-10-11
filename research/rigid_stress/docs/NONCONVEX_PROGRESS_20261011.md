# Native nonconvex contact recovery progress

This implements the active extension in `NONCONVEX_CONTACT_REQUIREMENTS.md`.
The previous egg milestone remains complete; native nonconvex acceptance is
not yet established. The initial branch source is `1d6babf7`.

## Required evidence and current work

| Requirement | Evidence to deliver | Status |
|---|---|---|
| Material normals and real surface topology | Cavity/U winding tests; no spatial sheet merging | Native implementation and CPU/CUDA checks pass |
| Surface provenance | Geom/pair and face/barycentric witnesses, actual boundary association, ambiguity diagnostics | Native resolver, raw witnesses and both-side GJK reference/frame checks pass CPU/CUDA; general live association pending |
| Intrinsic finite footprint | Verified paths, hole/gap exclusion, explicit radius/profile and distance error | Native path checks pass; footprint integration/tuning pending |
| Six-component vector fit | Original/projected/P2 wrench, local cones, offset tangential load and constrained corner fits | Standalone native dual, signed scatter and edge/corner/planar checks pass CPU/CUDA; live integration pending |
| Failure diagnostics | Ambiguity, unresolved integration/distance, numerical fitting and physical incompatibility | Pending |
| Same-mesh numerical recovery | Independent CPU FP64 nodal loads, complete residual/gauge rows, full/max stress | Nonconvex vector-load CPU FP64 comparisons pass on both backends; live acceptance pending |
| Dynamic lifecycle | Motion, slip/release, rotations, source changes, batch isolation, partial reset/state restore | Standalone source/traction transitions pass; complete online lifecycle pending |
| Existing Franka egg | Legal changing contacts, actual varied combined friction, rigid trajectory isolation | Pending |
| Matched performance | New-law four-scope baseline, detailed profiles, measured candidate decisions and repeated throughput | Pending |
| Reproducible delivery | Native code/example/tests, source and artifact identities, committed/pushed evidence | In progress |

## Design direction

First retain element ownership and edge adjacency by mesh vertex identity.
Orient the affine face normal away from its owning tetrahedron in Quadrants;
neither COM nor input winding defines the material side. Boundary asset
triangles must have one material owner. A selected boundary subset may have
open edges, but shared edges may not have more than two incident faces.

Next carry source identity and a witnessed reference anchor while retaining
the rigid application point. Build a device surface-distance path and a
six-dimensional local-Coulomb constrained fit. The elastic operator, complete
residual and full/max output lifecycle remain unchanged. No high-resolution
physical convergence experiment is part of this extension's iteration.

New runtime evidence uses the existing project data root, isolated run
`runs/20261011-nonconvex`. Python and numerical tests run only on allocated
`rtx-mid` compute nodes. No production stress arithmetic uses Torch.

## Geometry/source checkpoint

The production `StressModel` now records each supplied boundary triangle's
unique owning tetrahedron, opposite vertex and true edge neighbors. Its
Quadrants surface kernel orients normals away from that element. Duplicate
triangles, interior tetrahedral faces and nonmanifold shared edges receive
explicit asset errors. Coincident positions with distinct mesh indices do
not create neighbors.

`StressSurfacePaths` constructs edge ports and segments within actual
triangles. Port positions, edge lengths and shared shortest-path distances
are computed in Quadrants. A triangle-adjacency walk verifies coplanar
straight segments; it rejects missing triangles, crossings through holes and
unresolved vertex crossings. Otherwise, graph paths give conservative upper
bounds on intrinsic distance. Tests compare the full shared matrix against
independent SciPy Dijkstra, verify gap/hole exclusion, and check that nested
subdivision from 2 to 4 cannot increase the bounds. Neither count is selected
as the final production setting. Distance/profile approximation error and
initialization/storage cost still require measurement with the contact law.

`StressSourceState` carries geom/pair identity, the supplied witness, optional
reference face/barycentrics, declared uncertainty and free moment.
`kernel_resolve_sources` retains a reference witness or permits a bounded,
unambiguous nearest-triangle association. It preserves the original rigid
point and force, and forms `M_a = M_p + (p-a) cross F`. Source failures clear
the derived anchor/moment; one environment's transition does not affect other
environments. The current helper trusts an incoming reference witness as
authoritative. Mapping actual collision witnesses to that reference boundary
and preserving their geom/pair provenance are still required engine work.

These path/source helpers are **not yet connected to live recovery**. Live
recovery still uses the previous pad pressure mapping, with the corrected
material normals. The checkpoint does not establish native nonconvex contact
acceptance or its throughput.

Targeted checks use inexpensive connected U-shaped and closed-cavity voxel
meshes with arbitrary triangle winding. Twelve cases cover normals,
topological identity, invalid boundary assets, path gaps/holes, source
ambiguity, projected offset/free moments, failure transitions, and two
environments of full/max elastic output on each mesh. Independent CPU FP64
elastic checks retain `max(1e-11 N, 1e-8 ||rhs||)` for all equilibrium rows and
`rtol=1e-4, atol=1e-3 Pa` for stress fields. They do not yet check vector
tractions, contact scatter or moving-scene resets for this extension.

The targeted runs execute on Xeon Platinum 8562Y+ and RTX PRO 6000 Blackwell
Server Edition (97,887 MiB reported by the driver), Genesis 1.4.3 and
Quadrants 1.3.3, container `1_26`, FP64. Reproduction inside an allocated
`rtx-mid` node:

```bash
bash /absolute/path/to/Genesis/research/rigid_stress/scripts/check_nonconvex_geometry.sh cpu
bash /absolute/path/to/Genesis/research/rigid_stress/scripts/check_nonconvex_geometry.sh cuda
```

`RIGID_STRESS_RUN_DIR` selects a fresh output location under the project data
root; the script refuses to overwrite an existing result. Set
`RIGID_STRESS_CHECK_EXISTING=1` to include the existing rigid stress suite.
The script records source hashes, import location, GPU identity and pytest
output, and isolates compilation caches and temporary assets.

Fault injection restoring the COM rule fails both normal cases; substituting
Euclidean distance while retaining path diagnostics fails both path cases.
The initial launcher used a nonexistent pytest `--precision` argument and
was corrected to the existing precision marker. An initial source-wrench
assertion used the ideal authored anchor instead of the actual mapped anchor;
using the actual anchor fixes the frame check with the same `1e-22 Nm`
assertion. The failed outputs remain in the raw run directory.

The combined checkpoint passes all **12 CPU and 12 CUDA cases** (58.86 s and
28.20 s). The existing stress suite also passes all **52 CUDA cases**
(1100.34 s) and **31 CPU cases, 21 CUDA-only skips** (444.28 s).
These counts establish the stated geometry/source/elastic tests, not live
vector contact acceptance. Raw runs are under
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex`.

Source checkpoint: `2bb3d2e423ef1e69cd00733118b14bb2a9e05614`.
The [completed publication manifest](../evidence/20261011-nonconvex/geometry-checkpoint-v2/manifest.json)
records the exact tested source hashes, closed logs, fault-injection programs
and launchers. All **48 publication artifacts and their originals** pass
SHA256 verification; manifest SHA256 is
`3a9b84695c2fa23805b75f0ff3ee0841511a728edb11f2c48a2dfa2fcd8ab696`.
The immutable [earlier checkpoint](../evidence/20261011-nonconvex/geometry-checkpoint-v1/manifest.json)
contains 46 verified artifacts and records the existing CPU run as still
pending at its publication time; the completed checkpoint adds its now-closed
CPU and allocation logs. No new-law live/policy rate is reported.

## Next integration work

1. Associate retained GJK/MPR and nonaggregated SDF producer points with the
   actual reference boundary, carrying face/barycentrics and bounded source
   uncertainty. Finish primitive/special and aggregate-source representation.
   Internal decomposition faces must not become material exterior faces.
2. Connect witnessed anchors and intrinsic finite profiles to six-dimensional
   vector traction fitting and signed P2 scatter. Preserve the original
   wrench after projection and scatter, with per-sample inward Coulomb cones.
   The current rolling/torsional-mode rejection remains explicit until actual
   free moments are available.
3. Distinguish distance/integration resolution, numerical stagnation and
   physically incompatible inputs; demonstrate that ordinary egg grasps are
   not rejected by geometric approximation.
4. Validate complete contact/load/elastic CPU consistency and actual scene
   lifecycle, then establish the new four-scope baseline and detailed stage
   profiles. Evaluate local/frontier work, shared geometry, per-face distance
   reuse, interior fits, compaction, valid history and native scheduling from
   those measurements. The old load-model throughput is historical evidence.

## Collision witness producer checkpoint

The narrowphase now retains a typed `ContactWitness` beside each rigid
contact's geom/pair identity. It carries both world-space producer points,
their method and optional emitting-vertex indices. These fields are allocated
only for stress-enabled scenes, belong to the state record, follow raw contact
slots through the GPU queue/gather and contact sort, and are cleared on partial
resets. Hibernated-contact compaction also preserves their pair identity.

GJK preserves the correspondence after rejecting degenerate witness pairs.
MPR stores its original weighted portal points separately from the unchanged
rigid application point. Perturbed points are independently unrotated into
each geom's original frame. MPR's exact/approximate labels describe its portal
classification; they do not establish exact material-boundary coordinates for
arbitrary assets. The nonaggregated SDF path preserves the actual emitting
vertex and the queried-gradient projection, independently of a synthetic
emitted penetration. Primitive/special and aggregate paths retain an explicit
normal-offset estimate. Those estimates are not authoritative face witnesses.

This is source retention, not reference-boundary association. Actual
reference face/barycentric mapping, decomposition-interior rejection, SDF
uncertainty and aggregate-island representation remain pending. Live stress
still uses the old pressure mapping. No vector traction, new-law numerical
acceptance or throughput result follows from these source tests.

The five source/lifecycle cases exercise ordinary MPR, GJK and patch-enabled
GJK, real collision surface checks, two simultaneous geom pairs including a
rotated geom, environment isolation, partial reset and state-record restoration.
U-shaped and cavity fixtures check SDF emitting-vertex identity. A separate
eight-point octagonal box patch checks both actual geom boundaries and every
contact's midpoint/penetration correspondence. It uses an eight-vertex,
six-tetrahedron cube, not a high-resolution elastic mesh.

The initial one-pair check passes all five CUDA cases; CPU passes four, with
its sole failure a `2.16840434e-19 m` one-rounding-unit contact-position
difference in an added bitwise comparison. Floating collision comparisons
now use 32 FP64 epsilon at the 1 cm pair scale; integer identity remains exact.
The first two-pair GPU check then failed because independent atomic slot
reservations returned `[1, 2]` versus `[2, 1]`. Canonical identity/position
alignment fixes that comparison without changing its numerical thresholds.
These changes define the new source-retention test. Final contact wrench,
complete equilibrium and stress budgets are unchanged.

Both final source suites pass **five CPU and five CUDA cases** (434.94 s and
562.82 s). The octagonal patch separately passes **one CPU and one CUDA case**
(65.18 s and 107.02 s), giving six new cases on each backend. The existing
stress/geometry suite passes **64 CUDA cases** (1098.85 s) and **43 CPU cases,
21 CUDA-only skips** (460.39 s). Six selected collision checks also pass
(302.32 s): two GPU split/monolithic cases, CPU/GPU box-patch cases and two
GPU hibernation/pruning/reset cases. Those legacy collision cases do not
configure stress, so they do not establish stress-enabled hibernation coverage.

All final runs retain identical before/after source hashes. The corrected
five-case test ran under a temporary `_v5` module name to leave the concurrently
running regression's imported files frozen; its committed test has identical
bytes. An earlier development run used a nonexistent VectorTensor `.ndim`
attribute and was corrected to inspect its shape. Another development run
edited imported source during compilation, invalidating Quadrants' source/AST
lookup; it supplies no numerical evidence. Accepted runs froze imported code
and verified its hashes. Failed development logs remain in the raw data root;
the publication also preserves the failed two-pair comparison and its test.

GPU checks used RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, driver
595.71.05, Genesis 1.4.3, Quadrants 1.3.3, container `1_26`, FP64. These are
correctness timings with isolated compilation caches, not throughput results.

Reproduce inside an allocated compute node:

```bash
bash /absolute/path/to/Genesis/research/rigid_stress/scripts/check_nonconvex_witness.sh cpu
bash /absolute/path/to/Genesis/research/rigid_stress/scripts/check_nonconvex_witness.sh cuda
```

The script isolates runtime/cache paths, refuses existing evidence and checks
the tested source hashes before and after execution. Set
`RIGID_STRESS_CHECK_EXISTING=1` to include the old stress and geometry suites.

Source checkpoint: `039686c4624d1b375b98cd28903fdca7975b7ede`.
The [publication manifest](../evidence/20261011-nonconvex/witness-checkpoint-v1/manifest.json)
records closed CPU/CUDA logs, unchanged source hashes, source snapshots,
launchers, the test-module rename and the failed two-pair comparison. All
**43 publication artifacts and their immutable data-side originals** pass
SHA256 verification; manifest SHA256 is
`751ccc3d325bfc3bd75a08778e286527b96e13ae7a6124d2446cb578c8a91b21`.
This completes the collision producer checkpoint, not the active nonconvex
recovery goal. Reference association, vector loads and matched performance
remain pending as listed above.

## Reference-source and vector-traction checkpoint

The native source capture now selects the stressed link's actual A/B collision
witness, geom, pair, method and emitting vertex, and transforms it through the
existing association's material frame. It preserves the original force point.
Declared uncertainty remains an input: a witness method alone does not certify
reference-boundary accuracy. Actual contact moments still require their torque;
the existing stress configuration rejects enabled rolling/torsional modes.

The resolver accepts uncertainty spanning a covered part of the same planar
wall, while retaining ambiguity across distinct branches. A zero-distance
edge/vertex alternative must share the active mesh vertex identities. The
coplanar segment check also recognizes an endpoint on a truly incident target
face; coincident vertices on separate sheets do not qualify. The small live
GJK source fixture checks both stressed geom sides, rotations, tangential
forces, reference barycentrics, offset moments, contact death/rebirth and a
partial reset. This is source capture/association evidence, not a live
nonconvex stress trajectory. Primitive, SDF, decomposition and aggregate-source
association still need the complete online contract.

`stress/traction.py` implements the declared positive weighted vector-traction
QP through a six-component dual. Every positive integration sample uses its
local inward material normal and supplied actual friction coefficient. The
Coulomb projection and generalized Jacobian are native Quadrants. A regularized
six-by-six Newton solve, Armijo checks and a gradient fallback operate per
contact/environment; moments are scaled by the unchanged physical radius.
The internal termination budget is **1e-9** in these checks. The 1e-12 matrix
ridge is regularization, not a wrench acceptance threshold.

The scatter uses signed P2 shape values and retains the original six-component
wrench. Tests manufacture independent FP64 primal/dual KKT solutions on a
concave edge and cavity corner with actual **mu = 0, 0.25, 0.55, 0.85, 1.2**.
They compare sample and nodal forces, local cones, force and moment about the
original offset point, then recover full/max fields using the existing native
elastic model. Independent NumPy/SciPy FP64 elasticity checks
include all six gauge rows, complete stress tensors and the global peak.
Acceptance remains force/moment 1e-8, complete equilibrium
`max(1e-11 N, 1e-8 ||rhs||)`, and stress `rtol=1e-4, atol=1e-3 Pa`.
No high-resolution physical convergence workload was run.

The edge/corner fixtures supply area/profile weights offline on analytically
verified incident planar walls; they do not claim general intrinsic footprint
generation. A separate planar batch includes zero friction, interior friction,
sliding-boundary traction, eccentric sliding traction, a pulling wrench and
release/rebirth with environment isolation. The fitted separating certificate
is explicitly **sampled-wrench infeasibility**, not an assertion that an
unresolved continuous footprint is physically incompatible. Invalid input,
empty integration and numerical fitting failure have distinct statuses.

The first projection test's nonsmooth three-dimensional SLSQP oracle returned
a line-search failure on both backends although the native edge/corner checks
passed. Its replacement uses an independent convex two-variable QP with linear
constraints, plus Moreau polar feasibility/orthogonality and Jacobian finite
differences. Final wrench, residual and stress budgets were not relaxed. The
failed run and its exact source are preserved in `traction-v1`.

| Closed allocated run | CPU | CUDA |
|---|---|---|
| `source-association-v1` (existing geometry plus new sources) | 16 passed, 109.15 s | 16 passed, 111.61 s |
| `traction-v1` (initial oracle) | 1 failed / 2 passed, 15.09 s | 1 failed / 2 passed, 11.38 s |
| `traction-v2` (projection, edge/corner, planar/lifecycle) | 4 passed, 18.79 s | 4 passed, 11.92 s |
| `contact-checkpoint-v1` (final combined module layout) | 20 passed, 149.01 s | 20 passed, 139.10 s |

These use the same RTX PRO 6000 Blackwell Server Edition / Genesis 1.4.3 /
Quadrants 1.3.3 / FP64 setup as the witness checkpoint, with isolated caches.
The planar test was consolidated into `test_stress_traction.py` after the
closed v2 run. A combined source/geometry/traction rerun passes twenty checks
on each backend, with identical frozen before/after source hashes. Reproduce
that final module layout through the portable command below:

```bash
bash research/rigid_stress/scripts/check_nonconvex_contact.sh cpu
bash research/rigid_stress/scripts/check_nonconvex_contact.sh cuda
```

This checkpoint does **not** switch `RigidStressRecovery` or its fused graph
to the new law. Its general subproblem stores one profile per supplied
sample/contact/environment; that dense representation is not a selected live
layout. The remaining major work is native intrinsic profile/local integration,
general source mapping, online diagnostics/lifecycle, the Franka egg replay,
then a matched four-scope baseline, profiling and measured optimizations.

Source checkpoint: `4981d82391192777be4b9e8af52823f2f78f01e7`.
The [publication manifest](../evidence/20261011-nonconvex/contact-checkpoint-v1/manifest.json)
records all eight successful/failed test logs, frozen source snapshots,
launchers and the final test consolidation. All **72 publication artifacts
and their preserved data-side originals** pass SHA256 verification on an
allocated compute node; manifest SHA256 is
`7d74c3fc80864be7fc010a7dff9ccf76d9a79867c42f534dd9f11806d55a841a`.
The active nonconvex goal remains incomplete as stated in the manifest.
