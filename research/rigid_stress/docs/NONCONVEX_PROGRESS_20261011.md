# Native nonconvex contact recovery progress

This implements the active extension in `NONCONVEX_CONTACT_REQUIREMENTS.md`.
The previous egg milestone remains complete; native nonconvex acceptance is
not yet established. The initial branch source is `1d6babf7`.

## Required evidence and current work

| Requirement | Evidence to deliver | Status |
|---|---|---|
| Material normals and real surface topology | Cavity/U winding tests; no spatial sheet merging | Native implementation and CPU/CUDA checks pass |
| Surface provenance | Geom/pair and face/barycentric witnesses, actual boundary association, ambiguity diagnostics | Native resolver and raw collision witness retention pass CPU/CUDA checks; reference association pending |
| Intrinsic finite footprint | Verified paths, hole/gap exclusion, explicit radius/profile and distance error | Native path checks pass; footprint integration/tuning pending |
| Six-component vector fit | Original/projected/P2 wrench, local cones, offset tangential load and constrained corner fits | Pending |
| Failure diagnostics | Ambiguity, unresolved integration/distance, numerical fitting and physical incompatibility | Pending |
| Same-mesh numerical recovery | Independent CPU FP64 nodal loads, complete residual/gauge rows, full/max stress | Nonconvex elastic checks pass; vector contact loads pending |
| Dynamic lifecycle | Motion, slip/release, rotations, source changes, batch isolation, partial reset/state restore | Pending |
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
the publication will also preserve the failed two-pair comparison and its test.

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
