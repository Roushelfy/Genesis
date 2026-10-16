# Preserve physical U initialization coordinates on the next CAD mesh

This checkpoint follows the [2026-10-16 practical goal](PRACTICAL_VALIDATION_GOAL.md).
Its completed checks are accepted within their stated scope; unresolved physical
tails remain limitations or research follow-ups. The full historical matrix
is no longer an engineering-delivery prerequisite.

The existing shared-initialization U family retains the same pad5 geometry,
six physical initialization coordinates, materials, forces, friction, gravity
and12/604/64/64 loading stages. Every one of744 stages has an independent
original-force and full-field audit at each of three body levels. Final tensor
changes44.86%/12.31% remain unresolved. These unresolved tails remain recorded research limitations under the
[practical goal](PRACTICAL_VALIDATION_GOAL.md).

## Actual fourth-level failure and preserved original controls

The first new shared-pad5 producer720289 uses the existing16146-tet remesh at
0.875mm. It naturally fails before any accepted physical stage: two original
initialization coordinates are30.535674um from their nearest P2 nodes, exceeding
the unchanged1e-12m node check. Source and input-source closures match. The two
unstarted dependent jobs720290/720291 are cancelled only after confirming their
pending state; the journal and all failed outputs remain.

The legacy pad11 producer716311 is a different reference control. It does not
provide the missing shared-pad5 fourth level. No anchor or tolerance is moved
to turn the failed shared case into a pass.

Diagnostic720300 first fails because its isolated source package lacks the
transitively imported `validation_meshes` module. That naturally terminal,
source-closed failure is retained. After adding the missing source dependency,
diagnostic720307 completes with all six original coordinate/component values
unchanged and the same two30.535674um node mismatches.

Its point locator also flags some boundary coordinates within3.7e-17m of actual
nodes as outside. Consequently raw locator flags alone cannot establish material
exclusion at the boundary. The original existing-node distances and subsequent
CAD/used-vertex checks supply separate geometric evidence. No interpolated
off-node constraint or new nonlinear boundary algorithm is accepted here.

## Mesh conforms to the original physical coordinates

Generator720322 completes using the original U CAD constructor, mesh size and
five distinct physical anchor positions underlying the six component constraints.
It introduces these required CAD points through OpenCASCADE fragmentation;
the [official Gmsh API](https://www.gmsh.info/doc/texinfo/gmsh.html)
and [point-fragmentation example](https://gmsh.info/doc/cookbook/geometry/embed-point.html)
document the underlying conformal-entity operation. The actual pinned runtime
uses Gmsh4.14.1 and retains its own API/source identity.

The resulting mesh has16329 tetrahedra and4126 used volume vertices. Every
original anchor is an actual used tetrahedral vertex under the original1e-12m
check; the largest recorded distance is8.999e-35m. Original coordinate and
component arrays remain exact. CAD-surface distances at the original points
are at most1.520e-17m.

CAD volume before and after the point operation is exactly
2.180876878758457e-6m3 in the retained records. CAD COM passes the original
1e-14m check. Every tetrahedron has positive volume and the boundary manifold
count check passes. Reusable NPZ, Gmsh MSH and full metadata remain.

The discrete polyhedral volume is2.157955506779733e-6m3. Thus the mesh still
approximates the declared CAD geometry; CAD/body-volume/contact-trace errors
remain coupled. Material density is not retuned to force equal discrete mass.
Mesh generation solves zero contact equilibria and does not establish a
fourth-level stress or convergence result.

## Original comparison qualification and deferred full reference

The generalized four-level comparator first re-runs the original three closed
levels in720288. Every original result row, adjacent-change record and all75
material probes match the prior comparison exactly. Before/after source hashes
match. This qualifies accepting another declared level without dropping any
earlier level or altering the comparison arithmetic.

The corrected producer720325 now uses the anchor-conforming mesh with the same
pad5, six coordinates and all original controls. Independent all-stage
auditor720326 follows it; comparison720327 retains all four body levels,
all original probes, raw lower/upper global peaks and the original protocol.
After the user changed the delivery priority, producer720325 and its pending
auditor/comparator720326/720327 were cancelled. Their confirmed pre-cancellation
states, partial outputs and command journal are preserved. This is a priority
change, not a numerical failure or accepted fourth-level physical result.
The verified original-three-level comparison and anchor-conforming setup remain
usable evidence. See `meta/application-first-priority-job-cancellations-v1.json`.

Exact commands, source identities, cancelled-dependency state checks and the
original-three-level exact-reproduction proof remain under the established
runtime root:

```text
meta/u-shared-initialization-fourth-body-commands-v1.json
meta/u-shared-fourth-anchor-failure-dependent-cancellation-v1.json
meta/u-shared-fourth-anchor-transfer-command-v1.json
meta/u-shared-fourth-anchor-transfer-command-v2.json
meta/u-shared-four-comparator-original-three-exact-records-v1.json
meta/u-shared-anchor-conforming-mesh-command-v1.json
meta/u-shared-initialization-fourth-body-commands-v2.json
```

The [closed diagnostic/setup archive](../evidence/20261014-physical/u-shared-fourth-anchor-diagnosis-v1/manifest.json)
includes both real failures, corrected
geometry checks, exact original-comparison reproduction and all direct inputs.
All228 entries verify against228 runtime originals. Manifest SHA256:
`9798cd2888fa6321fb2a9b6805121f8ee3a72715ee18b2ad672abe09fb8c1c49`.
It preserves the compact diagnosis without requiring another complete raw744-stage
publication as the next delivery step. The practical work now prioritizes
representative egg/Franka and nonconvex R/L/M reconstruction comparisons,
separate feedback effects, recommended settings and warmed application cost.
