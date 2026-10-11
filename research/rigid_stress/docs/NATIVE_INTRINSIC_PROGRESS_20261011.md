# Native intrinsic contact integration checkpoint

The active extension starts at `1d6babf77848c46ba616da8bbb7062260a7ccd09`.
This work follows the published `c3083710` footprint checkpoint. It does not
complete the nonconvex goal or establish new-law throughput. Production stress
numerics, including source projection, path distance, vector traction fit,
signed P2 scatter and elastic recovery, execute in Quadrants.

## Integrated path and unchanged acceptance

`RigidStressOptions(contact_model="intrinsic")` selects the native integration.
The previous pressure API/default remains available. The new path preserves
the original rigid force point, raw geom/pair and producer witness, and the
configured radius and actual combined friction. It streams every valid contact
through bounded contact blocks; additional contacts receive subsequent blocks.
The first implementation reads one device maximum contact count to schedule
these blocks. This synchronization is part of its eventual timing scope.

Only the anchor face uses three finite incident triangles. Other faces use the
same Q10 quadrature and declared intrinsic port-path/profile law. Shared
geometry and a 300-sample local anchor buffer replace the dense per-contact
geometry cloud. Positive integration samples are packed with their original
indices; their exact area/profile weights and signed P2 bases are preserved.
The packed capacity accommodates the full original 8,200-slot rule, so packing
does not discard samples or change the physical footprint.

The internal dual stopping tolerance is `8e-9`, subject to separate final
original-point force/moment acceptance at `1e-8`. Every fitted sample is checked
against the unchanged inward material Coulomb cone. CPU FP64 elastic checks
retain the complete equilibrium matrix, including all six gauge rows, at
`max(1e-11 N, 1e-8 ||rhs||)`, and stress `rtol=1e-4, atol=1e-3 Pa`.

The streamed U/cavity tests manufacture independent constrained wrenches and
check five contacts with a two-contact block, unequal counts across three
environments, release/rebirth, nodal additivity, complete residuals and full
stress. A nonidentity link offset verifies authored/reference/world frames.
The saved first finger-contact fixture at tick 110 checks the relaxed internal
tolerance without changing final numerical acceptance.

## Actual slipping-contact precision diagnosis

The first every-step new-law run on B=4, seed 623001 failed at tick 110 with
the former `1e-9` internal tolerance. Six earlier CPU oracle captures passed.
Changing only internal stopping to `8e-9` accepts tick 110 in 14 evaluations,
with normalized force error `7.7807629104e-9` and moment error
`2.1015628889e-9`. Complete same-mesh checks for this saved fixture pass on
CPU and CUDA. The following actual tick 111 still fails: native force error
stagnates at about `1.279e-8`, including a 1,024-evaluation run. Loosening an
internal threshold within the final budget cannot repair this input.

Independent continuous-geometry analysis finds exactly three triangles
intersecting the Euclidean radius ball: faces 49, 51 and 69. That ball is a
superset of every intrinsic footprint of the same radius. An analytic polar
ray has negative support on all three inward Coulomb cones and a normalized
target-force margin above `1.27877e-8`, proving continuous force incompatibility
at the unchanged `1e-8` budget. This conclusion does not depend on quadrature.
Earlier inscribed/circumscribed sampled-cone LPs alone did not establish it.

The original URDF import routes collision meshes through MuJoCo's FP32 vertex
storage. The maximum reference/collision vertex difference is `2.52249 nm`;
face 49's inward normal changes by `3.87990e-8`. The saved forces are on that
collision facet's actual friction boundary (normalized cone margins within
`2.3e-16`), while outside the FP64 reference facet's cone. Neither clipping the
force, enlarging the radius nor changing friction is an admissible fix.

The new explicit `URDF(preserve_mesh_precision=True)` option keeps the source
mesh and its source pose together, retaining the authoritative parsed link
inertia, joints and collision parameters. It checks mesh correspondence
between the two existing URDF parses. The Franka example exposes this input as
`--egg-geometry mesh64`; its default `urdf` input remains available. Comparisons
must use the same geometry input for rigid-only and stress scopes. This is an
explicit change to the benchmark input, not stress feedback into dynamics;
the new baseline must record it. On the corrected input all 42 source vertices
match the existing FP64 reference exactly in the measured scene.

Native `kernel_certify_continuous_force` supplies the conservative continuous
force proof when fitting remains unresolved. Raw status 35 and the persisted
six-component polar ray/margin distinguish physical reference-model
incompatibility from numerical failure. It does not claim general continuous
six-dimensional feasibility. The saved tick-111 regression retains the
original incompatible input and checks that increasing actual friction in
another environment cannot inherit its failure. The tick-110 input is still
accepted within the existing final budget.

## Edge-source mapping diagnosis

The corrected FP64 scene passes tick 111 and then exposes two false source
ambiguities at tick 113. Each producer witness lies on a real shared mesh
edge; candidate projection separations are `1.03518e-16 m` and
`9.44787e-17 m`. The fixed dimensionless `32 epsilon` active-barycentric test
incorrectly treats tiny coordinates `7.64e-15` and `8.12e-15` as separate
nonshared vertices. The resolver now identifies a common edge/corner by mesh
vertex identity and measures distance to that feature in the existing
geometric tolerance. Separately indexed coincident sheets remain distinct.
It retains the actual mapped anchor and its offset moment.

The initial direct-Mesh trial correctly rejected attempts to change mass after
fixed stress operators had been built; it contributes no live acceptance.
An inertia audit initially compared MuJoCo's principal-frame tensor directly
to the authored tensor. The corrected offline audit rotates that tensor back
using its declared inertial quaternion; its tolerance remains `1e-16`.
These failed development outputs and their sources are retained.

The edge-feature change passes tick 113; tick 114 then exposes a projection
separation `1.65684e-16 m`, just above the local-only roundoff estimate
`1.59843e-16 m`. Its source arose from a world-to-body transformation near
world position 0.65 m. The resolver's geometric roundoff budget must include
that transformation's operand scale, not only the roughly 0.02 m local point.
This internal budget correction is still being validated. It does not change
the physical projection limit or final numerical acceptance.

## Closed evidence before the latest checks

| Run | Result | Scope |
|---|---|---|
| `packed-regression-v2` | 30 CPU / 30 CUDA pass, 210.60 / 234.25 s | Packed samples/geometry |
| `intrinsic-regression-v1` | 30 CPU / 30 CUDA pass, 227.45 / 252.70 s | Streamed integration/lifecycle |
| `intrinsic-regression-v2` | 32 CPU / 32 CUDA pass, 246.87 / 278.06 s | Adds tick-110 budget checks |
| `intrinsic-regression-v3-cpu` | 33 pass / 2 failed, 260.40 s | Continuous-force certificate passes; inertia-frame audit corrected afterward |
| `intrinsic-regression-v4` | 36 CPU / 36 CUDA pass, 267.60 / 328.78 s | Shared-feature and corrected inertia-frame checks; before/after source hashes match |
| `online-v1` | 110 completed steps | Every-step recovery, original URDF, internal `1e-9` |
| `online-v2` | 111 completed steps | Same input, internal `8e-9` |
| `online-mesh64-v2` | 113 completed steps | Exact source vertices; false edge ambiguity |
| `online-mesh64-v3` | 114 completed steps | Shared-feature fix; local-only roundoff scale |

The closed 640-step attempts remain partial trajectories; they provide no full lifecycle
acceptance. `dynamics-mesh64-v2` encounters the same tick-114 source failure
and supplies no complete dynamics-isolation acceptance. Subsequent validation
must compare every rigid state component against a matched rigid-only scene
through motion and resets. Pending runs provide no acceptance claim until
their logs close and source hashes match.

Hardware is RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, driver 595.71.05;
Genesis 1.4.3, Quadrants 1.3.3, container `1_26`, FP64. Compilation caches and
runtime outputs live under the established data directory. These are intrusive
correctness runs, not throughput measurements or RL training.

The immutable [online checkpoint manifest](../evidence/20261011-nonconvex/online-checkpoint-v1/manifest.json)
publishes 956 complete artifacts (11,000,161 bytes), including closed logs,
per-run source copies/hashes, saved contacts, oracle captures, independent
geometry/cone diagnoses and launchers. Every original and published artifact
passes SHA256 verification. Manifest SHA256 is
`dcc41597f3cf647e37ba11eb1f01087e54ed362aea029e81214fef73a1e0b470`.
Partial/failed trajectories remain labelled as such; no filtering of failed
environments contributes to an acceptance or rate claim.

## Reproduction and remaining work

Inside an allocated `rtx-mid` container `1_26`, from this checkout:

```bash
source research/rigid_stress/scripts/cluster_env.sh
RIGID_STRESS_RUN_DIR="$RIGID_STRESS_DATA_ROOT/runs/unique-cpu" bash research/rigid_stress/scripts/check_nonconvex_contact.sh cpu
RIGID_STRESS_RUN_DIR="$RIGID_STRESS_DATA_ROOT/runs/unique-cuda" bash research/rigid_stress/scripts/check_nonconvex_contact.sh cuda
"$RIGID_STRESS_ENV/bin/python" research/rigid_stress/probe_nonconvex_online.py --envs 4 --steps 640 --seed 623001 --contact-block-size 2 --egg-geometry mesh64 --output "$RIGID_STRESS_DATA_ROOT/runs/unique-online"
"$RIGID_STRESS_ENV/bin/python" research/rigid_stress/probe_nonconvex_dynamics.py --envs 4 --steps 640 --seed 623001 --egg-geometry mesh64 --output "$RIGID_STRESS_DATA_ROOT/runs/unique-dynamics"
```

Complete the every-step trajectory and broader seeds, source producers and
aggregate/decomposition mappings, justified integration/distance refinement,
partial reset/state restoration and isolation checks. Then rebuild and profile
matched rigid-only, recovery-only, live and policy scopes at B=1024,2048 and
larger batches. Local candidate work, device scheduling, failure compaction,
verified histories, shared operator application/scatter, layout and precision
remain applicable optimization work. No new-law rate or plateau is claimed.
