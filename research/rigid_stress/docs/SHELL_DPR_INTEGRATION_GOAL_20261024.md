# Native thin-shell DPR integration and throughput goal

**Active user goal, 2026-10-24.** Turn the completed CPU thin-shell study into
an optional, usable backend of Genesis's existing rigid stress recovery.
Maximize useful throughput, including parallel environments. Keep the mature
automatic-radius + finite-pressure DPR pipeline and replace the elastic
recovery discretization when the shell backend is selected.

This supersedes the completed volume-DPR throughput task as the immediate
priority. Earlier pressure/enrichment, SCR-repair and high-resolution physical
ranking campaigns remain historical unless directly needed for this task.

## Starting point and scope

Read [the CPU study](../shell_cpu/README.md),
[algorithm and migration notes](../shell_cpu/ALGORITHM.md), and
[the delivered volume-DPR baseline](DPR_THROUGHPUT_20261023.md).

The CPU study supplies a curved linear shell, eccentric friction loading,
prefactorized recovery, exact compact response operators, batched transfer,
and gauge/scaling experiments. It is research evidence, not an existing
native shell backend or a GPU performance prediction.

Implement the feature inside the existing rigid solver, with production
numerical work in **Quadrants**, including native operator preparation and
runtime recovery. Use CPU research implementations as independent oracles.
Do not create a separate deformable dynamics simulator or a NumPy/SciPy/Warp
runtime alongside the native feature. Existing volume FEM remains available;
expose a clear pre-build backend choice and retain existing defaults until
measured evidence justifies a change.

Only reference shape, thickness and material are fixed. Preserve arbitrary
changing/asymmetric convex and nonconvex contact locations, normal and friction
forces, contact count and surface side. Contact may vary smoothly during a
grasp, but correctness must survive release/recontact, slot changes and resets.

## Preserve the load model and observations

The intended pipeline is:

```text
rigid contacts -> cheap automatic radius -> mature finite-pressure DPR
               -> selectable prefactorized volume or shell recovery
               -> maximum or complete stress output
```

Keep the native DPR support-ball pressure law, supplied wrenches, validity
checks and conservative transfer. Extend transfer to the shell's actual
physical surface and consistent force/rotation DOFs, retaining the eccentric
moment from tangential loads on inner/outer surfaces.

The main CPU prototype uses a projected-Hertz variant; it is not an exact port
of native DPR. Use the separate native-law CPU control when matching native
load semantics, and identify any geometry/discretization difference explicitly.
Do not silently substitute the projected law, enlarge the contact radius, move
application points or relax checks to obtain admission or speed.

Keep maximum/full modes configurable before running. Clearly name shell
outputs as in-plane membrane-plus-bending stress, with transverse resultants
separate. Preserve body/world frame semantics, numerical acceptance, batching,
environment isolation, update cadence and partial-reset/checkpoint behavior.
The shell recovery remains an observation of rigid motion; it does not feed
elastic displacement back into dynamics or contact.

## Immediate priority: low-resolution native performance

First deliver a coarse working path and focused CPU/native numerical
consistency. Then profile and optimize the actual native implementation.
Do not spend GPU time on high-resolution convergence campaigns before the
profiling and applicable throughput work are complete.

Use the full Franka grasp workflow with an egg-shaped hollow rigid object as
the main performance scene, including approach, closing, frictional loading,
motion and opening/release. Preserve any failed stages rather than filtering
them. Add small convex/nonconvex and asymmetric controls as needed for load
semantics and lifecycle coverage.

Produce a detailed stage profile: initialization/operator preparation,
contact/witness/curvature/radius work, pressure fit/integration/scatter,
recovery, stress evaluation/max reduction, launches, copies and synchronization.
Keep setup costs separate from repeated steps. Measure small batches and the
useful large-batch range; choose configurations from measured results.

Use the CPU findings as candidates, not rigid implementation prescriptions:

- Share fixed geometry, factors and response data across identical environments.
- Compare reused batched solves with compact exact surface response operators;
  account for construction, quadratic storage and active-column traffic.
- Preserve friction couples while exploiting compact surface-force mappings.
- Reuse and validate collision witnesses and continuous-contact work; pack
  exceptional/failed environments without hiding fallback or overflow costs.
- Batch/fuse transfer, recovery and output work; avoid host round trips and
  per-environment Python loops. Stream exact sampled maxima where worthwhile.
- Test distributed translation gauges, scaling, ordering/fill and appropriate
  FP32/mixed storage or arithmetic against full fields and residuals. The CPU
  study does not establish that FP64 is mandatory or that TF32 is accurate.
- Explore other profiling-supported optimizations and retain negative results.

Measure both module costs and matched complete-scene timings against pure
rigid and existing volume DPR, in maximum and full modes. Compare manual and
automatic radius where useful. Pin geometry, load law, precision, cadence and
control settings; do not multiply unrelated component speedups or present
CPU recovery timings as GPU/robot throughput. Report memory and useful batch
size together with environment-steps/s.

## Accuracy boundaries and deliverable

During integration/tuning, require numerical consistency with the corresponding
CPU shell model, conservative wrenches, rigid-mode invariance and complete-field
checks on focused fixtures. Keep discretization/model error separate from
implementation error; validate the native load law independently of recovery.

The CPU egg's roughly 45-micrometre Hertz footprint in a 0.5-mm wall remains
unresolved, and shell in-plane stress omits 3D near-contact normal stress.
Successful integration or numerical agreement cannot certify a physical global
3D maximum. Preserve that limitation and retain volume FEM for comparison.
Do not revive analytical particular-field/boundary-traction/combined-field
routes or make expensive contact-region solves part of this task.

After profiling and applicable throughput optimization, run targeted physical
checks at suitable scales, reusing analytical and independent FEM references
(and Code_Aster when available). Quantify useful global membrane/bending
accuracy separately from unresolved local contact peaks; avoid broad sweeps
without a specific unresolved question.

Deliver native code, focused regression coverage, a runnable Franka example,
recommended backend/output/precision/batch configurations, detailed measured
profiles and a concise completion report. State what is implemented, tested,
unsupported or unmeasured. Follow existing storage/reproduction policies and
preserve the volume baseline and CPU evidence.
