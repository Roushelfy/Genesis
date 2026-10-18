# Parallel-plate comparison progress

The active specification is [PARALLEL_PLATE_COMPARISON_GOAL.md](PARALLEL_PLATE_COMPARISON_GOAL.md),
pinned at remote commit `914a558df`. The superseded Franka campaigns remain
stopped and their accepted packets and unfinished source changes are preserved.

## Checkpoint

See [the measured diagnostic results](PARALLEL_PLATE_RESULTS_20261018.md).
The initial adapter/geometry failures have been retained and corrected in new
campaigns. The selected contact-entry control uses20um clearance and a3mm
in-plane plate translation, with actual initial geometry assertions. Native
settling uses2s holds. No physical accuracy ranking is admitted: the QIPC
contact field remains spatially unresolved, SCR fails cancellation-sensitive
final moment acceptance, and original Shell fails coupled equilibrium.

## Initial pilot contract (historical)

Use the original full normal-offset hollow egg, E=10GPa, nu=0.3,
density=2000kg/m3, no gravity, and 120mm square, 10mm thick parallel plates.
The lower plate is fixed. The upper plate has mass50g and one axial slide;
positive force closes the gap. Initial outer-surface clearance is22um at
each plate. No egg vertices are fixed. All methods receive the same force
history; plate separation is an observable.

Begin with a20-step empty-plate analytic check of backward-Euler mass/damping
response. The pinned original-main QIPC exposes its existing finite-stiffness
prismatic guide, not the separate integration fork's minimal-coordinate API.
Check actual plate lateral motion and orientation when qualifying that guide.
The first launch `empty-v1` failed before initialization because that unsupported
keyword was used; retain its log and source. `empty-v2` uses the original API.

The initial hollow-solid pilot uses the existing fixed-domain base1/surface1/
wall4 mesh, dt=1ms, loads0.1/0.3/1N, each ramped over0.1s and held0.2s,
with50Ns/m plate damping. These are proposed pre-failure levels; pilot failure
does not admit them. History continues between levels. Observe mechanics
every10steps and save complete spatial fields only at predeclared endpoints.
The pilot includes observation/IO and is excluded from primary cost.

Provisional endpoint gates are: both actual plate reactions within1% of
command (1mN force floor), plate speed below1um/s, egg inertial resultant
below1% of command, egg kinetic energy below1e-10J, and complete nodal
equation residual below1e-3 of contact norm. Require a stable settling window,
admitted solver exits and numerical/control sensitivity before calling any
endpoint qualified. A single passing endpoint is insufficient.

Initial QIPC Newton velocity tolerance is1e-8 and PCG tol_rate1e-4. Compare
bounded tighter Newton and PCG controls on the selected fields before long
jobs; the linear criterion is an inner preconditioned residual criterion,
not the final nonlinear accuracy. Preserve cap and contact failures.

## Native friction limitation

The unmodified native contact API requires friction>=0.01 and the collider
also clamps actual coefficients to0.01. Thus a zero friction ratio cannot
produce exact frictionless DPR/SCR mechanics. Keep that baseline intact and
label its actual coefficient. QIPC mu=0 and mu=0.01 controls must bound the
effect on the normal comparison; record actual tangential forces. Do not
claim an exact common-mu=0 comparison or silently patch the production solver.
The declared reconstruction radius is6mm and is not fitted to FEM fields.

## Execution and storage

New runtime root is `/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261018-parallel-plates/`.
All scientific execution uses rtx-mid Slurm and GENESIS_IMAGE_VER=1_26 gs-srun;
each campaign captures source, exact commands and private node-local caches.
See [DATA_LOCATION.md](../../../DATA_LOCATION.md) for the successful data probe
and preserved home fallback.
