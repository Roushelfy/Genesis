# Reproducing the parallel-plate comparison

`qualify_prescribed_load --data-root RUN_ROOT --output NEW_DATA_PATH` verifies
the independent C2 reference's fixed wall/surface/joint/integration targets after
the named field campaigns finish. `contact_fem_exact_columns
--decompose-three-fields --physical-volume` accepts candidate, same-mesh control
and fine-reference fields in that order, and directly integrates signed tensor
differences/cross terms. Run `check_column_decomposition` first. See the
[prescribed-reference report](PRESCRIBED_REFERENCE_QUALIFICATION_20261019.md)
for exact sources, passed controls and the remaining method comparisons.
`finite_area_shell --coarse-patches 12` selects the primary configuration's
preconditioner for a separately labeled static traction replay. Its default
remains zero patches; the allocation fix preserves exact load-map values.

`pad_absolute_errors --data-root RUN_ROOT --output NEW_DATA_PATH` rebuilds the
normal81-row absolute/relative table from original complete columns and rehashes
its actual fields and qualification inputs. Submit this CPU postprocessing with
`shell_comparison.submit --cpu-only --cpus 1 --memory-gb 16`; the same `rtx-mid`
partition and `GENESIS_IMAGE_VER=1_26 gs-srun` workflow remain required. Default
submission still requests a GPU. See the [dimensional-error report](NORMAL_ABSOLUTE_ERRORS_20261019.md)
for field/peak quadratures, units, exact receipts and scope. CPU-only common-column
comparisons can use the same submission option with their own memory budget.

For exact reconstructed native boundary loads, use `replay_native_pad_tractions
--run ORIGINAL_ADMITTED_RUN --levels LEVEL ... --output NEW_DATA_PATH` in the
native environment. Then run `audit_native_reconstruction --capture CAPTURE
--additional-captures OTHER ... --output NEW_DATA_PATH` in the independent FEM
environment. Both use Slurm and fresh data paths. The reader checks original
journal/frame identities, source replay, signed P2 scatter and unchanged native
force/stress arrays. The independent audit recomputes interpolation from geometry
and reuses one factor for matching meshes. See the
[reconstruction/material report](NATIVE_RECONSTRUCTION_MATERIAL_20261019.md)
for exact successful and failed tasks, measured plate-side redistribution and
the distinction between numerical-core and physical load-model accuracy.

For completed250um nonlinear mechanics and500-to250um surface controls, see the
[spatial checkpoint](FRICTION_SPATIAL250_20261019.md). `friction_plot_data
--data-root RUN_ROOT --reference-run ADMITTED_REFERENCE_RUN --output NEW_DATA_PATH`
prepares actual complete force histories and unsmoothed contact samples through
Slurm. Its own-history/data checks do not qualify reference resolution or native
every-step stress. Final rendering requires the matching fine spatial admission;
the executed500um reader control is preparatory only.

For the completed coarse nonlinear time/contact sampling controls, see the
[q9/q25 report](FRICTION_COMMON_CONTROLS_20261019.md).
`audit_stvk_time_contact --data-root RUN_ROOT --output NEW_DATA_PATH` rehashes
the actual endpoint fields and verifies their own histories/momentum checks,
control magnitudes and sampling differences. Execute it through Slurm after
the named field campaign completes; it does not qualify pending fine meshes.

Read the [results](PARALLEL_PLATE_RESULTS_20261018.md) and
[audit](PARALLEL_PLATE_AUDIT.md) before interpreting fields or rates.
The baseline is a full free hollow egg under common normal force. A passing
discrete equation check does not qualify its continuum contact field.

For the latest admitted normal case, read the
[normal-pad qualification](NORMAL_PAD_QUALIFICATION_20261019.md) and its
[exact checkpoint](../evidence/20261019-qualified-ranking/normal-pad-reference-v1/README.md).
`pad_normal_qualification --data-root RUN_ROOT` reconstructs the change gates,
complete-history checks, conditional ranges and CSV from those named receipts.
Use `contact_fem_exact_columns --physical-volume --common-peaks` for shared
tangent-plane peaks and separate full3D-only stress components; retain raw
peaks and independently check angular/depth and spatial resolution.

`pad_common_peak_qualification --data-root RUN_ROOT` rebuilds the separate
common-peak qualification. `pad_response_figure --data-root RUN_ROOT --style
PATH_TO_RETAINED_nature_mpl_style.py` plots clearance-corrected force/closure
and conditional peak errors. All commands require a fresh `--output` on data.

For primary whole-workflow timing, add `--primary-cost --cost-warm-levels 1`
with four repeated loads to the native/Shell drivers. Audit those exact runs
using `audit_pad_primary_cost --runs ... --data-root RUN_ROOT`. Batch drivers
`plate_native_pad_batch` and `plate_shell_pad_batch` additionally accept
`--envs 1/2/4/8` and optional independent `--env-load-scales`; use
`audit_pad_batch` on their actual histories and fields. Native batch pad
mechanics is serial per environment; its complete cost is included.
`pad_memory_profile --task DRIVER_TASK_JSON` runs one such driver in the same
process with intrusive memory sampling and explicitly non-primary timing.
The JSON contains `module` and `args`, without `--output`; audit its `run/`
subdirectory separately. Captured task files in the checkpoint specify every
parameter and engine source. Profile admission cannot replace primary admission.

For common-scale surface maps, add `--export-surface-fields` to
`contact_fem_exact_columns`. The retained representative0.1N map campaign uses
one centroid per fine reference face solely for display; retain the separate
9/25-point peak controls. `pad_spatial_figure --data-root RUN_ROOT --fields
SURFACE_EXPORT_CAMPAIGN --style STYLE_FILE --output NEW_DATA_PATH` reproduces
the complete outer/inner maps and actual upper/lower pressure samples. It
checks identical material positions, shared reference tensors, actual reactions
and inclusion of every force-bearing point in the view. `profile_p2_stvk
--mesh MESH --frame ADMITTED_FRAME --output NEW_DATA_PATH` provides the retained
intrusive nonlinear-force CPU profile; execute through the same Slurm workflow.

`plate_solid_pad --elasticity stvk --constitutive-correction direct
--cache-constitutive-evaluations` enables the separately validated exact-state
cache. `check_p2_stvk` tests in-place state invalidation and returned-array
isolation; the complete cached trajectory also has its own mechanical and
full-field equivalence controls. See the
[friction resolution/memory report](FRICTION_RESOLUTION_MEMORY_20261019.md)
for true250um Shell friction timestep/mesh controls and complete B1/B4 memory
receipts. Runtime outputs and logs remain on data.

The separately validated `--reuse-contact-velocity` option requires the modal
solver and avoids a duplicate factor solve by reusing its full compliance
response. Retain the complete physical equation/momentum and field-equivalence
gates; see the [control and friction cost report](FRICTION_KEYFRAME_COST_REUSE_20261019.md).
For the matched friction keyframe cost, captured tasks use five0.1N levels,
`--primary-cost --cost-warm-levels 2 --stress-stride 200 --stress-output full`.
Every-step native failures remain distinct configurations. `inspect_pad_rejection`
accepts an optional `--mesh` and `--radius` for a conservative source-cone
certificate, including immutable before-call inputs with unknown native status.

`pad_cost_matrix --data-root RUN_ROOT --batch-sizes 1 4 --output NEW_DATA_PATH`
joins the completed primary/memory cases only after verifying their exact
own-run audits, source receipts and physical/hardware configuration match.
`pad_phase_profile --task DRIVER_TASK_JSON --data-root RUN_ROOT --output
NEW_DATA_PATH` produces a separate synchronized phase profile and verifies
its own400-step history and complete endpoint field. See the
[complete cost report](COMPLETE_PAD_COST_20261019.md) for phase boundaries,
warm-repeat dispersion and the retained premature-dependency failures.

`plate_solid_pad --elasticity stvk --contact-solver modal --warm-contact`
selects the verified full-compliance modal iteration. It does not fix or
discard rigid or elastic modes. Reproduce `check_matrix_free_pad`, the
complete400-step dense/modal comparison and `audit_solid_pad` before new
refinement. `--quadrature-order 6/10` and `--dt` isolate integration and time
controls; their completed trajectories and fields must pass independently.
`--constitutive-correction direct` assembles the StVK nonlinear remainder
without cancellation of full nodal forces. The default historical subtracted
formula remains available; the direct control retains independent full-Piola
equation/moment acceptance and unchanged convergence thresholds.

`plate_native_pad` and `plate_shell_coupled` accept `--stress-stride 1/20`
and `--stress-output full/max`. `--cost-warm-levels 1` records segmented
preflight costs after the first load level. It does not produce primary
uninstrumented throughput. Run each method's own mechanical audit and then
`audit_pad_cost --run CAMPAIGN --data-root RUN_ROOT`; retain failures.
Shell's optional `--paired-output-check` compares the two observers on each
identical frozen state, outside the primary observation timing segment.

The later [native physical-pad checkpoint](../evidence/20261019-qualified-ranking/native-pad-history-v1/README.md)
uses `qipc_comparison.plate_native_pad` with its captured tasks, after the
[physical-pad declaration](COMPLIANT_PAD_DECLARATION.md). Select the existing
`Euler` semi-implicit integrator explicitly when reconstructing the variant.
`approximate_implicitfast` is the retained failing gap-prediction control.
`--method rigid/dpr/scr` chooses no observation or the corresponding independent
native observer; `--source-reduction wrench/none` distinguishes sparse wrench
compression from all actual centroid sources. Keep the radius, friction,
gravity ramp, mass and timestep recorded in each task. In particular, changing
friction while the actual shear is zero can still change SCR's reconstruction;
that is a parameter study, not a repeat.

Run `qipc_comparison.audit_native_pad --runs ... --compare-motion` on matched
histories to audit the source interface and absence of observation feedback.
Use `compare_native_pad_sources` to separate actual-force changes from sparse
packet changes, and `contact_fem_exact_columns --physical-volume` for complete
field differences. A passing endpoint or source-wrench check cannot override
a failed complete history or field-sensitivity gate. These research adapters
do not implement a new default native collider or a production external-load API.

For the joint refined independent reference, `compliant_plate_reference
--cvxopt-factor` selects the verified existing CVXOPT1.3.2 long-index CHOLMOD
interface. Do not combine it with `--cholespy-path`. The default remains
unchanged for historical reruns. First reproduce `check_cvxopt_factor` and the
captured coarse field control from the
[factor packet](../evidence/20261019-qualified-ranking/full-source-factor-control-v1/README.md).
The larger reference also needs its separately recorded memory allocation;
a successful small control is not proof of a completed large reference.

## Existing cluster paths and pinned inputs

Run submission commands from the task checkout; the helper captures analysis
source and launches scientific work through Slurm `rtx-mid` and
`GENESIS_IMAGE_VER=1_26 gs-srun`. The login host only handles submission/metadata.

```bash
cd /mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
plate_root=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261018-parallel-plates
plate_source=/mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery
plate_qipc=/mnt/home/zhaofeng/workspace/Genesis/.worktrees/qipc-stress-reference-roundoff
plate_shell=/mnt/home/zhaofeng/workspace/Genesis/.worktrees/kashu-shell-pr22-diagnostics
plate_native_python=/mnt/data/zhaofeng/venvs/genesis-shell-contact/bin/python
plate_qipc_python=/mnt/data/zhaofeng/venvs/qipc-owned-coarse-tree-check01-20261015/bin/python3.12
plate_analysis_python=/mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python
```

The QIPC original-source revision is
`fe252bd30acc6c717507b74add31512812f48761`; the captured patch and file inventory
are essential parts of the pin, not an assertion of a clean checkout. Its
inherited diagnostic patch changes the non-KKT line-search energy comparison
by16 double epsilons times abs(E0), without changing energy/derivatives/CCD.
All plate QIPC results use this labeled diagnostic, not unmodified upstream.
The patch SHA256 is
`c9ecba4791247396c4bad5cf2937a9bb23973e9182296cdcb4a94da0ee1c32ea`.
Its loaded native binary SHA256 is
`d4b90d5cbcbee44fb2c92b9406d5ca15f70e4dcda82469c5fb9865bc37526211`.
Each campaign's `source.json`, `source.patch`, `analysis-files.json`, captured
analysis directory and `start.json` record the actual source/observer/input
identities. Source overrides in mixed campaigns pin original Shell separately.
The diagnostic Shell checkout revision is
`fd051fe3f6ed58b52810846d4295d7b6fbb5e2ab`; its read-only instrumentation
patch is retained as`shear-native-shell-v1/extra-source0.patch` with the
matching file inventory. It is not a repaired contact/elastic solver.
Use those captured sources for exact historical replay; current source reruns
are new measurements and may enter a different QIPC iteration regime.

The base1/surface1/wall4 input is under
`20261018-qipc-reference/controls-v1/group0/inputs/`, SHA256
`40256c5711abb9e9e9f313adbb41944efb8511ae409538001a75d019e88728ea`.
The common surface2/wall4 metric domain SHA256 is
`9d0d4f99c71578bd3f6bb039f9c72a5abbf2a8159f0209b4b891a3bda2647998`.
The native elastic input SHA256 is
`3b2b90d7ee39562a296422dab1e5ee66e87265949ee38eefd27c729633a995a4`.
These existing large inputs/environments stay on data; do not redownload them
into home. Task JSON includes exact absolute paths, modules and arguments.

## Concrete reruns

Use fresh output names. The helper rejects existing destinations; change a
suffix for another rerun. It owns cache/log paths and removes private node-local
caches. No `--allow-home-output` is needed for these data-side destinations.
The helper explicitly disables Slurm requeue because it has no scientific
restart protocol. After preemption, preserve partial outputs and submit unchanged
tasks to a fresh destination. TERM/INT/HUP produce143/130/129 shell receipts;
always require terminal scheduler status, final results and the own-run audits,
not a returncode file alone. `shell_comparison.check_submission_signals --output
NEW_DATA_PATH` runs the stdlib/shell orchestration controls without scientific work.

```bash
python3 -m research.rigid_stress.shell_comparison.submit \
  --source "$plate_qipc" --python "$plate_qipc_python" \
  --tasks "$plate_root/reference-loads-v1/tasks.json" \
  --output "$plate_root/reproduce-reference-loads-v1" \
  --minutes 20 --concurrency 1 --node-local-cache

python3 -m research.rigid_stress.shell_comparison.submit \
  --source "$plate_source" --python "$plate_native_python" \
  --tasks "$plate_root/native-equilibria-v2/tasks.json" \
  --output "$plate_root/reproduce-native-equilibria-v1" \
  --minutes 20 --concurrency 2 --node-local-cache --keep-going

python3 -m research.rigid_stress.shell_comparison.submit \
  --source "$plate_shell" --python "$plate_native_python" \
  --tasks "$plate_root/shell-operator-v1/tasks.json" \
  --output "$plate_root/reproduce-shell-operator-v1" \
  --minutes 20 --concurrency 3 --node-local-cache --keep-going

python3 -m research.rigid_stress.shell_comparison.submit \
  --source "$plate_source" --python "$plate_analysis_python" \
  --tasks "$plate_root/closeout-analysis-v1/tasks.json" \
  --output "$plate_root/reproduce-closeout-analysis-v1" \
  --minutes 10 --concurrency 4 --node-local-cache
```

The last command reanalyzes the **original recorded runs** because those paths
are in its task JSON. To analyze new reruns, copy that small task JSON to a
fresh data-side `meta/` file and replace its input paths explicitly. Do not
overwrite historical jobs or edit a running worker's captured source.

For held-contact cost, reuse `native-cost-v1/tasks.json` with the native
checkout/environment and `reference-cost-v1/tasks.json` with QIPC. Intrusive
`native-profile-v1`/`reference-profile-v1` are separate from primary cost.
`native-stability-v1`, `shear-reference-v1`, `shear-native-shell-v1` and
`contact-band-v1` task lists reproduce the bounded controls; mixed shear tasks
retain their explicit Shell source override. Empty-test task lists must use
their corresponding backend. No broad Franka campaign is part of this recipe.

Derived analysis uses `plate_audit` for mechanical windows/CPU same-load checks,
`plate_fields` for common material fields, `plate_failure` for rejected native
state, and `plate_summary` for cost/profile/mass/stability. Figures use
`closeout-figures-v1/tasks.json` in the analysis environment. All commands and
scalar/history/field schemas are inspectable in `qipc_comparison/plate*.py`.

## Interpretation and reproducibility bounds

Inspect completion, actual reactions, kinetic/inertial measures, observed
equation residuals, PCG/Newton cap histories and intersections before using
an output. Missing SCR fields are expected retained rejections. Shell's finite
completion does not pass the common-force equilibrium gate. QIPC cap-sensitive
reruns can differ materially in cost and equation admission despite matching
configuration; retain every run and do not select only a faster passing subset.

The original raw fields and continuous journals remain on data. The compact
[publication packet](../evidence/20261018-parallel-plates/closeout-v1/README.md)
contains small receipts, exact task lists, failure arrays and editable figures,
plus a SHA256 manifest linking external full fields and source inventories.
The old home fallback remains in place under its previously authorized bound;
this workflow neither migrates it nor writes new runtime outputs there.

## Exact material-column metrics and the declared polar-land control

The additional geometry and its limitations are recorded in
[qualified-stage progress](QUALIFIED_STRESS_PROGRESS_20261019.md).
It has2mm flat polar regions, a C2 height transition ending at4mm, and500um
normal-offset wall. Its contact support must independently pass qualification.
The original smooth geometry remains a separate benchmark.

```bash
qualified_root=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking
python3 -m research.rigid_stress.shell_comparison.submit \
  --source "$plate_source" --python "$plate_analysis_python" \
  --tasks "$qualified_root/meta/land-mesh-tasks.json" \
  --output "$qualified_root/reproduce-land-mesh-v1" \
  --minutes 5 --concurrency 1 --node-local-cache

python3 -m research.rigid_stress.shell_comparison.submit \
  --source "$plate_source" \
  --python /mnt/data/zhaofeng/venvs/rigid-stress-recovery/bin/python \
  --tasks "$qualified_root/meta/contact-active-set-tasks.json" \
  --output "$qualified_root/reproduce-contact-active-set-v1" \
  --minutes 8 --concurrency 2 --node-local-cache

python3 -m research.rigid_stress.shell_comparison.submit \
  --source "$plate_source" --python "$plate_analysis_python" \
  --tasks "$qualified_root/meta/contact-active-set-fields-tasks.json" \
  --output "$qualified_root/reproduce-contact-active-set-fields-v1" \
  --minutes 5 --concurrency 1 --node-local-cache
```

The first command generates new assets. The other task lists intentionally
reuse the recorded input meshes/operators and field runs; change their paths
explicitly when analyzing new results. `contact_fem_exact_columns` integrates
P1 stresses across the union of tetrahedron depth partitions. It keeps angular
quadrature and geometric refinement uncertainty separate and retains all raw
peaks. Optional stored material directions support the declared land geometry.
`plate_contact_fem` records SLSQP and active-set status separately and admits
only the final primal/dual/complementarity and full physical equation checks.


## Coupled Shell linear and contact-iteration controls

The frozen first-linear diagnostic is deliberately truncated and cannot supply
an admitted trajectory. Starting at the external optional-coarse repair
`a86ba81d06d16dcea7bba6904a617c8140b83958`, apply
[`shell_first_linear_capture.patch`](../shell_comparison/patches/shell_first_linear_capture.patch)
in an isolated checkout. `plate_shell_coupled --capture-first-linear` checks
its exact solver hash. Then run `audit_shell_linear --field <capture>/frozen-linear.npz
--output <fresh-data-output>` to independently assemble and solve that system.
The complete commands and source receipts remain in
`20261019-qualified-ranking/shell-linear-capture-v1/` and
`shell-linear-audit-v1/` under the established run root.

The successful land mechanical control uses the complete optional-coarse
repair, **without** the truncation patch:

```bash
python -m research.rigid_stress.qipc_comparison.plate_shell_coupled \
  --mesh "$ranking_root/land-mesh-v2/group0/meshes/land-contact-0.npz" \
  --loads 0.1 --clearance 0 --plate-offset-x 0.003 \
  --coarse-patches 0 --floating-sums --tolerance 1e-6 \
  --velocity-tolerance 1e-10 --iterations 8000 \
  --ramp 0.05 --hold 0.15 --record-stride 1 --output "$fresh_run"
```

Set `ranking_root` and `fresh_run` explicitly under data and execute through
the established submission helper/Slurm workflow. Floating sums are an explicit
setting, with repeat dispersion measured separately. `audit_coupled_shell`
checks50ms mechanical windows; `compare_coupled_shell` compares complete
same-mesh fields for timestep/repeat/history sensitivity, with0.5% field and1%
raw-peak targets. Passing these checks does not qualify a continuum reference
or complete the three-method ranking.


## Quadratic contact reference controls

`validate_p2_contact` checks independent constant-stress/pure-bending solutions,
quadratic face-gap extrema and P2 prism-column interpolation. Run it first in
the existing physical-reference environment through Slurm. `plate_contact_fem
--element-order 2 --continuous-quadratic-contact` then uses quadratic elasticity
and exact surface-gap constraint generation. Omitting both options preserves
the P1 reference. `--column-ordering` uses the existing pymetis dependency;
`--contact-operator` accepts only a matching mesh hash and element order.
The retained `contact-p2-pilot-v1` nodal run and
`contact-p2-continuous-v1` rank-deficient run are failed controls, not references.

`audit_quadratic_contact --mesh <mesh.npz> --field <load-0.1.npz> --load 0.1
--output <fresh-data-output>` verifies dual point-force transfer and exact
whole-triangle gaps. Consistent P2 nodal force signs must not be interpreted as
physical pressure support. `contact_fem_exact_columns` accepts P1 constant
or P2 corner-linear tensors and integrates through the same declared material
columns; stress values at all four tetrahedron corners retain unsmoothed P2
extrema. All commands require explicit data-side output paths and the existing
Slurm submission wrapper. The current task lists, sources, logs and fields are
under `20261019-qualified-ranking/contact-p2-*`.


For a failed quadratic active-set solve, first run the inexpensive
`quadratic_contact --mesh <mesh.npz> --field <retained-load.npz> --load 0.1
--output <fresh-data-output>` replay. It verifies the retained operator's mesh
and order, then diagnoses constraint generation without repeating the full
elastic factorization. A passing condensed replay must be followed by
`plate_contact_fem --contact-operator <retained-contact-operator.npz>` with the
same mesh/order and continuous-contact option; only this full solve checks
elastic equations and gaps across the entire outer surface.


For isolated wall resolution, `prepare_contact_reference --cases 5e-5:0.002:4
--wall-depths 0 0.05 0.15 0.4 1` keeps the outer surface and grades the500um
wall. The depth array must contain exactly layers+1 strictly increasing values
with endpoints0/1. Omitting it preserves uniform layering. Run
`validate_p2_contact` first; `contact_fem_exact_columns` reads the stored depth
array and integrates each actual layer rather than assuming uniform spacing.
Older mesh files without this metadata retain uniform-layer interpretation.

A cached `--contact-operator` may cover a superset of the requested candidate
nodes. Mesh/order and all selected node/sign/rigid/gap entries must match
exactly. A reduced search window still requires the full elastic rerun and
whole-boundary gap admission; it must not be interpreted as an imposed contact
radius or tuned against reference stress errors.


For the declared0.1N land case, paired1ms and0.125ms controls establish that
`--tolerance 1e-4 --velocity-tolerance 1e-10` matches1e-6 complete fields to
within5e-8 relatively. Use the1e-4 setting for the next timestep controls;
retain1e-6 in commands reproducing the original baseline. This result does not
relax the mechanical or complete-field gates and does not admit the current
timestep-sensitive field for a physical ranking. `compare_coupled_shell` also
saves complete per-face squared differences and reference densities for later
maps; regional summaries are diagnostic only and never exclude faces.


## Independent finite-area load control

`finite_area_reference --mesh <land-mesh.npz> --radius 0.002 --loads 0.1
--quadrature-orders 8 10 --output <fresh-data-output>` integrates the prescribed
C2 axial pressure profile with independent P2 facet functions. It saves full
fields under `order-8/` and `order-10/`, including quadrature positions, physical
point forces, area weights and traction densities. Run `validate_p2_contact`
first. Use `contact_fem_exact_columns` on those directories for quadrature,
wall and surface sensitivity. This prescribed-traction control does not solve
plate contact and cannot close the end-to-end goal by itself.

`audit_contact_qp --field <retained-contact-load.npz> --load 0.1
--solver-path <isolated-clarabel-0.11.1-install> --output <fresh-data-output>`
checks a retained contact QP independently. It requires pinned Clarabel0.11.1,
installed without modifying active environments, and uses the existing
physical-reference NumPy/SciPy environment. Its positive certificate covers
only the retained finite constraint set; inspect exact candidate-face gaps and
perform a full elastic/global-boundary rerun before physical admission.
`AlmostSolved` is retained as a failed certificate. Execute both modules through
the established Slurm/gs-srun wrapper with explicit data-side outputs.

Use `audit_contact_qp --form whitened-primal --constraint-rounds 20` for the
equivalent primal energy formulation and exact candidate-face constraint
generation. Its result separately certifies each finite QP and the candidate
face gaps. Feed a passing run directory to `plate_contact_fem
--contact-certificate <certificate-directory> --contact-operator <same-operator>`
with the same mesh, load, P2 order and continuous-contact option. This performs
a fresh full elastic solve and checks every outer face, including those outside
the candidate set. Retained operator, source-field and load identities must
match. Original active-set defaults remain available.

For the prescribed finite-area comparison, `prepare_native_mesh --mesh <mesh>
--output <assets>` exports the exact solid and collider. `finite_area_native
--assets <assets> --reference <independent-run/order-10> --method pcg` runs the
production core, DPR and SCR on that mesh with radius fixed in advance. Its
direct-method pilot exceeded the construction budget on4498 outer vertices;
the existing PCG path passes. `finite_area_shell --mesh <land-mesh>
--field <independent-load.npz>` runs explicitly labeled static relaxation and
checks actual elastic forces. It uses the pinned optional-coarse Shell repair,
floating sums, relative1e-4 and absolute velocity1e-10 settings.

`contact_fem_exact_columns --meshes <mesh-a> <mesh-b> --runs <run-a> <run-b>`
accepts either per-load directories or explicit NPZ fields. Mark Shell positions
in that list with `--shell-indices 0` and use its `frames/level0.npz`. The last
mesh defines the common material surface. `--surface-quadrature 1|3|9|25`
controls physical area integration separately from exact through-wall integration.
Run `validate_p2_contact` before using a changed interpolation/quadrature path.

`finite_area_reference --through-wall-ordering` changes only matrix permutation.
The optional `--cholespy-path <isolated-cholespy-2.2.0-directory>` switches the
independent reference to double-precision CPU CHOLMOD; pass the same option to
`validate_p2_contact` and verify a same-mesh full-field control first. This
optional research dependency does not change native Quadrants recovery or
active environments. Installation receipts, exact task lists and failed
factorizations remain under the qualified-ranking data root.

Use `contact_fem_exact_columns --physical-volume` for the complete stress
tensor metric weighted by actual reference tetrahedron volume. The final mesh
must be a solid; angular quadrature and through-wall tetrahedron partitions
remain explicit. The comparator checks its integrated volume against the full
tetrahedron sum. Membrane/bending resultants retain the declared normal500um
material coordinate and physical midpoint area measure.

`plate_contact_fem --element-order 2 --continuous-quadratic-contact
--qp-solver-path <isolated-clarabel-0.11.1-install> --qp-rounds 40` directly
solves hard contact from the original candidate constraints with the certified
primal formulation. Run `validate_p2_contact --contact-solver-path <install>`
first. A passing discrete contact state still needs spatial convergence and
an actual finite-area footprint; the hard-land pilot concentrates force near
a ring despite passing equilibrium.

For the separately [declared physical pad](COMPLIANT_PAD_DECLARATION.md), run
`compliant_plate_reference --mesh <land-mesh> --qp-solver-path <install>
--pad-stiffness 1e9 --candidate-radius 0.003 --loads 0.1
--quadrature-orders 4 6 --output <fresh-data-output>` through the same Slurm
wrapper. The optional `--cholespy-path` uses the already validated factor.
Pad stiffness is force per area per compression in N/m3. It is held fixed while
mesh, quadrature and timestep change. Full fields, positive quadrature forces,
actual pad gaps, projected areas, forces/moments, elastic and constitutive
residuals are retained under each `order-N/`. An excluded boundary with
compression fails the search-region check. This compression-only spring model
has its own admission checks and must not be labeled a hard-contact result.

`finite_area_shell --mesh <same-pad-solid-mesh> --field <pad/order-10/load-0.1.npz>
--solve-pad-contact` recomputes pressure and free rigid/plate coordinates from
the current Shell displacement. Run `validate_p2_contact` controls first.
The source field supplies shared quadrature geometry, areas and declared
stiffness/load; the pad solve does not replay its pressures. Exact source mesh
identity is required for this control. The result includes a separate
`static_pad_equations_admitted` check and `pad-contact.npz` with actual pressure,
gap, plate coordinates and complete affine envelope gaps. Omitting the flag
retains the prescribed-traction elasticity replay. Both are explicitly static
relaxation methods and require separate dynamic history qualification.

For continuous physical-pad Shell runs, apply
[`shell_physical_pad.patch`](../shell_comparison/patches/shell_physical_pad.patch)
on isolated source `a86ba81d06d16dcea7bba6904a617c8140b83958` (the resulting
committed variant is `41b2a57be`). Run `prepare_shell_pad --mesh <land-mesh>
--output <geometry-output>` through Slurm. Then run `plate_shell_coupled
--mesh <same-mesh> --pad-geometry <geometry-output/pad-geometry.npz>
--loads 0.1 0.3 1 0.3 0.1 --dt 0.001 --ramp 0.05 --hold 0.15
--record-stride 1 --tolerance 1e-4 --velocity-tolerance 1e-10
--iterations 8000 --coarse-patches 0 --floating-sums --output <fresh-output>`
using that isolated engine. Run `audit_coupled_shell --runs <fresh-output>`
to check each equilibrium window, including actual pad pressure and transpose
force transfer. This history never clears velocity. Existing diagnostic
half-thickness envelope gaps are distinct from physical pad compression;
use `physical_pad` and the saved per-face pad gaps for the pad law.

Use `compliant_plate_reference --fixed-point` to avoid forming the dense
normal-contact compliance matrix for this soft pad. The method repeatedly
applies the complete free-body elastic factor and solves the four free
rigid/plate coordinates. The force-change criterion is1e-10; an iteration cap
is a failure. Fresh full elastic and pad-law checks remain mandatory. On the
coarse mesh, independent Clarabel comparisons at orders0/6/10 agree in
complete stress to about5e-11 (`727876`). This verifies the solver, not mesh
accuracy or universal fixed-point convergence. The default condensed QP
remains available as an independent control.

`--quadrature-orders 0` means one explicit face centroid. Order1 from the
quadrature library actually has three points per triangle. To isolate
continuous/static Shell error, use the order0 reference field with
`finite_area_shell --solve-pad-contact`, then compare its frame against
the continuous frame using `compare_coupled_shell`. High-order reference
fields separately measure centroid quadrature error; do not attribute that
difference to timestep or nonlinear convergence.


### Independent transient solid and radius controls

The research `qipc_comparison.plate_solid_pad` entrypoint advances the full
P2 consistent mass and free rigid modes. It uses the existing data-side
physical environment and CVXOPT1.3.2 long-index factor. Run it only through
the established rtx-mid / `GENESIS_IMAGE_VER=1_26` submission workflow, with
all output paths explicitly under the project data run directory. Source,
mesh and dependency receipts are retained by `shell_comparison.submit`.

The common two-window friction arguments are:

```text
--mesh <coarse-land-mesh.npz> --loads 0.1 0.1 --dt 0.001
--ramp 0.05 --hold 0.15 --record-stride 1 --friction 0.3 --gravity-x 1
```

The original `--elasticity linear --contact-solver condensed` trajectory
passes its affine force/contact equations but fails the actual current-position
moment target in shear. Preserve it as a diagnostic, not a qualified friction
reference. The new explicit `--elasticity stvk` option iterates objective
elastic forces with the same mass and pad law. Its complete history and
current-position moment must pass independently; analytic controls alone do
not qualify it. `audit_solid_pad --run <run>` records both material-frame and
actual-position moment balances. The nonlinear frames contain exact P2 corner
gradients and a mass-weighted material rotation. Common-column comparisons
re-evaluate Cauchy stress at each sample; use `--depth-quadrature 2/4/8` for
integration sensitivity instead of assuming nonlinear stress is affine.

`--contact-solver matrix-free` uses contact-impulse Newton/PCG applications
through the full dynamic factor. It supports redundant high-order contact
quadrature without inverting the contact compliance. This alternative must
match the dense solver on admitted full histories before fine-mesh use.
`check_condensed_pad`, `check_matrix_free_pad`, `check_p2_stvk` and the existing
`validate_p2_contact` controls run in the same allocated physical environment.
Failure source captures, logs and receipts remain immutable.

For full-source native recovery,2mm remains the primary radius. The declared
1.5/2.5mm perturbations use `plate_native_pad --source-reduction none --radius`
and retain the original method/material/tolerance settings. Runs728155 and
728160 measure parameter effects over normal0.1/0.3/1N and nonzero sustained
shear; these effects are not solver uncertainty and do not authorize fitting
radius to a reference field. Full fields and actual source packets remain
under `native-pad-radius-v1` and `native-pad-radius-fields-v1` on data.


## Separate native production performance and terminal C2 controls

Follow [the production pilot](PRODUCTION_THROUGHPUT_PILOT_20261019.md) for the
new fixed-horizon B1/B8/B32 paired native scene tasks and source-capture guards.
Its default collider/error scope differs from this pad research workflow.
The [C2 method report](PRESCRIBED_METHOD_RANKING_20261019.md) records the completed
27-row errors and direct tensor decomposition, with exact task recipes.
The [three-wall friction checkpoint](FRICTION_WALL250_20261019.md) records
completed250um controls and the exact125um time/contact comparison prerequisites.


## Native scene conformance and resident-history module timings

See [the measured production/module report](NATIVE_PRODUCTION_PERFORMANCE_20261019.md)
for terminal results, retained strict motion failures and exact captured tasks.
The separate workers are native_production_controls (intrusive trajectories and
optional device-resident replay), production_throughput_audit (fixed-horizon
receipt audit), audit_native_production_controls (complete motion/output and
condition-matched batches), native_module_replay_audit (source-history and
module-time checks), native_motion_diagnosis and check_native_controller_inputs.
Use the existing Slurm runner and recorded data-side tasks; CPU-only analysis
keeps --cpu-only, while GPU diagnostics use their pinned source. The optional
--kernel-profiler flag belongs only to separate profiling allocations and is
recorded in submission.json; it is not enabled for primary throughput.


For native production consistency, use `qipc_comparison.native_production_controls`
with the exact pinned condition bank, followed by
`qipc_comparison.audit_native_production_controls`. `--group-indices` maps
separately captured batch groups; nonstandard batches require explicit groups.
The larger-batch cross-check assumes the same repeated frozen condition bank.
See the [MPR contact report](NATIVE_CONTACT_CONSISTENCY_20261019.md) for
349 passed B1/B8/B32 checks, source1ff39b077 and retained original failures.
`production_throughput_audit --batches ...` checks primary fixed-horizon timing
receipts independently; it does not infer complete motion or total GPU memory.

The [250um Shell separation report](SHELL_PAD_CONTROL_SEPARATION_20261019.md)
uses `finite_area_shell --inertia-relief` for explicit mass-distributed body
corrections while preserving point loads and the original force/static gates.
This opt-in static relaxation is separate from physical continuous dynamics.
`contact_fem_exact_columns --decomposition-indices i j k` selects three fields
for a signed difference decomposition while retaining the final physical solid
as the external volume carrier. The selected third field still sets the
stress normalization; carrier geometry never supplies a replacement tensor.

`native_production_memory --task EXACT_BENCHMARK_TASK_JSON --output NEW_DATA_DIR`
samples NVIDIA total process memory in a separate fresh benchmark process,
including build and warmup. Its nested timing is explicitly intrusive and
cannot replace primary rates; compare its own runtime counters and separately
qualified motion/output configuration. Preserve sample interval limitations.
