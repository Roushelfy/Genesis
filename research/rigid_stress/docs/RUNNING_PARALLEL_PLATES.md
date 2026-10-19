# Reproducing the parallel-plate comparison

Read the [results](PARALLEL_PLATE_RESULTS_20261018.md) and
[audit](PARALLEL_PLATE_AUDIT.md) before interpreting fields or rates.
The baseline is a full free hollow egg under common normal force. A passing
discrete equation check does not qualify its continuum contact field.

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
