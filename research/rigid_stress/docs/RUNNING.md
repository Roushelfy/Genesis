# Reproducing the device implementation

> Historical prototype commands, 2026-10-10. The new feature and workflow are
> specified in [NATIVE_QUADRANTS_PLAN.md](NATIVE_QUADRANTS_PLAN.md) and `../GOAL.md`.
> Use low-resolution numerical checks and detailed performance iteration first.
> Do not run the fine-mesh replay, level-6 presets or physical convergence below
> during this phase. These commands reproduce the older external-library path;
> the native Quadrants feature should provide its own normal Genesis entrypoints.

Run numerical work on allocated hardware. The measured cluster image is
`genesis:1_26`, Python 3.10.12; the device is NVIDIA RTX PRO 6000 Blackwell
Server Edition with 95 GiB VRAM. This is different from RTX 6000 Ada.
Use the repository root as the working directory. Source, configuration and
small publication artifacts stay in the worktree; runtime outputs and caches
use the explicit data directory.

## Setup and checks

Within an allocated container, set paths before running any setup or workload:

```bash
export RIGID_STRESS_DATA_ROOT=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery
export RIGID_STRESS_ENV=/mnt/data/zhaofeng/venvs/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
bash research/rigid_stress/scripts/cluster_setup.sh
```

The setup reuses the data-side environment and pip cache. It installs optional
native cuDSS 0.8/CVXOPT dependencies in that environment. The reusable native
binding also accepts `CUDSS_LIBRARY_PATH` for an explicitly installed library.
It checks the public library version before creating a factor.

One command checks CPU math, device math/mapping, independent history,
native direct/graph recovery, one/four-substep observations and actual Panda
nominal/32-seed trajectories:

```bash
"$RIGID_STRESS_ENV/bin/python" -m research.rigid_stress.acceptance_suite \
    --gpu --native --grasp --output "$RIGID_STRESS_DATA_ROOT/runs/acceptance"
```

The suite stops at the first failed subprocess and records its exact command,
return code, wall duration, JSON artifact and combined log. Its actual Panda
checks use level 2 for integration and CPU comparisons; physical convergence
is a separate expensive option. Add `--physical` to include the complete
levels 4/5/6 and quadratures 6/10/16 for prescribed raw-point and force-line
finite patches. Use at least 128 GiB host memory for that option. To run only
the CPU mathematical checks, omit the GPU/native/grasp flags; those checks
do not require Genesis, Torch, CuPy or cuDSS.

The engine contact-normal regression has its own native pytest command:

```bash
"$RIGID_STRESS_ENV/bin/python" -m pytest -q tests/rigid/test_contact_normal.py --backend gpu
```

Use `--backend cpu` for its CPU counterpart. Neither test changes the friction
cone or clips a contact force.

## Real fine-shell visualization

This records the ordinary rigid Panda and egg, with auxiliary complete FP64
stress recovery. Setup/JIT, rendering and recording are outside throughput
claims. The nominal scripted controller completes an actual lift and hold.

```bash
"$RIGID_STRESS_ENV/bin/python" -m research.rigid_stress.check_panda \
    --level 6 --envs 1 --steps 600 --nominal --synchronous --record-only \
    --factor-backend cudss --inertia quadratic --body-products fused \
    --video "$RIGID_STRESS_DATA_ROOT/runs/demo/panda-egg.mp4" \
    --output "$RIGID_STRESS_DATA_ROOT/runs/demo/rollout.json"
```

`--record-only` records actual loads and diagnostics without solving the CPU
oracle every frame. The CPU validation commands below use a separate explicit
CPU factor. The committed nominal video is in
[`evidence/20261009-performance/visualization`](../evidence/20261009-performance/visualization/).

## Timing and independent same-RHS validation

Every timing command requires at least three ten-second repeats. Fine runs
take much longer: the minimum includes complete independently delayed grasps
and resets, with a full grasp warmup. Submit a sufficiently long allocation
before starting. Setup/analysis/factor/JIT/graph capture are reported separately.

The measured B=32 graph calibration example is runnable as one command:

```bash
"$RIGID_STRESS_ENV/bin/python" -m research.rigid_stress.benchmark_live \
    --level 6 --envs 32 --cpu-factor none --factor-backend cudss \
    --inertia quadratic --body-products fused --sparse-layout F --recovery-graph \
    --scope live --seed 510000 \
    --output "$RIGID_STRESS_DATA_ROOT/runs/calibration/live32.json"
```

Change `--scope` to `rigid`, `policy-rigid` or `policy` for the explicit live
scopes. Rigid scopes retain the same scene/collision/solver settings. The
fixed FP64 MLP is 26->64 tanh->64 tanh->7 tanh, with residual action amplitude
1e-4 rad. Policy observations/actions remain on GPU. This measures inference
and rollouts; it is not PPO training.

The benchmark's `warmup_quality` audits the actual complete warmup trajectory
and publishes contacts, radii/friction, tangential forces and the stated
lift/hold success criterion. Its diagnostic work is excluded from steady-state
timing. Every requested stress substep and all reset/policy/recovery work stay
inside the timed scope. Contact quality counts use each scene step's final
contact snapshot; stress observations retain the maximum over all substeps.

For saved fine contacts, independently compare the complete device RHS with
CPU mapping/body assembly and same-mesh FP64 CPU direct peaks:

The committed actual fine held-out input is
`research/rigid_stress/evidence/20261010-final/inputs/heldout32.contacts.npz`.
Its companion `heldout32.json` contains the exact phase/reset records. The
archive preserves 68,928 actual contacts from 1,200 frames and 32 independent
seeds; its manifest hashes both files. Use this input directly or create a
new actual rollout with `check_panda`.

```bash
"$RIGID_STRESS_ENV/bin/python" -m research.rigid_stress.benchmark_replay \
    --level 6 --envs 32 --cpu-factor cholmod --factor-backend cudss \
    --inertia quadratic --body-products fused --validation-only \
    --verify-frames 60 --verify-stride 20 --verify-mapping \
    --save-rhs "$RIGID_STRESS_DATA_ROOT/runs/oracle/rhs" \
    --replay research/rigid_stress/evidence/20261010-final/inputs/heldout32.contacts.npz \
    --output "$RIGID_STRESS_DATA_ROOT/runs/oracle/strict.json"
```

The complete NPZ RHS arrays are large and belong in the data directory. This
is untimed validation; it does not claim CPU solves or transfers inside a
GPU throughput loop. Sixty sampled frames are sixty CPU comparisons, not
1,200 CPU comparisons. Graph replay warmup can separately compare every frame
against eager FP64 direct recovery of the identical complete device RHS.
The graph warmup invokes the ungraphed direct baseline with its existing
immutable operators and shared factor. It releases each temporary eager
displacement before the next mapped load. A second complete validation
operator set would exceed memory at the measured B=256 live point; neither
this sharing nor temporary release changes any timed recovery.
The replay also releases its previous complete RHS before mapping the next
frame, retaining the current RHS for same-input verification. This matches
the live callback lifetime and avoids retaining an extra 5.03 GB at B=256.

`benchmark_suite` reads a frozen JSON `benchmark_arguments` list and runs
rigid, matched policy-rigid, stress replay, live and policy scopes. Each scope
uses its own subprocess and artifact directory. `--scope` runs one scope in
its own allocation; the default `all` runs them sequentially on one GPU.
The command manifest records the configuration SHA256 and literal arguments.

The checked-in selected full level-6 configuration is
[`configs/blackwell_level6.json`](../configs/blackwell_level6.json): B=256,
C layout, strict FP64 native graph recovery. Run each scope in a separate
GPU allocation with eight CPUs, 128 GiB host memory and six hours, for example:

```bash
"$RIGID_STRESS_ENV/bin/python" -m research.rigid_stress.benchmark_suite \
    --config research/rigid_stress/configs/blackwell_level6.json \
    --scope live --seed 610000 \
    --output "$RIGID_STRESS_DATA_ROOT/runs/final-reproduction"
```

Use `--scope rigid`, `policy-rigid`, or `policy` for the other live scopes.
For `--scope stress`, also supply
`--replay research/rigid_stress/evidence/20261010-final/inputs/heldout32.contacts.npz`.
The default sequential `all` needs the sum of scope durations in its allocation.
The final stress measurement adds `--stages` through the replay CLI, recording
a separate full 1,200-frame stage pass and 100 CUDA-event samples; its exact
literal command is retained in the final launch manifest.

The fine model has 2,457,630 DOFs; exported CPU factors may exceed the public
export API's limits. Fine GPU timings use native cuDSS and `--cpu-factor none`
to skip an unused offline CPU factor. CPU oracle checks use `cholmod`, and
reject requests for an absent factor. The original sparse-mass, dense body
products, eager direct, exported SpSM and temporal paths remain selectable
for their respective reproducible checks and ablations.
