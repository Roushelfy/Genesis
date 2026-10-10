# Reproducing the device implementation

## Current native low-resolution workflow

The normal example and benchmark use only the engine feature. From the
repository root with Genesis and its regular dependencies installed:

```bash
python examples/rigid/franka_egg_stress.py --envs 8 --steps 600
python examples/rigid/franka_egg_stress.py --envs 8 --steps 600 --output-mode full
python -m pytest tests/rigid/test_stress.py --backend gpu -q
QD_KERNEL_PROFILER=1 python examples/speed_benchmark/rigid_stress.py \
    --scope profile --envs 1024 --varied --steps 100 --warmup 900 \
    --output "$RIGID_STRESS_DATA_ROOT/runs/native/profile.json"
python examples/speed_benchmark/rigid_stress.py \
    --scope live --envs 1024 --varied --warmup 900 \
    --steps 1200 --repetitions 3 --minimum-seconds 10 \
    --output "$RIGID_STRESS_DATA_ROOT/runs/native/live.json"
```

Set output and Genesis/Quadrants cache directories explicitly before running.
Choose `rigid`, `recovery`, `live`, `policy-rigid` or `policy` in separate
processes for their matched timing scopes. The current iteration mesh is
full-shell level 1, 2,430 DOFs. These are numerical and
performance-development meshes, without a physical convergence claim.
Default throughput measurements use at least 1,200 steps, three repeats and
ten seconds per repeat. Kernel profiling is a separate intrusive pass.

`--output-mode full` additionally writes all tetrahedral corner tensors and
von Mises values; the default `max` allocates no complete field. Both modes
retain the same complete load and residual budgets. `--unfused-pipeline`
compares separate native passes. The policy benchmark defaults to FP32
inference, with identical seeded weights and a declared 1e-9 rad control
error budget against an FP64 reference during warmup. `--policy-precision 64`
supplies its matched ablation; all stress arithmetic remains Quadrants FP64.

The `recovery` timing scope repeatedly processes a frozen actual snapshot;
it is a kernel diagnostic and omits changing rigid trajectories. Supplement
it with synchronized stress-stage costs on a complete changing trajectory:

```bash
python -m research.rigid_stress.native_recovery_trajectory_profile \
    --envs 1024 --seed 510000 --warmup 900 --steps 2400 --repetitions 3 \
    --output-mode max --output "$RIGID_STRESS_DATA_ROOT/runs/native/dynamic-recovery-max.json"
python -m research.rigid_stress.native_trajectory_oracle \
    --envs 32 --steps 2400 --seed 623001 --output-mode full \
    --output "$RIGID_STRESS_DATA_ROOT/runs/native/full-field-oracle.json"
```

The dynamic profile includes native recovery, contact-radius writes and stress
invalidation during partial resets. Its synchronizations are intrusive;
actual aggregate rates come from ordinary `live`/`policy` timing, including
controller, observation, all reset motion and contact parameter updates.

The default `auto` method uses a memory-bounded shared inverse on this small
mesh. Compare `--method direct --serial-solve`, `--method direct`, and
`--method inverse --inverse-precision 32` at identical mesh/tolerances.
`--history 4` enables the tested temporal variant; h=0 is selected because
the full changing-contact measurement was faster and used less memory.
`--serial-pressure` selects the native scalar pressure path for a matched
pressure scheduling ablation. CUDA warp pressure is the selected default.
`--full-inverse` disables the exact complete-exterior operator, and
`--serial-scatter` disables face reduction. `--seed 623001` selects another
independent varied workload; use the same seed for matched comparisons.
`--contact-warp-scatter` compares the original cooperative contact scheduling
against bounded face tasks. `--scatter-tasks-per-env 1` deliberately exercises
the complete device-side overflow fallback while preserving the contact law,
samples and final acceptance budgets. The ordinary timing includes that
fallback, and each repeat records `scatter_overflow_calls`.
`--uncached-peak` and `--uncached-face-bounds` disable the tested immutable
geometry reuse variants. `--source-revision` records the declared checkout;
the benchmark also records exact numerical source hashes and GPU UUID.
The current adaptive-pad workload has actual combined contact friction
between 0.42 and 0.84. It differs from the earlier fixed-friction benchmark.
See [CONTACT_REPAIR_20261010.md](CONTACT_REPAIR_20261010.md) for checkpoint
revisions, regression results and performance-development evidence.
`--trace` exports a separate 50-step Torch/CUPTI CPU/CUDA trace after timing;
its overhead is excluded from the reported rates. The pressure microkernels
repeat query/integration/small-solve work to diagnose the hotspot and are
excluded from the additive pipeline table. Captured Quadrants graph kernels
are omitted by its kernel profiler, so use wall stages and the optional trace.

`recovery` repeats a frozen actual contact snapshot with its matching
pre-integration frame; it is not changing-contact trajectory throughput.
`live` includes the whole rigid trajectory, changing contacts/radii, slip,
release and independently delayed resets. `policy` adds device observations
and a fixed seeded FP64 26->128->128->7 tanh MLP; it measures inference and
rollouts, not completed reinforcement-learning training.

The MLP has tanh after the first two linear layers and a linear output.
Use the same command with `--scope rigid`, `policy-rigid`, `recovery` and
`policy`, changing the output name, to reproduce the five selected scopes.
Use `--envs 8` for the serial-baseline comparison. The exact selected source
for the earlier fixed-friction milestone is `dbfdf20d`; the evidence manifest records revisions and arguments for
earlier ablations, including the normalized baseline patch.

Every-step validity is checked separately from throughput:

```bash
python -m research.rigid_stress.native_validity_probe --scope live \
    --envs 1024 --warmup 900 --steps 1200 \
    --output "$RIGID_STRESS_DATA_ROOT/runs/native/validity-live.json"
python -m research.rigid_stress.native_validity_probe --scope policy \
    --envs 1024 --warmup 900 --steps 1200 \
    --output "$RIGID_STRESS_DATA_ROOT/runs/native/validity-policy.json"
python -m pytest tests/rigid/test_stress.py --backend cpu \
    -k 'not lifecycle' -n 0 -s -q
```

The audit keeps native failure counters across independent resets, reading
them once at the end. It checks every one of the 2,150,400 environment steps
per scope, including warmup, and publishes no throughput rate.
GPU tests include the actual rigid lifecycle checks; CPU runs the 19
numerical cases. Do not run CPU numerical work on a login node on the cluster.

Allocate the measured cluster hardware before sourcing the environment:

```bash
source /mnt/data/shared/config/env.sh
GENESIS_IMAGE_VER=1_26 gs-srun --partition=rtx-mid --ntasks=1 --gpus=1 \
    --cpus-per-task=8 --mem=64G --time=00:20:00 bash
export RIGID_STRESS_DATA_ROOT=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery
export RIGID_STRESS_ENV=/mnt/data/zhaofeng/venvs/rigid-stress-recovery
source research/rigid_stress/scripts/cluster_env.sh
mkdir -p "$RIGID_STRESS_DATA_ROOT/runs/native"
```

Run the commands from the repository root with the environment Python
(`"$RIGID_STRESS_ENV/bin/python"`); place redirected stdout/stderr in that
run directory. CPU tests use the same partition without `--gpus=1`.

Warmup quality checks are outside steady-state timing. They record actual
nonzero egg contacts, both finger contacts, tangential force, radii, friction,
height, complete residual and correction/fallback counts. They do not certify
stress discretization convergence. See [NATIVE_API.md](NATIVE_API.md) for the
finite-footprint contract and current support boundaries.

An independent untimed temporal-load probe is reproducible with:

```bash
python -m research.rigid_stress.native_history_probe --envs 8 --steps 1200 \
    --output "$RIGID_STRESS_DATA_ROOT/runs/native/history.json"
```

It stores actual complete RHS arrays under the requested data path and checks
optimistic h=0/1/4 load-subspace acceptance offline. It does not run production
NumPy recovery or contribute a throughput rate.

## Copied-condition native batch scaling

To scale the batch without introducing new randomized contacts, first save
the verified 1024-condition bank, including all three IK targets and reset
delays. Subsequent scenes repeat every bank row modulo its row count. Every
environment still runs independent rigid dynamics and complete stress recovery.

```bash
"$RIGID_STRESS_ENV/bin/python" research/rigid_stress/scripts/copied_condition_sweep.py \
    --output "$RIGID_STRESS_DATA_ROOT/runs/copied-scaling/discovery" \
    --envs 1024 2048 4096 8192 16384 32768 65536
```

The runner records commands, exit statuses, per-repeat rates and the bank
SHA256 after each process. Discovery uses one full repeat per point. Repeat
the best points with `--repetitions 3 --bank .../conditions1024.npz`, changing
the output directory. Use `--scope policy`, `rigid` or `policy-rigid` with
the identical bank for matched scopes. The underlying benchmark also accepts
`--conditions bank.npz` and `--save-conditions bank.npz`; `--compact-log`
keeps full diagnostics in JSON while shortening stdout.

The separate every-step validity probe accepts `--conditions bank.npz`.
Copies preserve declared inputs/control targets; floating-point trajectories
must still be checked at each batch size. A copied batch is a scaling study,
not additional coverage of randomized contact configurations.

## Historical external-library prototype

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
