# Fixed-shape rigid stress recovery

## Current direction: native Quadrants feature

The user's 2026-10-10 direction is in [GOAL.md](GOAL.md) and
[docs/NATIVE_QUADRANTS_PLAN.md](docs/NATIVE_QUADRANTS_PLAN.md). Implement stress
recovery inside Genesis's existing rigid solver, with all production numerical
computation in Quadrants. Prioritize detailed profiling and throughput
optimization on low-resolution full-shell meshes; defer physical mesh and
quadrature convergence until the applicable performance work is complete.
Same-mesh numerical consistency remains required throughout.

The native implementation is in `genesis/engine/solvers/rigid/stress/`, with
`RigidLink.configure_stress_recovery`, `get_max_stress` and
`set_stress_contact_radius`. The normal example is
`examples/rigid/franka_egg_stress.py`; it imports no research implementation.
Initial same-mesh operator, direct solve, finite-pressure and lifecycle checks
pass. Performance profiling and optimization are in progress; see the
[native journal](docs/PROGRESS_20261010_NATIVE.md).

The research code, historical status table, native-library setup and fine-mesh commands below
describe the existing research prototype and historical evidence. They do not
establish that the requested native feature exists, and are not the new default
implementation or iteration workflow. Read the new plan before using them.

Goal: maximize **measured aggregate environment transitions/s** for a Franka
Panda grasping a hollow egg-shaped rigid entity, while returning its global
maximum von Mises stress at each requested simulation step. Geometry stays
rigid in Genesis; an auxiliary small-strain elastic model recovers stress.

Upstream baseline: `Genesis-Embodied-AI/genesis-world`, main commit
`e9e1214d192914ddec85ca28014c459cfbc6c860`, 2026-10-08. Work branch in the
existing user fork: `Roushelfy/Genesis:rigid-stress-recovery`.

## Historical research prototype status

| Component | Status |
|---|---|
| Full egg P2 FEM, consistent mass, inertia relief, shared CPU LU | Runnable, imported from tested experiments |
| Arbitrary finite-area vector loads, spatial index, global corner maximum | CPU tested, wrench-constrained admitted vector tractions |
| Previous-frame predictor, compact failed RHS, per-env cached history | CPU tested; same-tolerance batching ablations included |
| FP32 CPU accuracy study and one-step refinement | Code and measured evidence included |
| `cpu.py`, resets, batched contact-facing adapter | CPU smoke checked; see `evidence/seed_cpu_validation.json` |
| `assets.py`, exterior mesh plus shell mass/COM/inertia URDF | Generated and XML/numerical checked |
| `gpu_prototype.py` | CUDA-capable Torch prototype checked on CPU; no CUDA test |
| `sparse_gpu.py`, shared sparse L/U + cuSPARSE SpSM, fused full peak | Device validated on RTX PRO 6000 Blackwell |
| `device_pressure.py`, changing finite pad wrenches | Device vs CPU FP64 validated, including an actual sliding snapshot |
| `franka_egg.py` | Real 600-step GPU grasp and CPU direct comparison executed, contact-solve pose matched |
| Full-shell mesh and quadrature convergence | Full levels 4/5/6 pass 2% for asymmetric, minimum-radius and nominal live loads |
| GPU independent histories, FP32 refinement, native cuDSS, resets, substep observers | Device tested; strict/throughput math paths retain full residual |
| Reusable Panda and full contact-replay/live/policy benchmarks | Runnable; fine-grid performance selection and held-out validation active |

The live path preserves the complete contact wrench with a declared compliant
pad model. See [docs/PAD_LAW.md](docs/PAD_LAW.md) and the scope-specific measured
[initial progress report](docs/PROGRESS_20261009.md) and the
[new GPU acceptance checkpoint](docs/PROGRESS_20261009_GPU.md), and the
[fine performance and real grasp checkpoint](docs/PROGRESS_20261009_PERFORMANCE.md). The full acceptance and optimization
goal remains active. Current coarse timings have a separate physical error.

## Start here

Read [GOAL.md](GOAL.md), [AGENTS.md](AGENTS.md) and
[docs/NATIVE_QUADRANTS_PLAN.md](docs/NATIVE_QUADRANTS_PLAN.md), then consult the
relevant historical references:

1. [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md): binding scope and proposed accuracy profiles.
2. [docs/ALGORITHM.md](docs/ALGORITHM.md): equations and predictor/corrector.
3. [docs/GENESIS_INTEGRATION.md](docs/GENESIS_INTEGRATION.md): audited contact APIs and Franka integration.
4. [docs/GPU_IMPLEMENTATION.md](docs/GPU_IMPLEMENTATION.md): proposed device implementation and optimization order.
5. [docs/VALIDATION.md](docs/VALIDATION.md): acceptance suite and throughput contract.
6. [docs/CPU_EVIDENCE.md](docs/CPU_EVIDENCE.md): measured results, precision tradeoffs and rejected assumptions.

## CPU-only mathematical reference

From repository root, in an isolated Python 3.10–3.13 environment:

```bash
python -m pip install -r research/rigid_stress/requirements-cpu.txt
python -m research.rigid_stress.check_cpu --output "$RIGID_STRESS_DATA_ROOT/runs/check/cpu.json"
python -m research.rigid_stress.assets --level 1 --output "$RIGID_STRESS_DATA_ROOT/assets/egg"
```

Optional Torch prototype check (CPU, not GPU):

```bash
python -m research.rigid_stress.check_cpu --torch --output "$RIGID_STRESS_DATA_ROOT/runs/check/cpu-torch.json"
```

The optional check needs an installed PyTorch matching the local platform.
The sparse/direct reference requires no Genesis, PyTorch or CUDA installation.
Default application wrapper uses the full level-2/two-wall-layer P2 model
(9,630 DOFs, 1,920 tets); the short check uses level 1 (2,430 DOFs, 480 tets).
Neither is physically stress-converged by this check.

## Reproduce the previous experiments

The original relative layout is preserved under `reference/`; scientific
scripts are intentionally imported without algorithm/style rewriting. Run
them in a disposable worktree, as several write result files beside themselves.

```bash
python research/rigid_stress/reference/output/rigid_stress_reference.py
python research/rigid_stress/reference/output/arbitrary_contact_cpu/traction_validation.py
python research/rigid_stress/reference/output/arbitrary_contact_cpu/pipeline_benchmark.py --frames 16 --repeats 3
python research/rigid_stress/reference/output/smooth_friction_cpu/temporal_benchmark.py --frames 128 --repeats 2
python research/rigid_stress/reference/tmp/precision_validation/check_fp32.py
python research/rigid_stress/reference/tmp/batch_history_validation/benchmark.py --quick
```

Omit `--quick` to reproduce the full batch/history ablations (long CPU run).
Core dependency versions used in the latest local check are recorded in
`evidence/seed_environment.json`. Plot/report/convergence scripts can require
Matplotlib, Pillow or pymetis in addition to the minimum numerical stack.

## Franka integration diagnostic

Install upstream Genesis and its assets as documented in the root README;
PyTorch is a separate platform-dependent prerequisite. Then:

```bash
python -m research.rigid_stress.franka_egg --backend cpu --envs 1 --stress cpu --vis \
    --output "$RIGID_STRESS_DATA_ROOT/runs/cpu/rollout.json"
python -m research.rigid_stress.franka_egg --backend cuda --envs 8 --stress off \
    --output "$RIGID_STRESS_DATA_ROOT/runs/rigid/rollout.json"
python -m research.rigid_stress.franka_egg --backend cuda --envs 1 --stress gpu --verify-gpu \
    --save-contacts --video "$RIGID_STRESS_DATA_ROOT/runs/demo/grasp.mp4" \
    --output "$RIGID_STRESS_DATA_ROOT/runs/demo/rollout.json"
```

The last command checks GPU mapping/direct recovery against CPU FP64 every
frame and records actual rigid motion. Its diagnostics transfer to host.
The timing includes verification, recording and first compilation. Use the
separate warmed benchmark command for the explicitly assembled-RHS scope.
The progress report lists reproducible allocated-node commands and remaining
physical and live-throughput acceptance work.

## Reusable device path

One-command setup, selected acceptance checks, an actual fine-shell video and
scoped benchmark examples are in [docs/RUNNING.md](docs/RUNNING.md). All
runtime/cache destinations are explicit; run numerical work on allocated nodes.

Install the optional native dependencies for large full-shell operators:

```bash
python -m pip install -r research/rigid_stress/requirements-cholmod.txt \
    -r research/rigid_stress/requirements-cudss.txt
python -m research.rigid_stress.check_panda --envs 32 --steps 1200 --temporal \
    --output "$RIGID_STRESS_DATA_ROOT/runs/check/panda32.json"
python -m research.rigid_stress.benchmark_live --level 6 --cpu-factor none \
    --factor-backend cudss --envs 8 --scope live \
    --output "$RIGID_STRESS_DATA_ROOT/runs/benchmark/live8.json"
python -m research.rigid_stress.benchmark_replay --level 6 --envs 8 \
    --replay "$RIGID_STRESS_DATA_ROOT/runs/check/panda32.contacts.npz" \
    --output "$RIGID_STRESS_DATA_ROOT/runs/benchmark/stress8.json"
```

CPU factor `none` skips only an unused offline factor in device timing. CPU
oracle checks use `superlu` or native `cholmod`; they reject a missing factor.
Benchmarks require at least three ten-second repeats and warm a complete grasp.
Run all experiments on allocated hardware with explicit data/cache destinations.

## Boundaries

Only geometry is assumed fixed with respect to contact patterns. Fixed
linear material/mass are declared model inputs needed to reuse one factor.
Contact positions, directions, counts, areas and symmetry may all change.
Friction is required. Smooth contact history is a performance opportunity,
not a correctness condition; failures/impacts/reset fall back independently.
Output is a scalar maximum per environment, not a full stress field.

Measured GPU recovery rates have an explicit microbenchmark scope. The old quarter-model
convergence evidence is diagnostic only and cannot establish convergence for
the required arbitrary-contact full model. No physical egg failure calibration
or elasticity-to-rigid feedback is supplied.

All added source is covered by the repository's Apache-2.0 license. Reference
file provenance and SHA256 hashes are in `evidence/import_manifest.json`.
