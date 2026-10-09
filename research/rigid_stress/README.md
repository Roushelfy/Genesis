# Fixed-shape rigid stress recovery: agent handoff

Goal: maximize **measured aggregate environment transitions/s** for a Franka
Panda grasping a hollow egg-shaped rigid entity, while returning its global
maximum von Mises stress at each requested simulation step. Geometry stays
rigid in Genesis; an auxiliary small-strain elastic model recovers stress.

Upstream baseline: `Genesis-Embodied-AI/genesis-world`, main commit
`e9e1214d192914ddec85ca28014c459cfbc6c860`, 2026-10-08. Work branch in the
existing user fork: `Roushelfy/Genesis:rigid-stress-recovery`.

## What is ready, and what remains

| Component | Status |
|---|---|
| Full egg P2 FEM, consistent mass, inertia relief, shared CPU LU | Runnable, imported from tested experiments |
| Arbitrary finite-area vector loads, spatial index, global corner maximum | CPU tested; no contact symmetry or fixed load basis |
| Previous-frame predictor, compact failed RHS, per-env cached history | CPU tested; same-tolerance batching ablations included |
| FP32 CPU accuracy study and one-step refinement | Code and measured evidence included |
| `cpu.py`, resets, batched contact-facing adapter | CPU smoke checked; see `evidence/seed_cpu_validation.json` |
| `assets.py`, exterior mesh plus shell mass/COM/inertia URDF | Generated and XML/numerical checked |
| `gpu_prototype.py` | CUDA-capable Torch prototype checked on CPU; no CUDA test |
| `franka_egg.py` | Current-public-API integration seed; not executed or accepted as a grasp/throughput benchmark in this environment |
| Production sparse GPU backend, fused peak, GPU history, policy observation | **To implement and validate** |
| Live-contact footprint/wrench fidelity and full-model mesh convergence | **To implement and validate** |

The seed mapper preserves force, but reports the moment difference caused by
replacing a point force with a finite patch. This is a known integration gap,
not an acceptable hidden approximation. The `goal` requires resolving it.

## Start here

Read [GOAL.md](GOAL.md), [AGENTS.md](AGENTS.md), then:

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
python -m research.rigid_stress.check_cpu
python -m research.rigid_stress.assets --level 1
```

Optional Torch prototype check (CPU, not GPU):

```bash
python -m research.rigid_stress.check_cpu --torch
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

## Franka integration seed

Install upstream Genesis and its assets as documented in the root README;
PyTorch is a separate platform-dependent prerequisite. Then:

```bash
python -m research.rigid_stress.franka_egg --backend cpu --envs 1 --stress cpu --vis
python -m research.rigid_stress.franka_egg --backend cuda --envs 8 --stress off
python -m research.rigid_stress.franka_egg --backend cuda --envs 8 --stress cpu
```

The last command transfers live contacts to CPU recovery for integration
debugging. It is not the intended GPU hot path. The seed's timing includes
first compilation and is labeled accordingly; it must not be used as steady
RL throughput. The agent must add the sparse GPU backend, validate the grasp,
run the complete suite, and replace this seed benchmark with warmed measurements.

## Boundaries

Only geometry is assumed fixed with respect to contact patterns. Fixed
linear material/mass are declared model inputs needed to reuse one factor.
Contact positions, directions, counts, areas and symmetry may all change.
Friction is required. Smooth contact history is a performance opportunity,
not a correctness condition; failures/impacts/reset fall back independently.
Output is a scalar maximum per environment, not a full stress field.

This handoff makes no measured GPU/RL FPS claim. The old quarter-model
convergence evidence is diagnostic only and cannot establish convergence for
the required arbitrary-contact full model. No physical egg failure calibration
or elasticity-to-rigid feedback is supplied.

All added source is covered by the repository's Apache-2.0 license. Reference
file provenance and SHA256 hashes are in `evidence/import_manifest.json`.
