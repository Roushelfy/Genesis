# DFSPH solver development status

Companion of [SPH_SOLVER_GOAL.md](SPH_SOLVER_GOAL.md) (target) and [SPH_SOLVER_DESIGN.md](SPH_SOLVER_DESIGN.md)
(equations and acceptance). This file records what is implemented, what is measured, and where the evidence is. Campaign
drivers and their JSON receipts live in FluidBench (`adapters/genesis/sph_validation/`, `results/sph_validation/`,
branch `bench/auto-research` of https://github.com/Roushelfy/fluidbench). Every number below names its receipt.

Hardware of record: NVIDIA RTX PRO 6000 Blackwell Server Edition (cluster partition rtx-mid), fp32.

## Milestones

| Milestone | State |
| --- | --- |
| M0 equations, units, cases, acceptance, measurement boundaries | done: SPH_SOLVER_DESIGN.md; baseline measurements below |
| M1 implicit interior and wall viscosity with analytic validation and load accounting | in progress |
| M2 reaction readout and rigid stage integration | readout implemented (fixed and prescribed links), rigid integration pending |
| M3 strong pressure-viscosity-rigid coupling | pending |
| M4 robustness campaigns | pending |
| M5 optimization ablations | pending |
| M6 FluidBench re-run and documentation | pending |

## Baseline at the starting revision (92c78327)

- Existing SPH tests (`tests/particles/test_sph.py`, `tests/coupling/test_sph_rigid.py`, legacy solver): 17 passed, 2
  failed (`test_pressure_carries_column_weight[WCSPH|DFSPH]`: the band `heights >= 0.3` holds no particle, so its mean
  pressure is NaN). Log: `/mnt/data/zhaofeng/projects/workspace/genesis-world/logs/pytest_sph_base.log`.
- FluidBench S1 f2 R2 with DFSPH: uninstrumented timing record `gs-dfsph__S1__f2@sphgoal-m0-base__R2`, instrumented
  profiles `results/profiles/gs-sph-dfsph_S1_f2_R2_{coarse,fine}_*.json` (FluidBench), earlier profile at 49f63720:
  pressure solves 52 %, other SPH 13 %, rigid 28 % of a sub-step.

## Findings that change the model

1. Explicit viscosity replaced by an implicit backward Euler solve (conjugate gradient, block-Jacobi preconditioner) of
   the Rayleigh dissipation of SPH_SOLVER_DESIGN.md section 3.3. The material `viscosity` is the kinematic viscosity
   (m^2/s), converted once to the dynamic viscosity.
2. Boundary pressure of the fork (Akinci particles, full kernel gradient) acts as a rough wall: fluid in rigid rotation
   at water viscosity inside a fixed cup lost 58 % of its angular momentum in 1 s (spin-down test, dx 5 mm), where the
   physical spin-down takes tens of seconds. Projecting the boundary kernel gradient on the wall normal (smooth
   impermeable wall, reaction at the fluid particle) keeps 100-102 % over 1 s. Development logs:
   `logs/spin_down.log`, `logs/spin_down2.log`.
3. Wall viscosity: the pair form between fluid and a single layer of boundary particles is mostly normal to the wall,
   so its tangential friction is several times too weak and the fluid slips. It is replaced by ghost-fluid columns
   behind each boundary sample with linear extrapolation of the tangential velocity (second-order no-slip), projected on
   the tangent plane, with the reaction at the fluid particle.
4. The SPH viscous operator with a support of two particle spacings reproduces only 0.70 of the Laplacian of a shear
   field on a cubic lattice and 0.73-0.75 on a relaxed arrangement (shear-wave decay, independent of resolution at fixed
   neighbor count). A per-particle trace renormalization of the fourth kernel moment is in place (it corrects the
   neighbor deficit at walls and free surfaces), the remaining part is the anisotropy of the particle arrangement.
5. The viscosity-pressure agreement of a substep is solved by a block iteration whose density solves use a forcing term
   of 0.1 and whose pressure is extrapolated by Chebyshev semi-iteration on the rigorous spectrum bound [1 / L, 1].
6. Test geometry: a container built from several overlapping meshes gives boundary samples whose normal points into the
   fluid and makes the start explode. Validation containers are single watertight solids of revolution.

## Open items

- M1 acceptance runs (trough V1, Couette V2, pipe V3, decay V4) at three resolutions.
- Cost of the coupling passes at moderate viscosity (8-12 passes per substep at nu = 1e-3 before the tangential
  projection of the wall tensor).
