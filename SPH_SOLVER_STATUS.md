# DFSPH solver development status

Companion of [SPH_SOLVER_GOAL.md](SPH_SOLVER_GOAL.md) (target) and [SPH_SOLVER_DESIGN.md](SPH_SOLVER_DESIGN.md)
(equations and acceptance). This file records what is implemented, what is measured, and where the evidence is. Campaign
drivers and their JSON receipts live in FluidBench (`adapters/genesis/sph_validation/`, `results/sph_validation/`,
branch `bench/auto-research` of https://github.com/Roushelfy/fluidbench). Every number below names its receipt (file name
prefix in `results/sph_validation/`). Receipts tagged `dev-*` or `*_dirty` come from development snapshots (a commit plus
uncommitted changes, recorded in the receipt as `genesis_dirty`); acceptance numbers come from committed revisions only.

Hardware of record: NVIDIA RTX PRO 6000 Blackwell Server Edition (cluster partition rtx-mid), fp32.

## Milestones

| Milestone | State |
| --- | --- |
| M0 equations, units, cases, acceptance, measurement boundaries | done: SPH_SOLVER_DESIGN.md; baseline measurements below |
| M1 implicit interior and wall viscosity with analytic validation and load accounting | implemented; V2, V4, V5 measured, V1 and V3 open (see open items) |
| M2 reaction readout and rigid stage integration | done: readout for fixed, prescribed and dynamic links, fluid stage inside the rigid substep |
| M3 strong pressure-viscosity-rigid coupling | block iteration converged with momentum-exact exchange (V6); constraint consistency loop open |
| M4 robustness campaigns | in progress (light link, leakage); thin wall, coarse, state open |
| M5 optimization ablations | pending |
| M6 FluidBench re-run and documentation | pending |

## Baseline at the starting revision (92c78327)

- Existing SPH tests (`tests/particles/test_sph.py`, `tests/coupling/test_sph_rigid.py`, legacy solver): 17 passed, 2
  failed (`test_pressure_carries_column_weight[WCSPH|DFSPH]`: the band `heights >= 0.3` holds no particle, so its mean
  pressure is NaN). Log: `/mnt/data/zhaofeng/projects/workspace/genesis-world/logs/pytest_sph_base.log`.
- FluidBench S1 f2 R2 with DFSPH: uninstrumented timing record `gs-dfsph__S1__f2@sphgoal-m0-base__R2`, instrumented
  profiles `results/profiles/gs-sph-dfsph_S1_f2_R2_{coarse,fine}_*.json` (FluidBench), earlier profile at 49f63720:
  pressure solves 52 %, other SPH 13 %, rigid 28 % of a sub-step.
- Water at rest in a cup (dx 3.5 mm, regular sampler, no surface tension) keeps a particle velocity of 6-8 cm/s rms for
  over a second at the starting revision: the potential energy the initial lattice releases as it compacts under
  gravity stays as kinetic energy at the particle scale, which water viscosity damps over dx^2 / nu ~ 10 s. The same
  holds at every later revision. Validation cases therefore settle with their velocities reset. Logs:
  `logs/noise_water_92c783279.log`, `logs/noise3_*.log`, `logs/noise4_*.log`.

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
4. Anisotropy of the viscous operator. The pair form reproduces the Laplacian only if the fourth moment tensor of its
   weights is isotropic, which two particle spacings of support do not give: on a cubic lattice it reproduces 0.70 of
   the Laplacian of a shear flow and 1.19 of that of an extension (offline lattice computation). A per-particle
   correction of the pair weights by a function of the direction in the 15 even spherical harmonics up to degree 4,
   solved from the moment equations with the wall ghosts included, brings both to 0.99 on the lattice and 0.97 +- 0.06
   on a shaken one. The pair weight is the mean of the corrections of both particles, clamped at zero, so the operator
   stays symmetric, positive semi-definite and central.
5. Linear-shear consistency at walls. The ghost columns are a continuum and the fluid neighbors of a near-wall particle
   sit on discrete layers, so a linear shear flow along a wall leaves a spurious force on the first layer (a fifth of the
   viscous stress gradient of a unit shear on a cubic lattice). Each slot tensor is scaled so that the operator is exact
   for a linear shear along the faced wall. Couette torque at dx 5 mm went from 15-35 % to 2-12 % above the reference
   (receipts `couette__corr1_pbs_dx0.005`, `couette__g3_pbs_dx0.005`).
6. The viscosity-pressure agreement of a substep is solved by a block iteration whose density solves use a forcing term
   of 0.1 and whose pressure is extrapolated by Chebyshev semi-iteration on the spectrum bound [1 / L, 1]. At nu = 1e-2
   the passes contract by about 0.87 each (receipt `spindown__devhist_nu1e-2`, `worst_coupling_history`), slower than
   the bound predicts, since a pass runs one density iteration on the pressure of the previous one.
7. Rigid coupling. The fluid stage runs inside the rigid substep, before the constraint solve, with the mobility of the
   coupled links in the pressure and viscosity solves. The viscous impulse a link takes is the reaction of the impulse
   the fluid took at the link velocities it saw, so the exchange balances momentum at every pass, converged or not, and
   the implicit link velocities enter the next pass as a correction (stable for light links).
8. Rigid integrator defect (fixed in 546010aeb): the single precision midpoint integration of free bodies stopped its
   Newton iteration at the initial guess when the residual was below 1e-6 (1 + 2 |I w| / h) N m, which drops every
   torque below about 1e-6 N m on a small body. The fluid torques on a 3 cm box are of that order, so the box kept its
   spin while the fluid lost angular momentum. With the tolerance floor scaled by the torque, the closed fluid-box system
   keeps linear and angular momentum to 1.4e-5 and 7.6e-6 per second at nu = 1e-3 (receipt `momentum__mid2_nu1e-3`,
   V6 limit 1e-4), and 7e-7 and 7.9e-6 inviscid (`momentum__mid2_nu0`). Regression test
   `tests/rigid/test_dynamics.py::test_small_free_body_torque`.
9. Leakage through a single boundary layer. A free-surface particle short of fluid neighbors reaches the density target
   only very close to the boundary particles and can cross their layer, after which the boundary pressure keeps it
   inside the geom. A wall clearance guard after the solves keeps every particle a quarter spacing in front of the
   surface it faces, as a unilateral velocity constraint with a slow position recovery and the reaction on the link.
   The faced surface comes from the boundary particles whose tangent region contains the particle, the nearest along
   the normal, so that the far face of a thin wall does not count. With it, a floating box keeps no fluid inside
   (`logs/mid2_gap.log`, box frame count).
10. A cube at half the density of water is unstable face down (a cube floats face down below 0.21 and tilts onto an edge
    between 0.21 and 0.79): once torques reach the box (finding 8), it tilts by about 45 degrees. The V8 case at density
    0.5 is a plank of aspect 2.
11. Couette torque grows with the hydrostatic pressure: at dx 3.5 mm the torque on the deep layer is +4 %, +21 % and
    +48 % above the reference at gravity 2, 9.81 and 20 m/s^2 (receipts `couette__mid3_g2_dx0.0035`,
    `couette__g3_pbs_dx0.0035`, `couette__mid3_g20_dx0.0035`). Projecting the boundary pressure on a smooth surface
    normal (kernel-weighted mean of the normals of the faced samples) does not change it (`couette__face2_g*`), and a
    128-facet rotor gives the same as the 32-facet one (`couette__mid2_rot128_dx0.0035`). Open.
12. Particle noise near the edges of a box immersed in fluid at rest (fixed or articulated paddle): the viscous
    dissipation readout of a fluid at rest is 20 times that of the same cup without the box, concentrated at its edges
    and corners. It drops with the unprojected boundary gradient (whose tangential roughness damps it), and does not
    change with the density constraint alone unprojected. Open; it limits the V10 energy balance.

## Measured validation (development snapshots, to be repeated on the committed revision)

| Case | Result | Receipt |
| --- | --- | --- |
| V2 Couette dx 5 / 3.5 / 2.5 mm, depth 0.05 m | torque within 2-9 %, 0.3-7 %, 1-7 % for t >= 0.1 s | `couette__g3_pbs_dx*` |
| V2 Couette depth 0.08 m | 2-12 % at 5 mm, 16-25 % at 3.5 mm, 10-29 % at 2.5 mm (finding 11) | `couette__g3_pbs_dx*` |
| V4 mode decay nu = 1e-3 | rate -39 %, -17 %, -13 % at dx 3.5, 2.5, 1.75 mm; dissipation readout 0.99 of the energy loss | `decay__face2_decay_nu1e-3_dx*` |
| V5 spin-down nu = 1e-2, dx 5 mm | decay rate -11.6 % to -19 % (within 20 %) | `spindown__corr1_dx0.005_arm5e-4`, `spindown__g3_spin_arm5e-4` |
| V5 light link (no armature) | stable, rotor stops within the first step like the reference | `spindown__g3_spin_arm0` |
| V6 momentum, nu = 1e-3 and inviscid | 1.4e-5 / 7.6e-6 per s, 7e-7 / 7.9e-6 per s | `momentum__mid2_nu*` |
| V10 hold (slide + hinge) | joint forces vs fluid readout within 1 %; vs Archimedes 8 % / 25 % | `articulation__g3_hold` |
| V10 stir | joint torque vs fluid torque 1.4-3.5 %; energy balance open (finding 12) | `articulation__mid2_stir` |

## Open items

- V1 (Stokes first problem) and V3 (pipe flow) drivers: the sliding-floor trough is limited by the closed-cavity return
  flow and the free-surface pressure resolution.
- Findings 11 and 12.
- V8 draft offset of about one particle spacing (receipts `floating__mid3_*`), convergence ladder.
- Constraint consistency loop (SPH_SOLVER_DESIGN.md section 6), V9, V11, V12, M5, M6.
