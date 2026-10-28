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
| M3 strong pressure-viscosity-rigid coupling | block iteration converged with momentum-exact exchange (V6); constraint consistency loop implemented (`SPHOptions.max_constraint_iterations`, contact receipts) |
| M4 robustness campaigns | in progress: light link, leakage guard, snapshot and selected-environment reset (V12 driver, warm starts in the state); thin wall and coarse cups open |
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
2. Boundary pressure (Akinci particles, full kernel gradient) acts as a rough wall: fluid in rigid rotation at water
   viscosity inside a fixed cup lost 58 % of its angular momentum in 1 s at the starting revision (spin-down test, dx
   5 mm, boundary spacing 0.75 particle sizes), where the physical spin-down takes tens of seconds (`logs/spin_down.log`).
   Projecting the boundary kernel gradient on the wall normal (a smooth impermeable wall) kept 100-102 % over 1 s
   (`logs/spin_down2.log`), but the projected force is not the gradient of the density the boundary adds, so the
   pressure solve feeds the density variation of fluid sliding past the samples back along the normal only: a closed
   column of water at rest heated from 4.4e-4 to 8.9e-4 J of kinetic energy over 0.9 s (`logs/fg_cn_proj75.log`,
   dev/closed_noise.py), where the full gradient lets it decay from 3.3e-4 to 1.3e-4 J (`logs/fg_cn_full50.log`,
   `logs/fe_cn.log`). The projection is therefore not used. The roughness drops with denser boundary sampling: with the
   isotropic pressure of finding 11, the rotating fluid kept 0.74, 0.86 and 0.88 of its angular momentum over 1 s at
   spacings of 0.75, 0.5 and 0.35 particle sizes (`logs/fg_spin_full{75,50,35}.log`). The spacing is 0.5 particle
   sizes; at it the fluid keeps 0.72 without the isotropic pressure (`logs/fe_spin_inviscid.log`).
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
5. Linear-shear consistency at walls (tried, removed). The ghost columns are a continuum and the fluid neighbors of a
   near-wall particle sit on discrete layers, so a linear shear along a wall leaves a spurious force on the first layer
   (a fifth of the viscous stress gradient of a unit shear on a cubic lattice, offline lattice computation). Scaling
   each slot tensor to make the operator exact for a linear shear lowered the Couette torque, but that torque was
   inflated by the pressure friction of finding 11; with the isotropic pressure the scaling leaves the torque 13 % low
   and the spin-down 27 % slow, against 3-7 % and 17 % without it (receipts `couette__mxA_couette`, `couette__mxB_couette`,
   `spindown__mx{A,B}_spin`), so it is not used.
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
11. Pressure friction. The Couette torque grew with the hydrostatic pressure: at dx 3.5 mm the torque on the deep layer
    was +4 %, +21 % and +48 % above the reference at gravity 2, 9.81 and 20 m/s^2 (receipts
    `couette__mid3_g2_dx0.0035`, `couette__g3_pbs_dx0.0035`, `couette__mid3_g20_dx0.0035`), and its excess was the same
    2-3e-4 N m at nu = 3e-4, 1e-3 and 3e-3 (`couette__c1_nu*`): an additive, pressure-proportional shear stress. The
    stress budget at mid gap (`logs/stress_g9.81.log`, dev/couette_stress.py) shows the cause: the symmetric pressure
    pairs carried a shear stress of -2.3 Pa at a mean pressure of 330 Pa, twice the viscous stress, because a sheared
    neighborhood is anisotropic and the pair form turns the pressure into a stress along the anisotropy. The wall
    geometry was not involved (smooth-normal projection `couette__face2_g*`, 128-facet rotor
    `couette__mid2_rot128_dx0.0035`, no wall guard `couette__tog4_noguard_*`: unchanged). Particle shifting reduced it
    (`couette__shift*`) but broke angular momentum by 4e-3 to 1.4e-2 per second (`momentum__shift*_mom`) and lowered the
    hydrostatic pressure, so it is not used. A symmetric directional correction of the pressure pair weights in the
    degree 2 harmonics makes the pressure virial of every particle isotropic (central, momentum conserving): with it
    the torque no longer depends on gravity (`couette__iso1_g20_dx0.0035`: within 4.5 % deep at g = 20), the momentum
    balance is unchanged (`momentum__iso1_nu1e-3`), and with the full boundary gradient of finding 2 the Couette torque
    is within 2-6 % at both depths (`couette__fgE_couette`) against 10-27 % and 26-48 % without it
    (`couette__fe_couette_dx0.0035`). But the corrected force is not the gradient of the density estimate, and water at
    rest heats: the closed column of finding 2 went from 2.3e-3 to 3.7e-3 J of kinetic energy over 0.9 s
    (`logs/fg_cn_full50_iso.log`). The correction is therefore the option `SPHOptions.isotropic_pressure`, off by
    default. In FluidBench S1 (water) the spin-down time is 2.0 s without it and 2.8 s with it (OpenFOAM 7.9 s; records
    `gs-dfsph__S1__f2@sphgoal-fe`, `@sphgoal-fgE`).
12. Particle noise near the edges of a box immersed in fluid at rest (fixed or articulated paddle): with the projected
    boundary gradient, the viscous dissipation readout of a fluid at rest was 20 times that of the same cup without the
    box, concentrated at its edges and corners, and the isotropic pressure tripled it (0.022 J and 0.059 J over 1.5 s,
    receipts `articulation__mx{D,C}_stir0`). With the full boundary gradient of finding 2 it is 2.0e-4 J
    (`articulation__fe_stir0`).
13. Velocity noise under gravity. With the projected boundary gradient, a closed column at rest at nu = 1e-3
    dissipated 1.6e-4 W per 0.3 kg whatever the warm start of the density solve or the substep (receipts
    `decay__ws*_rest_g9.81`): the projection of finding 2 kept feeding the noise. With the full gradient the noise of
    the column decays (rms speed 2.9 to 0.6 mm/s over 0.9 s, `logs/fi_cn_nu1e-3.log`). In FluidBench S1 the initial
    lattice releases 2.5e-3 J as it compacts in the first control step (`timeseries.csv` of the S1 runs).

14. Particles ejected from the corner of the cup in FluidBench S1 at the FluidBench time step (dt 0.31 ms, 52
    substeps per 16 ms control step): particles in the edge between the floor and the wall of the cup, 0.5 mm above
    the floor and up to 1 mm past the wall surface, leave it at 3 to 14 m/s within a control step, some through the
    wall, also after the stirrer has stopped (`logs/eject_trace.log`, dev/eject_trace.py). Particle speeds above 2 m/s
    were recorded at 99 to 114 control steps over 10 s at dt x1, 43 at dt x2 and none at x4 and x8
    (`timeseries.csv` of `@sphgoal-nl1-it100-dt*`, `@sphgoal-ch1-dt*`), which fits a position error at the edge that
    the density solve corrects within one substep, at a speed that grows as the substep shrinks (not verified). The
    wall clearance guard follows the nearest faced
    surface of a link; guarding a second surface at concave edges made it worse (speeds up to 59 m/s, 3.2 % of the
    liquid out of the domain, `@sphgoal-g2p-dt*`) and is not used. Open.
15. Chebyshev acceleration of the density solve and the particle noise. Stopped at the FluidBench tolerance (average
    density error 0.05 %), the accelerated solve leaves a closed column of water at rest with 2.9e-4 J of kinetic
    energy after 1 s, where the plain solve leaves 1.3e-4 J; with the tolerance ten times tighter both leave 1.4e-4 J
    (`logs/ch1_cn.log`, `logs/fe_cn.log`, `logs/kp_cheb_tight_cn.log`, `logs/kp_plain_tight_cn.log`): the plain
    Jacobi iterate is already converged in this respect at the tolerance, the accelerated one is not. Accelerating the
    density solves of the later coupling passes as well broke the viscosity-pressure coupling of a viscous scene (33
    passes per substep and 3029 unconverged substeps against 9.5 and none, `articulation__ch1_stir0`); with plain
    later passes it converges as before (`articulation__kp_stir0`). The acceleration therefore applies to the first
    density solve of a substep, which stops at a tenth of the tolerance.

## Measured validation (development snapshots, to be repeated on the committed revision)

Full boundary gradient at a boundary spacing of 0.5 particle sizes, isotropic pressure off (receipts tagged `fe_*`), and
the same with `isotropic_pressure=True` (`fi_iso_*`, `fgE_*`).

| Case | Default | With isotropic pressure | Receipt |
| --- | --- | --- | --- |
| V2 Couette dx 3.5 mm, depth 0.05 / 0.08 m, t 0.05-1 s | torque +10-27 % / +26-48 % | 0.3-3.4 % / 2-10 % | `couette__fe_couette_dx0.0035`, `couette__fi_iso_couette_dx0.0035` |
| V4 mode decay nu = 1e-3, dx 2.5 mm | rate -23 %, dissipation readout 0.995 of the energy loss | -23 % | `decay__fe_decay_nu1e-3_dx0.0025`, `decay__fgE_decay` |
| V5 spin-down nu = 1e-2, dx 5 mm | decay rate -11.6 %, omega rms error 2.5 % | -17 %, 3.0 % | `spindown__fe_spin_arm5e-4`, `spindown__fi_iso_spin_arm5e-4` |
| V6 momentum, nu = 1e-3 | 9.2e-6 / 9.0e-6 per s (limit 1e-4) | | `momentum__fe_mom_nu1e-3` |
| V8 plank of density 500, dx 3.5 mm | draft -14 %, fluid load +2.6 % of the weight, tilt 0.5 deg | | `floating__fe_plank500_dx0.0035` |
| Contact, sphere of density 2000 on the floor | fluid + contact = weight within 0.4 %, buoyancy -23 % (no fluid under the contact cap) | | `contact__fe_contact` |
| V10 paddle at rest | dissipation 2.0e-4 J over 1.5 s | | `articulation__fe_stir0` |
| Water at rest, closed column | kinetic energy decays (3.3e-4 to 1.3e-4 J in 0.9 s) | grows (2.3e-3 to 3.7e-3 J); at nu = 1e-3 held at 3 mm/s rms | `logs/fe_cn.log`, `logs/fg_cn_full50_iso.log`, `logs/fi_cn_iso_nu1e-3.log` |

Earlier snapshots (projected boundary gradient, development record): V2 2-9 % at depth 0.05 m and dx 5 mm
(`couette__g3_pbs_dx*`), V4 -39 %, -17 %, -13 % at dx 3.5, 2.5, 1.75 mm (`decay__face2_decay_nu1e-3_dx*`), V10 hold
joint forces vs readout within 1 % (`articulation__g3_hold`), stir joint torque vs fluid torque 1.4-3.5 %
(`articulation__mid2_stir`).

FluidBench S1 (stirred water, f2, R2, 19172 particles; OpenFOAM spin-down time 7.86 s and steady angular momentum
7.86e-4; the FluidBench usability bar asks for 30 % and 20 %):

| Configuration | Status | tau_sd (s) | Lz steady | RTF (output run) |
| --- | --- | --- | --- | --- |
| default, dt 0.31 ms (52 substeps / 16 ms) | 0.1 % of the liquid left the domain at 4.4 s | 1.97 | 9.38e-4 | 0.033 |
| default, dt x2 / x4 / x8 | ok | 1.73 / 1.66 / 1.66 | 9.09e-4 / 8.99e-4 / 8.86e-4 | 0.057 / 0.086 / 0.137 |
| isotropic pressure, dt 0.31 ms | a particle reached 20 m/s at 0.16 s | 2.78 | 8.08e-4 | 0.026 |

Records `gs-dfsph__S1__f2@sphgoal-fe*`, `@sphgoal-fgE` in FluidBench `results/index`. No configuration meets the
spin-down bar: in water the spin-down is set by the numerical friction of the walls and of the pressure pairs, and by
the particle noise.

## Performance (M5)

Profiles of FluidBench S1 f2 R2 (19172 particles, 45280 boundary particles, one environment) with
`adapters/genesis/tools/dfsph_profile.py`; receipts in FluidBench `results/profiles/sphgoal/`. Times are per substep of
the plain (uninstrumented) windows, two windows each; `--kernel-profiler` adds the GPU time of every task.

Where the time goes (`nl3_kp_dt1`, GPU time per substep at dt 0.31 ms): viscosity geometry 1.9 ms (its two boundary
passes over the ghost columns, 0.95 ms each), one Jacobi iteration of the density solve 0.165 ms (two neighbor loops of
67 and 40 us, the rest in small serial and coupled-link tasks of about 6 us each), density and factor 0.64 ms. Over a
full S1 run the density solve takes 47 iterations per substep at the FluidBench tolerance (average density error
0.05 %), 94, 114 and 170 at time steps x2, x4, x8 (records `@sphgoal-nl1-it100-dt*`); with the default cap of 10
iterations those solves stop unconverged at the larger steps (`fe_coarse_dt8`: all 882 density solves).

| Candidate (SPH_SOLVER_GOAL.md) | State | Evidence |
| --- | --- | --- |
| 1 boundary geometry per fluid particle and link | implemented: gradient, wall tensor and clearance per boundary slot, computed once per substep and read by every iteration | solver design (M2) |
| 2 slot allocation of the hash grid | measured, not changed: reordering takes 0.07 ms per substep (1 %) | `fe_fine` |
| 3 host readback in the iterations | proposed: one scalar per iteration, synchronization 16 us per call | `fe_fine` (`sync_s_per_call`) |
| 4 static and moving boundary indexing | proposed: the boundary update takes 0.02 ms per substep | `nl3_kp_dt1` |
| 5 cached neighbor indices | implemented, validated, measured: 9.0 to 6.6 ms per substep at dt x1, 12.8 to 8.9 ms at dt x8 (1.36x, 1.44x) | `fe_coarse_dt*`, `nl1_coarse_dt*`; equivalence `momentum__nl1_mom_nu1e-3`, `decay__nl1_decay_nu1e-3_dx0.0025`, `couette__nl1_couette_dx0.0035` |
| 6 reorder and copy traffic | measured, not changed: under 2 % | `fe_fine` |
| 7 preconditioner reuse | proposed | |
| 8 block reductions | implemented, measured: the link wrench accumulates in the pressure acceleration pass, and only particles with an error add it to their environment; the predicted-source task of an iteration went from 67 to 50 us, a substep at dt x8 from 31.1 to 28.6 ms (iteration cap 100) | `nl3_kp_dt1`, `it1_kp_dt1`, `nl3_it100_coarse_dt8`, `it1_it100_coarse_dt8` |
| 9 independent stop of environments | proposed | |
| 10 prescribed-boundary path | proposed | |
| packed moment matrix of the viscosity correction | measured, not adopted: 1.99 to 1.93 ms | `nl2_fine_dt1` |
| compile-time constants in the iteration kernel | implemented, measured: statements outside the loops of a kernel run as serial GPU tasks of 6-7 us; one fewer per iteration, 28.6 to 28.1 ms per substep at dt x8 | `it2_kp_dt1`, `it2_it100_coarse_dt8` |
| Chebyshev semi-iterative density solves | implemented, validated, measured: bound 0.98 (best of 0.95 to 0.998 at every time step), on the density solves of the first two coupling passes, which work to a tenth of the tolerance (finding 15); 21.5, 56.1 and 69.3 iterations per substep at dt x1, x4, x8 against 51.1, 123.2 and 166.8, RTF 0.041, 0.099, 0.155 against 0.027, 0.048, 0.076, all solves converged; paddle at rest (nu = 1e-3) 57 s against 130 s with 6.9 coupling passes per substep and none unconverged | `ch3_dt*`, `cheb0_dt*`, `cheb*_dt*`; validation `*__ch3_*`, `logs/ch3_cn.log` |
| warm start of the density solve at 0.75 or 1.0 of the last pressure instead of 0.5 | measured, not adopted: 45 to 41 and 35 iterations per substep at dt x1, but water at rest stays noisier (closed column kinetic energy 3.7e-4 to 2.6e-4 J over 0.9 s against 3.3e-4 to 1.3e-4 J) | `gs-dfsph__S1__f2@sphgoal-ws*`, `logs/ws1.0_cn.log` |
| 4 instead of 8 ghost points per wall column | measured, not adopted: 6.65 to 6.08 ms per substep at dt x1, mode decay rate -24.0 % against -22.9 % | `g4_coarse_dt1`, `decay__g4_decay_nu1e-3_dx0.0025` |

## Open items

- V1 (Stokes first problem) and V3 (pipe flow) drivers: the sliding-floor trough is limited by the closed-cavity return
  flow and the free-surface pressure resolution.
- Pressure friction (finding 11) without heating: the isotropic pressure is accurate for viscous flows but heats water.
- V4 decay deficit (-23 % at dx 2.5 mm), V8 draft offset (-14 %), buoyancy of a sphere resting on the floor (-23 %).
- V12 acceptance: GPU atomics make two runs from the same state differ, so a restart is compared with a second replay
  and with a restart without warm starts. In three runs the first step after a restart departed from the reference by
  7.8e-5 and 4.4e-5 (second replay 8.0e-5 and 5.2e-5, no warm start 3.3e-3), and once by 2.5e-4 (second replay
  3.8e-5, no warm start 1.9e-3); receipts `state__pt3_state_{a,b}`, `state__fe_state`. The criterion needs the
  spread of the replays over several runs.
- V9, V11, S1 spin-down accuracy, M5, M6.
