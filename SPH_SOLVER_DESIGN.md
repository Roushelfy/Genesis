# DFSPH solver with implicit viscosity and strong rigid coupling: equations and acceptance

Companion of [SPH_SOLVER_GOAL.md](SPH_SOLVER_GOAL.md). This file freezes the discrete model, its residuals, the stage
order of a substep, the units, the validation cases with their acceptance criteria and the performance measurement
boundaries (milestone M0). Implementation status and evidence live in [SPH_SOLVER_STATUS.md](SPH_SOLVER_STATUS.md).

## 1. Notation and units

| Symbol | Meaning | Unit |
| --- | --- | --- |
| `d` | `particle_size` option (particle diameter) | m |
| `H` | support radius of the cubic spline kernel, `H = 2 d` | m |
| `W(r)`, `W'(r)` | cubic spline kernel and its radial derivative (`W' <= 0`) | 1/m^3, 1/m^4 |
| `grad W_ij` | `W'(r_ij) e_ij`, with `x_ij = x_i - x_j`, `r_ij = abs(x_ij)`, `e_ij = x_ij / r_ij` | 1/m^4 |
| `V0` | rest volume of a fluid particle, `particle_volume_factor * d^3` | m^3 |
| `rho0_i` | rest density of the material of particle `i` | kg/m^3 |
| `m_i` | particle mass, `rho0_i V0` | kg |
| `Psi_b` | volume of boundary particle `b` (sampling-density weight `1 / sum_b' W_bb'` over its own boundary) | m^3 |
| `nu` | kinematic viscosity, the `viscosity` field of `gs.materials.SPH.Liquid` | m^2/s |
| `mu` | dynamic viscosity, `mu = rho0 nu` (canonical internal coefficient) | Pa s |
| `h` | substep time step | s |
| `u` | generalized velocities of the rigid solver (one vector per environment) | rad/s, m/s |
| `M_r` | rigid generalized mass matrix used by the rigid solver for its smooth acceleration (includes armature and, for the implicit integrators, `h` times the implicit damping, see `func_compute_mass_matrix`) | kg, kg m^2 |
| `J_b` | Jacobian mapping `u` to the velocity of boundary particle `b`: `v_b = J_b u = cd_vel_L + cd_ang_L x (x_b - c_L)` for its link `L` (`c_L` the root COM point of the spatial velocity convention, see `func_vel_at_point`) | - |

The material option `viscosity` stays a kinematic viscosity (m^2/s, water about 1e-6). The solver converts it once per
material to `mu = rho0 nu` and only uses `mu` internally, so that materials of different densities exchange a consistent
viscous stress.

Normalized density of a fluid particle: `rhohat_i = sum_j V0 W_ij + sum_b Psi_b W_ib` (the sums include `j = i`).

## 2. Joint velocity space

Per environment, the unknown velocity is `w = (v_1, ..., v_N, u)`: all fluid particle velocities and all rigid
generalized velocities. The mass operator is `M = blockdiag(m_1 I_3, ..., m_N I_3, M_r)`.

Rigid links fall in three classes, each handled by the same equations:

- fixed link (no degree of freedom): `J_b = 0`, `v_b = 0`.
- prescribed link: its degrees of freedom are driven by the user (velocity or position setters every step, velocity or
  position control with high gains, or large armature). They stay rigid unknowns. Their response to a fluid impulse
  through `M_r^{-1}` is what the rigid solver itself would produce, which vanishes as the drive stiffens. The fluid
  load on such a link is reported as its actuator-equivalent load `tau_fluid` (section 8).
- free or articulated dynamic link: rigid unknowns, response through `M_r^{-1}`. Joints are exact in generalized
  coordinates. Contacts, joint limits and equality constraints act through the rigid constraint solver (section 6).

## 3. Discrete operators

### 3.1 Velocity divergence (DFSPH constraints)

`C_i = rhohat_i - 1` and its rate `(G w)_i = sum_j V0 (v_i - v_j) . grad W_ij + sum_b Psi_b (v_i - J_b u) . grad W_ib`.

`G` has the fluid blocks `G_ii = sum_j V0 grad W_ij + sum_b Psi_b grad W_ib`, `G_ij = -V0 grad W_ij`, and the rigid
block `G_iu = -sum_b Psi_b grad W_ib^T J_b`.

### 3.2 Pressure impulse

A pressure solve applies `w <- w + M^{-1} G^T lambda`. With `lambda_i = -h m_i kappa_i`, where `kappa_i >= 0` is the
`pressure_density2_norm` of the current code (pressure over squared density, divided by the rest density):

- fluid: `dv_i = -h sum_j V0 (kappa_i + (rho0_j / rho0_i) kappa_j) grad W_ij - h kappa_i sum_b Psi_b grad W_ib`
- rigid: `du = M_r^{-1} P_u` with `P_u = h sum_i m_i kappa_i sum_b Psi_b J_b^T grad W_ib`.

The boundary term on the fluid and `P_u` are equal and opposite (per pair `(i, b)` the impulse on `b` is
`+h m_i kappa_i Psi_b grad W_ib`), so the pair exchange conserves linear and angular momentum exactly.

### 3.3 Viscous dissipation

Rayleigh dissipation function, summed over unordered neighbor pairs:

    R(w) = 1/2 sum_(i,j) k_ij ((v_i - v_j) . e_ij)^2 + 1/2 sum_(i,b) k_ib ((v_i - J_b u) . e_ib)^2
    k_ij = 2 (3 + 2) mu_ij V0 V0 (-W'(r_ij) r_ij) / (r_ij^2 + eta^2)
    k_ib = 2 (3 + 2) mu_i  V0 Psi_b (-W'(r_ib) r_ib) / (r_ib^2 + eta^2)
    eta^2 = 0.01 H^2,   mu_ij = 2 mu_i mu_j / (mu_i + mu_j)   (zero if either is zero)

The viscous force is `f = -grad_w R = -D w` with `D = hessian(R)`. Properties:

- `k >= 0`, so `R >= 0`: the operator is dissipative and `D` is symmetric positive semi-definite (PSD).
- Pair forces act along `e_ij` and are equal and opposite, so linear and angular momentum are conserved, including the
  exchange with rigid links (the rigid generalized force is `J_b^T` of the pair force on `b`).
- Null modes: rigid motions of the whole coupled system (every `(v_i - v_j) . e_ij` vanishes for
  `v = a + omega x x`). A fixed or prescribed wall removes them from the fluid.
- On a divergence-free field the fluid part approximates `mu laplacian(v) = div(2 mu D(v))` for constant `mu` (the
  `2 (d + 2)` projected Laplacian already used by the explicit term), and the harmonic interface mean makes the shear
  stress continuous across a viscosity jump.
- Wall: boundary particles complete the kernel support of a fluid particle near a rigid geom, and their velocity is the
  rigid velocity, which gives a no-slip condition on fixed, prescribed and dynamic walls alike.

### 3.4 Implicit viscosity (backward Euler)

Given a velocity `w*` (fluid and rigid), the viscous step solves

    A w = M w*,   A = M + h D,

which is symmetric positive definite (SPD) since `M` is SPD and `D` PSD. Energy decreases monotonically for a closed
system (`w^T M w <= w*^T M w*`), so the explicit restriction `h < 0.01 H^2 / nu` is removed. The solve is matrix-free
conjugate gradient (CG) on the joint vector, preconditioned by the inverse 3x3 diagonal blocks of `A` on the fluid and
`M_r^{-1}` (the rigid factor) on the rigid block. Residual: `r_visc = M w* - A w`, measured as
`max_env sqrt(sum r^T M^{-1} r) / sqrt(sum w*^T M w*)`.

The viscous impulse exchanged with the rigid unknowns is `P_visc = M_r (u - u*)` (the rigid row of the solved system).

## 4. Substep stage order

Positions `x^n`, `q^n`, velocities `v^n`, `u^n`. All stages read the positions at `n`.

1. Rigid forward dynamics (`kernel_substep_dynamics`): `M_r(q^n)`, its factor, smooth force `tau_s`, smooth
   acceleration `a_s = M_r^{-1} tau_s`.
2. Boundary particles at `q^n` with Jacobians `J_b(q^n)`, fluid neighborhoods, `rhohat`, DFSPH factors.
3. Divergence solve on `(v^n, u^n)` (section 5, velocity level).
4. Explicit accelerations: fluid `v* = v + h a_ext` (gravity, surface tension, force fields, micro-rotation exchange),
   rigid `u* = u + h a_s`.
5. Coupled block iteration `k = 1, ..., K` (section 5.2) of the implicit viscosity solve and the density solve on
   `(v, u)` until both residuals pass.
6. The net fluid generalized impulse `P = M_r (u - u^n - h a_s)` enters the rigid solver as the smooth force
   `tau_s + P / h` (it updates `dofs.force`, `qf_smooth`, `acc_smooth` and the coupling wrench readout). Collision
   detection, constraint assembly and the constraint solve then see the fluid load.
7. Fluid-rigid constraint consistency (section 6). Then the rigid solver integrates to `(q^{n+1}, u^{n+1})` and the
   fluid advects `x^{n+1} = x^n + h v^{n+1}`.

Scenes without dynamic coupled rigid links skip the rigid parts of stages 1 and 6 to 7 (fixed walls only need boundary
positions, prescribed links only need `J_b u`).

## 5. Pressure solves with rigid response

### 5.1 Jacobi iteration

The Schur complement `S = G M^{-1} G^T` is solved by relaxed Jacobi on `kappa` (the DFSPH iteration). One iteration:

1. fluid pressure acceleration `a_i` from `kappa` (section 3.2) and, in the same neighbor pass, the per-link wrench of
   the boundary reaction,
2. rigid response `du = M_r^{-1} P_u(kappa)` through the rigid factor, and from it the link spatial velocity change, so
   every boundary particle gets `a_b = J_b du / h`,
3. predicted source `s_i = sum_j V0 (a_i - a_j) . grad W_ij + sum_b Psi_b (a_i - a_b) . grad W_ib`,
4. residual `res_i = src_i - h^k s_i` and update `kappa_i <- max(kappa_i - 0.5 res_i alpha_i / h^k, 0)`
   (`k = 1` divergence, `k = 2` density).

`du` is recomputed from the current `kappa` at every iteration, never accumulated, so the physical impulse is applied
once, after convergence. The diagonal `alpha_i^{-1}` gains the rigid term
`m_i (abs(F_iL)^2 w_lin_L + abs(T_iL)^2 w_ang_L)` per dynamic link `L` neighbor of `i`, with
`F_iL = sum_{b in L} Psi_b grad W_ib`, `T_iL = sum_{b in L} (x_b - c_L) x Psi_b grad W_ib` and the link inverse weights
of the rigid solver. The diagonal only sets the convergence rate, the fixed point is the exact coupled solution.
Residual: average over active particles of `max(-res_i, 0)` (density) or of the divergence rate (divergence), as now.

### 5.2 Viscosity-pressure block iteration

Stage 5 alternates (a) the viscosity solve `A w = M w* + G^T lambda` with the pressure impulse of the previous density
solve held fixed, and (b) a density solve on top of it. The block iteration stops when the viscous residual of the
combined velocity `r_visc = M w* + G^T lambda - A w` and the density residual both pass, or at the iteration cap. The
reported residual history contains both at every outer iteration. One pass is enough only when the residual history
shows it.

## 6. Constraints of the rigid solver

After stage 6 the constraint solve may change the rigid velocity of coupled links by `du_c` (contact, limit, equality).
The fluid assumed `u`, so the coupled residual is `r_fr = max_b abs(J_b du_c)` over boundary particles of dynamic
links, relative to the boundary velocity scale. If it exceeds the tolerance, stages 3-6 repeat with `du_c` held as an
external impulse on the rigid unknowns, then the constraint solve repeats, until `r_fr` passes or the cap is reached.
Iteration counts, residuals and cap exhaustion are recorded.

## 7. Free surface and pressure model

The pressure is unilateral (`kappa >= 0`): DFSPH corrects compression only, and particles with fewer than 20 neighbors
(free surface) take no divergence correction, as now. Hydrostatic pressure, buoyancy and the load on a container are
outcomes of the coupled solve, with no separate buoyancy force.

## 8. Load readout

Per environment and link, the fluid load of a substep is the pair sum of the pressure and viscous impulses on its
boundary particles divided by `h`, given as a force in world frame (N) and a torque about the link center of mass in
world frame (N m). The public readout averages it over the substeps of the last `scene.step()` (control step). It holds
the fluid load only (gravity, actuators and rigid contact are separate). For prescribed links it is the
actuator-equivalent load. The generalized form `tau_fluid = J^T` of the same impulses is available per degree of freedom.

## 9. Validation cases and acceptance

Tolerances below are the frozen acceptance criteria. An unconverged substep (iteration cap reached with a residual above
tolerance) invalidates the sample it belongs to.

| Id | Case | Observable | Acceptance |
| --- | --- | --- | --- |
| V1 | Stokes first problem: plate impulsively moved tangentially under a deep layer | velocity profile `U erfc(z / (2 sqrt(nu t)))` and wall shear force `A mu U / sqrt(pi nu t)` | profile L2 error <= 10 % at the finest of 3 resolutions, decreasing with resolution; force within 15 % after `t > 4 H^2 / nu` |
| V2 | Taylor-Couette: prescribed inner cylinder, fixed outer cylinder | steady `u_theta(r) = A r + B / r`, torque `4 pi mu L Omega R1^2 R2^2 / (R2^2 - R1^2)` | profile error <= 10 %, torque within 15 % (end effects included in the reference by a refined run) |
| V3 | Gravity-driven pipe flow (Hagen-Poiseuille) of a viscous column | profile `g (R^2 - r^2) / (4 nu)` | centerline velocity within 10 %, converging with resolution |
| V4 | Viscous mode decay in a closed box | kinetic energy decay rate vs `nu k^2` of the lowest no-slip mode | rate within 10 %, for `nu` spanning 1e-6 to 1e-2 m^2/s, and for `h` above the explicit limit |
| V5 | Free spin-down of a dynamic cylinder in fluid (two-way viscous torque) | angular velocity decay vs the quasi-steady Couette torque over its inertia | rate within 20 % |
| V6 | Closed isolated system: fluid blob and free link, zero gravity, no fixed geom | total linear and angular momentum | drift below 1e-4 of the initial magnitude per second (fp32) |
| V7 | Hydrostatics in a fixed container | pressure profile `rho g (H - z)`, container load = fluid weight | load within 2 %, pressure within 5 % below the top layer |
| V8 | Floating box (densities 0.5 and 0.1 of the fluid) | equilibrium draft `(rho_b / rho_f) height` | draft within 10 %, stable (no blow-up) for the light box |
| V9 | Stokes settling sphere, high viscosity | terminal velocity `2/9 (rho_s - rho_f) g R^2 / mu` with the container-wall (Faxen) correction | within 20 % at the finest resolution, converging |
| V10 | Hinged plate driven in fluid (articulated, joint torque readout) and a velocity-driven arm stirring a cup | joint torque readout vs the coupling wrench, energy balance (external work = kinetic energy change + dissipation) | balance closes within 5 % of the external work |
| V11 | Thin-wall cup (4 mm wall) at R0 and R1, offsets and orientations | leakage, start-up velocity from rest | no particle crosses the wall, start-up max velocity below 0.05 m/s from rest |
| V12 | State: snapshot/restart, reset, selected-environment reset | bitwise equal continuation on the same backend | exact |
| V13 | FluidBench S1 (stirring and spin-down), S2 (pouring), S3 (mixing) | FluidBench metrics against its frozen references | FluidBench tolerances |

## 10. Performance measurement boundaries

- Warmed solver-only time: the scene step loop after compilation and a warm-up window, without readout, synchronized
  before and after (same convention as the FluidBench timing runs: median of 3 repeats).
- Stage breakdown with synchronized timers (instrumented, separate run): neighbor/hash, density and factors, divergence
  solve, explicit forces, viscosity solve, density solve, outer coupling, rigid work, reductions and host
  synchronization, output.
- Reported per configuration: real-time factor (RTF, simulated over wall seconds), iterations and residuals per solve,
  build and compile time, peak GPU memory, and for batches environment-transitions per second and per-environment RTF.
- Hardware of record: NVIDIA RTX PRO 6000 Blackwell (cluster partition rtx-mid), fp32.
- Representative configuration: FluidBench S1 f2 at R2 (dx 2.5 mm, 19 172 particles) with water (`nu = 1e-6`),
  accuracy-qualified when V1-V8 pass at that resolution. Generality sweep: particle counts (R1-R3), boundary fraction
  (cup only, cup and stirrer, many floating links), viscosity (1e-6 to 1e-2 m^2/s), link count, 1 to 64 environments.
