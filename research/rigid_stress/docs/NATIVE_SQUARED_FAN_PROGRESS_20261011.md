# Squared contact integration and short surface segments

The subsequent explicit shared-support candidate and its per-original-wrench
oracle checks are recorded in
[the shared-support checkpoint](NATIVE_SHARED_SUPPORT_PROGRESS_20261011.md).

This follows [the patch/refinement checkpoint](NATIVE_PATCH_REFINEMENT_PROGRESS_20261011.md),
base source `897b5172fca34125046e0c1c6544e3e2ba742bc8`. The active
nonconvex goal remains incomplete. This checkpoint selects a validated local
integration repair and records the next live producer incompatibility. It
does not establish large-batch throughput or an optimization plateau.

## Selected contact integration

The native surface-fan rule now stores each closest point's barycentric
coordinates directly. A subtriangle's area is its opposite barycentric
coordinate times the parent triangle's area. This avoids the extra Cartesian
round-trip that gave a degenerate edge fan a spurious positive area. Small
positive coordinates from the original geometric calculation remain visible;
no area/profile threshold discards them.

Each unit-square Gaussian coordinate is squared. Both change-of-variable
Jacobians enter the positive area weight. The quadrature count remains Q10,
and each boundary face still has three finite subtriangles. Physical radius,
Gaussian width `0.45R`, compact squared taper, eight shared edge subdivisions,
true owner normals, circular local Coulomb cones and signed P2 scatter are
unchanged. The `coordinate_power=1` factory option retains the previous rule
for offline comparisons. Shared elasticity operators and material quadrature
are unchanged; this is local contact integration, not a high-mesh or material
quadrature convergence experiment.

The plane-segment predicate now includes an operand-scale rounding term:

```
plane_budget = angular_tolerance * segment_length
             + 32 * machine_epsilon * (|anchor| + |target| + |face_origin|)
```

The angular, barycentric and triangle-adjacency tests retain their existing
budgets. No shortcut crosses a missing triangle or connects distinct vertex
identities. Squared coordinates create approximately micrometre-long segments
on the egg. The previous length-only plane budget could be smaller than the
rounding error of coordinate subtraction/dot products, rejecting a valid
same-face segment and taking a millimetre-scale port detour. Independent
geometry/profile comparisons exposed that branch error. The repair changes
the internal geometric predicate; final numerical acceptance is unchanged.

## Validation and retained failures

The final source passes the complete targeted native suite on the allocated
Xeon Platinum 8562Y+ and RTX PRO 6000 Blackwell Server Edition, driver
595.71.05, container `1_26`, Genesis 1.4.3, Quadrants 1.3.3, FP64:

| Final source | Result | Wall time |
|---|---:|---:|
| CPU | 45 passed | 333.24 s |
| CUDA | 45 passed | 469.44 s |

Before/after source hashes match. The suite includes concave/cavity and narrow
gap topology, plane holes, source ambiguity, offset/friction/free moments,
moving contacts and independent environments. Eight saved full-shell cases
compare independent CPU geometry/profile, constrained traction fitting,
signed nodal loads, every stress value, the global peak and full equilibrium
including gauge rows. Ticks 147, 427 and 454 exercise both integration rules;
ticks 594 and 1076 use squared coordinates. Each saved case has three
environments with actual contact friction `mu_original`, `0.85` and `1.2`.

Final normalized force/moment acceptance remains `1e-8`; complete equilibrium
remains `max(1e-11 N, 1e-8 ||rhs||)`; stress remains `rtol=1e-4`,
`atol=1e-3 Pa`. Internal fitting retains `9e-9`. There is no requirement to
drive the internal iteration to `2e-12` after its actual final budgets pass.

The earlier squared-only full suite is preserved: CPU 42 passed/3 failed and
CUDA 43 passed/2 failed. Those failures were independent profile assertions
in the older saved inputs; the final budgets were not loosened to pass them.
An isolated CPU failure, kernel query diagnostics and the eight CPU/eight
CUDA plane-predicate candidate checks are also retained. The experimental
warp's earlier 10 passed/2 failed source epoch has the same profile issue;
its corrected-source evaluation remains outstanding.

Reproduce the final suite on an allocated node with an explicit fresh data
directory:

```bash
RIGID_STRESS_RUN_DIR="$RIGID_STRESS_DATA_ROOT/runs/<id>/cpu" \
  bash research/rigid_stress/scripts/check_nonconvex_contact.sh cpu
RIGID_STRESS_RUN_DIR="$RIGID_STRESS_DATA_ROOT/runs/<id>/cuda" \
  bash research/rigid_stress/scripts/check_nonconvex_contact.sh cuda
```

## Longer live trajectory and the plane producer

The squared-coordinate B=32, seed 623101, GJK/direct-patch trajectory passes
the old tick594 case and completes 710 of 1280 requested steps. Five preceding
CPU captures check 160 complete fields; six environment resets occur before
the failure (three had occurred at the last CPU capture at tick640).
The experimental warp B=16, seed 623001 run passes tick1076 and completes
1160 of 1280 steps. Neither is a completed acceptance or throughput result.
All environments and failures remain in the evidence.

At tick710, env5/contact0, R=0.004364287080690719 m and
mu=0.6008812824796964, powers two/three/four and evaluation limits
128/512/2048 retain normalized force error near `0.00060889368`. An independent
outer model includes every true triangle intersecting the Euclidean R-ball,
a superset of the intrinsic footprint. Its separating ray is
`[0.0690705348999334, -0.6986435838235213, 1, 0, 0, 0]`:
the normalized target dot is `-0.0005933118896052636`, and the minimum exact
circular-Coulomb dual margin is `0.00014506820049453584`. This certifies an
incompatible force on that continuous support; more sampled fit iterations
cannot repair it. A coarse outer polygon-cone LP's feasible status violates
the actual circular cones and is not acceptance.

Capturing the real frame immediately before recovery identifies geom0
(ground plane) and geom24/link12 (egg), witness method `NORMAL_OFFSET`. The
world normal is vertical, to roundoff. The reference/collision vertices agree
within `5.51e-17 m`, the saved/world witness agrees within `4.33e-18 m`, and
the saved/reference normal agrees within `1.13e-16`. The egg witness is
`115.71735672095 micrometres` from its actual supporting plane in that
direction. Thus the frame transform is correct; this source is a plane
multi-contact input, not an EPA/direct-patch normal. Tight patch feature
alignment and retaining patch-witness normals both still fail at tick710.
Those controls retain all manifold points and are not selected fixes.

The native plane path detects contacts on slightly rotated geometry and
backmaps the plane normal and witnesses. A backmapped material vertex need
not be the unperturbed plane's supporting feature. Further producer/source
work must preserve a declared physical load model and all original loads;
there is no force clipping, radius enlargement or ordinary-contact unsupported
exception in this checkpoint. The current native continuous certificate does
not detect every common polar, so this live error still reports unresolved
numerical fitting even though the independent audit proves incompatibility.
Improved diagnostic coverage and a legitimate producer/aggregate-source
remedy remain required.

## Delivery limits

The complete capsule is
[online-checkpoint-v6](../evidence/20261011-nonconvex/online-checkpoint-v6/manifest.json).
It retains successful and failed source epochs, fixture extraction, all live
records, original snapshots, offline diagnoses and experimental driver errors.
The capsule contains 833 artifacts (24,897,811 bytes), with manifest SHA256
`9bffa97cfb3b7f9c961cd871a22e1917ce2857e08d37534911d5eb8e6fb3e562`.
Original and published artifact hashes were checked on allocated compute.
The source-replay development controls contain import/index/kernel-wrapper
errors; their successful geometry capture and independent frame audit are
identified separately and are not claimed as successful GJK replay.

The benchmark exposes the same explicit contact-patch option as the example
and labels the squared-coordinate law. After source/producer robustness, a
new matched baseline is still required. The earlier 3.887 frozen contact-stage
ratio concerns the unsquared warp candidate, not the selected law or rollout
throughput. General primitive/SDF/decomposition/aggregate association,
continuous/refinement diagnostics, broader lifecycle acceptance, cooperative
fit/scatter/commit optimizations and repeated four-scope B=1024/2048/larger
measurements remain actionable. The active goal is not complete.
