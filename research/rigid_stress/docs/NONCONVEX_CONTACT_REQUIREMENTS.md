# Nonconvex contact mapping: active implementation contract

User requirement, 2026-10-11. This extends GOAL.md and NATIVE_QUADRANTS_PLAN.md
after the previous low-shell milestone. The algorithm has an offline CPU FP64
prototype. Native Quadrants acceptance requires its own evidence.

The [2026-10-14 final report](NATIVE_NONCONVEX_RESULTS_20261014.md) and
[requirement audit](NATIVE_NONCONVEX_CLOSEOUT_20261014.md) now record native
implementation, CPU/CUDA numerical/lifecycle validation, matched performance,
output costs and verified publication. The contract below is preserved.

## Outcome

Support contact loads on valid nonconvex material boundaries: concave faces,
interior cavities, narrow gaps, adjacent surface branches, and edge/corner
contacts. Only reference geometry, material and mass are fixed. Contact
locations, counts, normals, friction, radii and symmetry remain unrestricted.
Smooth histories permit reuse, but contact birth/death, slip, branch changes
and partial resets must remain correct.

Add this to Genesis's existing optional rigid stress feature. Use Quadrants
for the complete production numerical path. Reuse the fixed-geometry elastic
operators, recovery solve, inertia relief, full/max output options and global
unaveraged peak calculation. Keep rigid dynamics rigid.

The Franka Panda gripping a full hollow egg remains the primary throughput
workload; add inexpensive nonconvex fixtures to exercise this extension.
Choose the implementation and tuning with engineering judgment. The contract
below fixes the physical and numerical properties, not a particular queue,
kernel layout or edge subdivision count.

## Contact mapping algorithm

1. **Preserve surface provenance.** Carry a reference boundary face and
   barycentric coordinates, plus the collision geom/pair or equivalent source
   identity. Keep the original rigid force application point. Map primitive,
   SDF and convex-decomposition witnesses to the actual reference boundary;
   an internal decomposition face is not a physical exterior face.
   Position and normal alone cannot distinguish all nonconvex branches.
   A bounded nearest-surface fallback is acceptable only when provenance and
   geometry make the selection unambiguous; otherwise report the ambiguity.

2. **Orient normals from material topology.** For a tetrahedral boundary face,
   record its owning tetrahedron and orient its normal away from that element.
   Do not use face-center minus link COM to infer material inside/outside.
   Support cavity surfaces and arbitrary input triangle winding.

3. **Select a finite footprint along the surface.** Use spatial grid/BVH
   queries only for candidate pruning. Starting at the witnessed position,
   propagate through actual surface adjacency up to a declared physical
   radius. Never create adjacency by merging nearby spatial positions on
   different sheets. Do not replace this with Euclidean radius plus normals:
   two nearby branches may have the same normal.
   Specify orientation/contact masks and any approximation as part of the
   footprint law. A plane shortcut must verify that its segment lies on
   actual triangles; a coplanar region can contain a hole.

4. **Preserve the original wrench under projection.** With original position
   p, force F, free moment M_p and reference anchor a, use

   $$M_a=M_p+(p-a)\times F.$$

   Match the resultant force and moment about the original p after mapping
   and after P2 nodal scatter. Do not silently discard the offset moment of
   tangential forces. Obtain actual rolling/torsional moments when enabled,
   or explicitly reject unsupported enabled modes; coefficients alone do not
   identify those moments.

5. **Fit physical vector tractions.** One reference law uses normalized
   positive area/profile weights w_q and solves

   $$\min_{\{f_q\}}\sum_q\frac{\|f_q-w_qF\|^2}{2w_q}$$

   subject to

   $$\sum_q f_q=F,\qquad \sum_q(x_q-a)\times f_q=M_a,$$

   $$f_q\cdot n_q\geq0,\qquad
   \|f_q-(f_q\cdot n_q)n_q\|\leq\mu(f_q\cdot n_q),$$

   where n_q is the true local inward material normal. The CPU prototype
   reuses WrenchFit's six-dimensional dual, Coulomb projection and Jacobian;
   scale moments by the physical radius for conditioning. Scatter
   $b_i=\sum_q N_i(x_q)f_q$ using signed P2 basis values. Positivity applies
   to physical compression, not to every equivalent nodal force.

6. **Make failure semantics explicit.** Distinguish ambiguous provenance,
   unresolved integration/distance, numerical fitting failure and physically
   incompatible wrenches. Refine local integration or distance when justified;
   do not jump to another branch, enlarge the physical footprint silently,
   relax the input force/friction, or substitute a singular point load.
   Multiple actual contact islands need separate sources/loads, or an explicit
   model for partitioning an aggregate wrench.

A point wrench does not uniquely specify the pressure distribution. Declare
the finite-footprint law and hold it fixed within matched accuracy/performance
comparisons. A useful reference profile is

$$\tilde w_q=A_q\exp[-d_q^2/(2(0.45R)^2)]
                 (1-d_q^2/R^2)_+^2,\qquad
w_q=\tilde w_q/\sum_j\tilde w_j.$$

## Distance implementation and throughput

The CPU prototype connects shared surface-edge sample positions by segments
contained in each triangle, then performs a radius-truncated shortest-path
query. Such paths give an upper bound on continuous intrinsic distance.
Acceptance therefore has an actual surface path of length at most R.
Coarse samples can overestimate distance and exclude valid footprint points;
they are not exact geodesics. Validated plane shortcuts remove this error on
fully covered straight paths.

Port refinement, local triangle unfolding or another validated intrinsic
method are implementation choices. Neither the prototype default s=4 nor
its s=32 experiment is a universal required production setting. Account for
distance/profile error separately from force/moment conservation.

Profile source association, candidate discovery, intrinsic distance, constrained
fit, scatter, solve, residual, global peak and history. Explore:

- Shared immutable topology, local geometry and operators across environments.
- Local candidate/frontier work instead of per-environment full-surface scans
  or full-surface distance clearing. Keep runtime data on the device.
- Plane/constraint-interior fast paths, fused small-matrix reductions and
  packing contacts needing iterative fitting or local refinement.
- Reuse of valid source faces, candidate neighborhoods and dual initial guesses
  during smooth motion, with verification and invalidation on contact/normal/
  radius/friction changes, topology version changes and resets.
- Precision and stopping tolerances chosen against the final wrench, friction,
  residual and stress budget, including correction/failure cost.

Measure matched rigid-only, recovery-only, live and policy aggregate transitions/s.
Rebuild the baseline after changing the footprint law. Retain the low-mesh,
performance-first workflow: necessary contact/same-mesh checks continue, while
expensive physical convergence is a later study. Keep progress and measured
decisions committed/pushed so another agent can resume.

## CPU evidence and minimum native checks

The offline audit used CPU FP64 on Xeon Platinum 8370C, frozen Genesis reference
460e87f56c6b4db02a56b99be5366872fbd92a9e. The CPU audit used NumPy 2.3.5,
SciPy 1.17.0, scikit-fem 12.0.2 and pygeodesic 0.1.11. The results below are
prototype evidence, not native GPU acceptance or physically converged stresses.
Raw results and figures remain in the separate CPU audit; this documentation
update publishes the algorithm contract and measured summary only.

| CPU check | Observed result |
|---|---|
| U inner-wall anchor | Old force-line selection moved 24.15 mm; witnessed mapping retained the inner wall |
| Material normals | COM rule flipped 128/800 U faces, 192/944 narrow-U faces and 48/240 cavity faces; owner rule flipped none |
| 1 mm gap, same-facing branches | Matched Euclidean mapping spilled 0.2354-17.5043% of sample-force magnitude onto the other arm; intrinsic mapping spilled zero |
| Random/contact motion | 60 random cases, 21 sliding frames, release and friction variants passed |
| Concave edge / cavity corner | Two- and three-face fits passed; 6 and 10 constrained evaluations respectively |
| Wrench after P2 scatter | Worst recorded normalized force/moment error approximately 1e-13 |
| Five independent scikit-fem comparisons | Worst relative peak error 3.61e-11; full equilibrium residual below 1.2e-10 |
| Curved-footprint approximation | Node-load change s=16 to s=32 was 0.594%; peak change s=4 to s=32 was approximately 1.62% |

The narrow-gap comparison keeps provenance, material normals, vector fitting
and friction identical; only the distance changes.

Port small fixtures into the normal native test path: inner wall/cavity,
same-facing narrow gap, edge/corner, offset point with friction, ambiguity,
moving contact/slip/release, rotated frames and partial resets. Compare
identical mesh/law/samples against CPU FP64, including signed nodal loads,
original wrench, local cones, complete equilibrium and full/max consistency.
Check independent environments and preserve the existing egg workload.
Avoid turning the archived experiments into a large mandatory test matrix.

Valid manifold material boundaries, resolved finite footprints and feasible
wrenches are the supported domain. Repair self-intersection/degeneracy before
mapping. Disconnected material domains require their own rigid-mode/connection
treatment; the CPU elasticity comparisons used connected material.
Sharp reentrant corners can have geometric stress singularities even under
finite-area loading. This contact repair does not imply universal peak-stress
mesh convergence; physical rounding or a declared averaged damage metric may
be needed.

For continuous-distance references, see
[CGAL surface shortest paths](https://doc.cgal.org/latest/Surface_mesh_shortest_path/index.html),
[Geometry Central geodesics](https://geometry-central.net/surface/algorithms/geodesic_distance/),
and [pygeodesic](https://github.com/mhogg/pygeodesic).
They are offline references, not runtime dependencies of the Quadrants feature.
