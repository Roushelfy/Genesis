# Automatic surface-to-shell preparation for DPR

**Active user goal, 2026-10-31.** Extend the existing mesh example so a user can supply a valid surface mesh
and select the thin-shell backend without preparing a Q4 NPZ manually. Deliver working code, focused
validation, measured costs and updated examples. This is the next implementation goal; the conversion is
not implemented by this document. Earlier throughput campaigns remain completed within their recorded scope.

## User-facing result

The intended commands, after implementation, are:

```bash
# Default hollow egg remains immediately usable with its existing assets.
QD_GRAPH=0 python -m examples.rigid.franka_egg_automatic_stress --backend shell --viewer

# Custom surface interpreted as the shell midsurface; thickness is in metres.
QD_GRAPH=0 python -m examples.rigid.franka_egg_automatic_stress \
  --backend shell --mesh model.obj --thickness 0.0005 --viewer

# Headless batch automatically uses maximum-only output.
QD_GRAPH=0 python -m examples.rigid.franka_egg_automatic_stress \
  --backend shell --mesh model.stl --scale 0.001 --thickness 0.0005 --envs 64
```

Without a supplied stress mesh, prepare and cache a compatible shell asset automatically. Continue accepting
an explicit matching Q4 NPZ. Reuse the current headless max/viewer full behavior, units and registration,
automatic Hertz radius and mature finite-pressure DPR recovery. Keep volume loading available and default;
selecting shell must be explicit. Existing default egg assets and authored-mesh workflows remain usable.

Treat the input surface as the midsurface and state this interpretation clearly. Thickness is a specified
material dimension, not inferred from a closed envelope. Selecting shell on a solid's outer surface changes
the physical model. Do not guess an internal cavity, extract a solid's medial surface silently, or confuse
metres of thickness with the input file's coordinate units. If an outer/inner-reference option is useful,
make its offset convention explicit and validate it rather than expanding the required interface.

## Preparation and integration

Start from the existing implementation at `aa5a7d093c31c3cd2974a87eef4c433534dc7003`:

- [asset preparation](../../../examples/rigid/stress_mesh_asset.py);
- [public example](../../../examples/rigid/franka_egg_automatic_stress.py);
- [surface display and material boundary](../../../examples/rigid/stress_field_view.py);
- [native shell guide](RUNNING_NATIVE_SHELL_DPR.md) and [usage](RUNNING_AUTOMATIC_DPR.md).

Prefer a dependable conversion into the existing Q4 shell representation over introducing a new elastic
solver. Triangle subdivision into conforming quads is one possible starting point, not a prescribed final
algorithm: examine element quality, mechanical accuracy and DOF growth before choosing. Preserve holes,
boundaries, creases and material connectivity. Quads must have valid Jacobians; consistently oriented
directors and side labels must support the current membrane/bending/shear operators and friction moments.
Automatic conversion must not weld nearby disconnected sheets across a gap or smooth away a narrow neck.

Cover closed orientable shells and orientable open surfaces with boundaries, including nonconvex examples.
Invalid, degenerate, self-intersecting or nonmanifold inputs need actionable diagnostics. If disconnected
components cannot be handled correctly by the current gauge/relief, report that limit explicitly; do not
give separate free components a single inappropriate gauge. Check the finite-thickness material boundary,
including open-edge caps, for invalid offsets/intersections before using its mass and inertia.

Keep collision geometry, material boundary, normals, pressure sides and stress display registered after
joint scaling/centering. Derive shell mass, COM and inertia from the actual finite-thickness material,
not the volume enclosed by a closed midsurface. Preserve both-side pressure, asymmetric moving contacts,
normal/friction loads and global interior/remote-hotspot recovery. A convex hull used for collision must
not silently seal a nonconvex cavity or bridge a gap while claiming matched surface contact.

Mesh import/conversion is one-time build work and may use appropriate CPU geometry libraries. It is not an
exception for adding a separate CPU elastic solver: production numerical shell preparation, assembly,
loading, recovery and batched runtime remain in the existing Genesis rigid feature in Quadrants. Reuse
existing APIs and preprocessing; avoid repeated meshing, host transfers or per-environment duplicate
assets during stepping. Cache identity/invalidation must account for source geometry and dependencies,
scale, conversion recipe/settings, and any thickness-dependent prepared quantities.

## Validation and performance

Use low-resolution models first and keep the checks proportionate to this feature:

- Exercise the real automatic entrypoint on a sphere/egg, a nonconvex U-shaped or narrow-neck shell, and an
  open curved or flat shell. Supply reproducible assets/recipes. Demonstrate headless max and actual viewer
  coloring; retain a shape-compatible Franka example rather than implying automatic grasp planning.
- Check units, topology, winding/directors, thickness geometry, material mass/inertia, registration and
  cache reuse. Reject unsupported inputs explicitly. Extend the existing mesh capability tests.
- Compare automatic preparation with reloading its saved Q4 asset for the same loads and compare native
  recovery with an independent CPU solve of that same discrete model. Check full stress tensors as well
  as maxima, normal/tangential eccentric loads, both sides, batch isolation and reset behavior.
- Separately check mechanical discretization quality: a small membrane/bending patch control and a bounded
  low-resolution refinement/reference comparison should expose locking, artificial mechanisms or badly
  shaped quads. Agreement between two solves on the same mesh does not establish spatial accuracy.
  Report this distinction; broad high-resolution contact/continuum campaigns are not a prerequisite here.
- Preserve current residual, force/moment, material and lifecycle acceptance. Reuse the current Q4/planar
  loading choices and their applicable checks; planar refers to the loading proxy, not new elasticity.

Profile before tuning. Measure cold conversion/build, warm-cache build, generated elements/DOF,
shared/per-environment memory, and native stress-module plus complete-scene max/full throughput at small
and useful large batches. Keep viewer/readback cost separate. Compare with an explicitly loaded version
of the same generated asset and with the existing authored egg shell where geometry/discretization differ;
label those differences rather than attributing all cost to conversion. Bound remeshing/DOF growth and
optimize measured bottlenecks while preserving full-field consistency. GPU results require actual GPU
measurements; report unavailable hardware honestly.

Finish with a concise results/progress document, commands and recommended mesh preparation settings.
Record supported/unsupported geometry and numerical/spatial limitations, actual source pins, CPU/GPU
checks, throughput and memory. Update the usage guide so the above commands are runnable. Preserve prior
evidence and scoped negative results; this feature does not reopen Atlas or unrelated completed studies.
