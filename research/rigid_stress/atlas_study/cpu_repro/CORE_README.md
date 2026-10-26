# Reconstructed CPU mechanics and geometry

This implementation reconstructs the reported common-mother experiment from
its retained report and the published `shell_cpu` finite element core. The
missing unpublished source has not been recovered. Source, fixture and timing
identities in this package define the new experiment.

## Frozen physical model

`Mother()` uses 54 coarse bilinear macro quads on a cube-chart egg. Each
macro has four subdivisions per direction, producing 864 fine quads, 866 nodes
and 5,196 translation/rotation coordinates. The coarse mesh has 56 nodes and
336 coordinates. Authored axes are 22/22/30 mm, asymmetry is -0.18, thickness
is 0.5 mm, Young's modulus is 10 GPa, Poisson ratio is 0.3 and density is
1,000 kg/m3. The unchanged published MITC4 shell operator supplies stiffness,
rigid modes, lumped translational and transverse-rotary mass, stress and drilling
stabilization. Output has four Gauss points per element and both material
sides, with three local in-plane components per point: 20,736 scalar rows.

Authored egg coordinates define the coarse **midsurface**. Fine physical
positions are bilinear interpolation of the frozen coarse positions, rather
than a new sampling of the smooth egg. Nodal directors are smooth authored
normals sampled once and normalized. `Geometry.position_convention` records
this construction. Independent body-frame tensors follow the original
`ShellModel.stress` convention. The shell output remains an in-plane
membrane/bending observation.

The initial pilot interpreted the authored axes as an outer surface and
subtracted half a thickness. That interpretation was corrected to the
report's stated midsurface convention before formal timing. Initial outputs
retain an `initial_outer_offset` diagnostic identity. Their geometry source
snapshot is `diagnostics/initial_outer_offset_geometry.py.txt`, SHA-256
`f7519ca720aa0868d3be5c24352c76f504a8a447926d32871053268c61c3bc69`.

## Actual reduced feature spaces

`Geometry.P_nodes` interpolates coarse nodal values onto the fixed mother.
`Mother.P` applies it independently to six coordinates per node. It exactly
reproduces rigid translations and rotations because the physical positions
use that same interpolation. Six distributed coarse translation rows remove
only rigid-motion ambiguity, and the corresponding mother coordinates are
the common gauge for every solve.

A feature first selects a conservative spherical cap of coarse macros. It
adds a vertex-adjacency halo. A fine node supplies six independent local
bubble coordinates only when all incident macros are active and that node is
not already a coarse vertex. Thus trial functions remain conforming at the
fine/coarse interface. The sparse transform is `T = [P_free, I_local]` and
the original reduced stiffness is `T.T @ K_mother @ T`. Feature matrices
actually have fewer independent coordinates. Every feature sees the same
physical geometry, mass, stress samples and material. The twelve icosahedral
features have 3,522 coordinates before the common gauge, or 3,516 independent
coordinates afterward, with 531 local fine nodes and 37 active macros.

The mother mass and rigid modes define relief before feature projection.
`Mother.balance(raw)` applies that one common linear relief. `Space.solve`
takes an already balanced load. `Space.project_raw` instead preprojects the
same relief modes to avoid forming a full balanced intermediate. Those paths
are algebraically equivalent. Projected-system residual measures numerical
solve accuracy. Mother-space residual also contains spatial approximation.

## Current material-cap loading

Each `Source` names its actual continuous unit material direction, nominal
radius and force. Radius divided by 22 mm defines a spherical cap angle.
A normalized radial Jacobi rule and 4q angular samples produce 4q^2 points.
This is a specified material-cap Hertz-shaped load adapter. It is distinct
from an independently solved contact area or the native support-ball law.

Every sampled direction is mapped to its authored egg ray. A stable closed
form ray/bilinear intersection selects the frozen macro surface, followed
by exact fine Q4 interpolation. Ambiguous face seams use an exhaustive
macro search. A supplied global force bypasses the unused center-normal
query. Otherwise the current physical midsurface normal and tangential
direction generate normal/friction load. `cap_tangent` can hold a smooth
material quadrature frame in selector-continuity controls.

Physical nodal force is accompanied by the couple
`side * h/2 * (director cross force)`. Both side signs are supported by the
load adapter. The optional audit compares resultant force and moment with
the same integrated physical offset surface, before common inertia relief.

## Public research interfaces

- `Mother` exposes `K`, `S`, `P`, `free`, `fixed`, `geometry`, `balance`,
  `solve`, `feature` and `space`.
- `Space` exposes `T`, `K`, `S`, `active_macros`, `local_nodes`, `project`,
  `project_raw`, `solve`, `solve_projected`, `lift` and `stress`.
- `n_dof` on a feature counts independent coordinates after gauge removal.
  `n_dof_total` includes its six gauge coordinates. Mother `n_dof` counts
  all physical coordinates and `len(mother.free)` counts independent ones.
- Load columns use `(ndof, batch)`. `assemble_sources` returns raw physical
  columns. Solvers accept balanced columns. Stress row order is side,
  element, Gauss point and local component.

Run from the repository root with one BLAS/OpenMP thread:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m research.rigid_stress.atlas_study.cpu_repro.geometry_check \
  --output /absolute/new-output/geometry_checks.json
```

That control compares the closed-form mapping with an exhaustive Newton
oracle in three physical coordinates, plus complete load/field and explicit
force-shortcut controls. The final 15,240-query audit validates that every
selected chart is independently admissible and compares local coordinates
under the same chart/domain-clipping convention. The maximum local parameter
difference is 1.67e-15 and physical position difference is 1.04e-17 m.

Adjacent macros can both be admitted within the declared domain tolerance.
The fast current-chart choice and minimum-index oracle choice differ for
170/40/334 boundary queries across the three configurations. These canonical
aliases are recorded explicitly. The initial audit incorrectly demanded ID
identity and compared clipped against unclipped coordinates. Its failed
result and source are retained as `diagnostics/strict_boundary_identity_initial.json`
and `geometry_check_strict_initial.py.txt`. The corrected physical-mapping
contract keeps the original numerical tolerances and the production core
unchanged. Complete source-load differences are below 2.63e-16, complete local
stress differences below 2.77e-15, and the explicit-force shortcut gives
identical load arrays in the declared controls.

Initial batched-solve and serialization failures remain in the retained
diagnostic logs. Model construction is separate from steady-state timing.
