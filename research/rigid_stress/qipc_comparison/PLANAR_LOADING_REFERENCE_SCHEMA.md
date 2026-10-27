# Planar loading reference schema v1

`export_planar_loading_reference.py` freezes the CPU primary
`797203553f246ca155bbef909c1bb33e7cd7b90c`. Run it in Slurm `rtx-mid`, image
`1_26`, with a fresh explicit data output:

```sh
python -m research.rigid_stress.qipc_comparison.export_planar_loading_reference \
  --git-source /path/to/actual/git/checkout \
  --output /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261027-planar-loading-dpr/oracle-v1
```

The module imports standard-library modules before checking its allocation.
It copies exact Git objects into `reference-source/`, including the primary
shell elasticity, pressure fitter and asset. A fresh interpreter receives only
that tree on `PYTHONPATH`; its imported reference paths and unchanged hashes are
checked. The current native implementation cannot supply the CPU oracle.

The default export retains k8/k16. A separate `--refinements 32` export uses
the unchanged pinned CPU implementation with finer loading geometry and the
same Q4 elastic DOF, source anchors and pressure law. Use a fresh `oracle-k32`
output; the predeclared campaign `meta/k32-plan.json` records its qualification
scope before testing. Existing exports remain preserved.

## Physical inputs and integer topology

Native preparation/runtime may read `inputs/`:

| Payload | Meaning |
| --- | --- |
| `proxy.json` | Refinement integers, diagonal topology and inner/outer ordering |
| `egg75/model.npz` | Physical midsurface vertices, Q4 connectivity, authored directors and scalar material/thickness/density/drilling parameters |
| `egg75/sources.npz` | Original point, full force, outward normal, supplied principal curvatures, side, radius flag/value, friction, current parent-face witness, environment/slot and case offsets |
| `egg75/cases.json` | Exact 75 published cases and their unchanged physical inputs |
| `static16/case_indices.npz` | Integer aliases into the 75-case bank, retaining the published inner/outer benchmark selection |
| `static16/cases.json` | Physical sources of those sixteen environments; sixteen sources, including one two-source environment and one release |
| `torus8/*` | Fixed nonconvex torus physical inputs and eight prescribed 0.5/1 mm cases |
| `controls/*/{model.npz,cases.json}` | Fixed planar/tilted/warped geometry and source controls |

The full egg bank has 77 contact sources: 16 interior, 26 triangle-seam, 18
material-edge, 12 finite-radius cases, two asymmetric multiple-contact cases
and one empty release case. Automatic-radius cases carry `radius_given=false`
and value zero as a sentinel; recompute the size from the supplied physical
curvatures and original force. Native dynamic radius qualification remains tied
to its actual collision geometry and curvature, separately from this controlled
source bank.

CPU source normals point **outward**, with compressive load `-F·normal`. Native
`StressContactState.normal` points **inward**. An adapter flips this normal once;
it preserves point, full force, side, pressure support and friction. Sources use
the three-dimensional surface-ball Hertz density. Proxy facet normals do not
replace the original physical normal/curvature inputs.

## Independent numerical oracles

Everything under `oracles/` is validation-only, including derived triangle
geometry and maps. CPU matrices, loads, fitted pressure, displacements and
stress must not enter native preparation, native current-load generation or
timing.

Each elastic bank (`egg75`, `torus8`) contains:

- `operators/{K.npz,K_free.npz,S.npz,numerics.npz}`: original physical stiffness,
  translation gauge, mass/rigid modes, relief ingredients, stress operators and
  Gauss samples. Sparse matrices use SciPy CSR NPZ format.
- `proxy-k{k}/{outer.npz,inner.npz}` for each requested refinement: independently derived triangle vertices,
  parent Q4 shapes/faces, normals, duals and bounds. These test native floating
  preparation; they cannot be uploaded as its implementation.
- `Q4-q{24,48}/` and `tri-k{k}-q{24,48}/`: raw source and grouped loads,
  balanced RHS, relief acceleration, original-gauge displacements, whole-object
  tensor fields, sampled peaks and N/M/Q. `Q4-q48` is the spatial reference;
  matched proxy/order results test numerical equivalence separately.
- `sources.json`: radius, exact-contained/clipped route, sample/query counts,
  pressure-fit diagnostics and original-point wrench checks for every source.
- `spatial.json` and `summary.json`: area-weighted field/resultant errors against
  the original Q4 q48 route, preserving each loaded case. Release is exact zero.

`oracles/independent/checks.json` replays the primary's 17 analytical, virtual
work, rigid-motion, load, inversion, quadrature and unchanged-elasticity checks.
`oracles/controls/warped-miss` retains an original Q4-valid load and the coarse
proxy miss. A runtime safeguard must select a validated original loading route;
force-line reanchoring changes the physical source and is excluded from this
export's candidate paths.

`published-reproduction.json` compares the recomputed q24 method statistics
with the primary publication at absolute tolerance `1e-8`. Published physical
source bytes are used exactly; the regenerated case family and physical vectors
are independently checked to `rtol=1e-13`, `atol=1e-15` first.
The primary has no published k32 statistic. Its k32 comparison row explicitly
records `publication_status=unpublished_derived`,
`reproduction_status=not_applicable`, `passed=null` and `gating=false`.
Q4 and every requested existing published method still pass their unchanged
reproduction check; this exception cannot hide a missing existing method.

## Array ordering and acceptance

Raw/generalized loads and displacement use `(6*physical_node, environment)`;
each node stores translation xyz then infinitesimal rotation xyz. Native shell
triplet block `2*i` is translation and `2*i+1` is rotation. Source loads have
source as their final axis; `environment` maps each source into a grouped RHS.

Local stress is `(side=2, element, Gauss=4, component=(xx,yy,xy), environment)`.
Full stress is `(side=2, element, Gauss=4, row-major tensor9, environment)`. Side
zero is outer/top. N/M are local symmetric tensors; Q is a local vector. Field
and N/M norms use Gauss area and shear multiplicity two; Q uses the same area
with vector L2. Peaks use both sides and every unaveraged Gauss sample.

The CPU solution uses the original translation gauge. Coordinate comparisons
must match that gauge or explicitly align the rigid nullspace; load, tensor,
resultant and original residual comparisons remain independent checks. Exact
numerical acceptance stays unscaled `rtol=1e-8`, `atol=1e-11`. The task freezes
existing spatial budgets separately from stricter k8/k16 fidelity targets in
the campaign's `meta/error-budgets.json`.
Native k32 qualification explicitly uses `target_reference_key=k16` and the
unchanged k16 limits: field 0.5%, sampled peak 0.3%, N 0.15%, M 0.75% and
Q 0.4%. This inherited target is mandatory, separately from publication
reproduction; a missing k16 target fails. Existing k8/k16 gates and defaults
remain unchanged, while k4 retains the original spatial budgets and an
explicitly inapplicable extra fidelity target. Export success alone does not
qualify k32 accuracy or cost.

`manifest.json` inventories every file with bytes/SHA256 and NPZ array hashes
where exported. It records source isolation, versions, all 75/8 cases, both
quadrature orders, peak RSS and elapsed export time. A successful export is an
independent oracle, not a native GPU throughput or physical-contact claim.
