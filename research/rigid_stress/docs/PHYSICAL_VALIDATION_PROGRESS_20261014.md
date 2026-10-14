# Load-model and physical-validation progress

The later [independent-gauge/control update](PHYSICAL_CONTROLS_20261014.md)
supersedes the running items below: the fine shell completes with1.017% absolute
tensor error but3.966% tensor tail change. The contact control retains11/12
accepted configurations and one unresolved case. The full stage remains active.

The active stage follows [LOAD_MODEL_PHYSICAL_VALIDATION.md](LOAD_MODEL_PHYSICAL_VALIDATION.md).
The [execution protocol](PHYSICAL_VALIDATION_PROTOCOL_20261014.md) declares
observables and initial criteria before inspecting the physical curves.
Production engine `76537c2f` and its completed performance evidence remain
unchanged. This stage is active. The initial results below establish its
reference algebra and asset construction, rather than completion of the
required load-model/contact/convergence matrix.

## Independent elasticity qualification

`skfem_reference.py` uses scikit-fem12.0.2's own vector P2 tetrahedra,
quadrature, stiffness, consistent mass, boundary integration and gradients.
It imports neither `P2Shell` nor native operators. Coordinates/components
identify the correspondence between independently numbered degrees of
freedom. A mass-orthogonal rigid gauge solves the free-link system. Complete
residuals use every equilibrium row, including the six native gauge rows.

The normal test `tests/rigid/test_stress_independent_fem.py` has8 cases per
backend: U/cavity geometry, native direct/PCG and uniform-strain/pure-bending
continuous boundary tractions. Each also tests independent asymmetric nodal
loads and nonzero angular velocity. Both analytic cases use their exact
full3D free-body displacement/stress fields. Common physical probes audit
the independent affine stress reconstruction. Native recovery compares
mass, stiffness, relieved RHS, gauge-aligned displacement, energy, full
unaveraged tensor/von Mises and global peak against that reference.

CPU job711785 passes8/8 in52.772s; CUDA job711786 passes8/8 in35.992s.
Neither skips a case. The32 independently checked fields retain the original
stressRTOL1e-4/ATOL1e-3Pa and complete residual
`max(1e-11N,1e-8||rhs||)`. Maximum matrix relative errors are
5.8485e-16 for stiffness and8.4671e-16 for mass. Worst complete native
residual budget fractions are0.80403 CPU/0.91279 CUDA. Maximum scalar peak
differences are1.1114e-5Pa CPU/9.4807e-6Pa CUDA.

Source before/after hashes are identical for both runs. The later test-signature
formatting changes only whitespace/comments. The collector explicitly checks
whole-module abstract syntax tree identity and records both byte hashes.
The independent FEM package version and installed wheel hashes are retained.
The [package assembly documentation](https://scikit-fem.readthedocs.io/en/stable/api.html)
and [pinned elasticity source](https://github.com/kinnala/scikit-fem/blob/12.0.2/skfem/models/elasticity.py)
describe the external basis/form interfaces.

## Mesh construction and first physical sweep

Pinned Gmsh4.14.1 generates independent full3D ellipsoid, egg, capsule,
rounded U, smooth thin-neck, spherical cavity and1mm narrow-gap assets.
Each records the physical dimensions, continuum parameters, mesh source,
positive tetrahedral volume, boundary count and scaled-Jacobian distribution.
No symmetry reduction is used.

The first thin-shell Gmsh volume attempt fails because independently meshed
coarse inner/outer sphere facets intersect. Its complete error is retained.
The successful egg construction instead uses independently generated Gmsh
surface triangles and matched normal-offset layer topology. The physical
egg surface/taper and0.5mm wall remain fixed. All7 initial assets complete
in job711811. This construction check supplies no physical-convergence claim.

`spherical_shell_validation.py` compares independent P2 FEM with the Lamé
solution for a full spherical shell: inner/outer radii8/12mm, internal
pressure10kPa, external pressure0, E10GPa, nu0.3 and no body forces.
It tracks the raw global maximum/location, energy and36 common physical
tensor/von Mises/displacement probes. Every linear solve checks full
equilibrium. The analytic boundary conditions and exact fields follow the
[spherical elasticity solution](https://solidmechanics.org/text/Chapter4_1/Chapter4_1.htm).

The first sweep (job711824) has a21.7–23.8% coarse peak error and4.11%
finest peak error. Its finest energy/tensor-probe errors are3.73%/4.91%.
Two requested coarse sizes produce the identical mesh and count as one useful
level. These data are retained and fail the initial3% target. The generator
now supplies an explicit constant background size field and disables implicit
point-size overrides. The extended sweep has finer independent resolutions.
Its completed h0.11 field reduces the peak error to0.464%, energy error to
1.226% and von Mises probe error to1.852%, while full-tensor probe error is
still4.041%. A peak-only acceptance would miss that remaining discrepancy.
The h0.08 reference remains in progress at this checkpoint.

## Remaining required work

The full representative convex/nonconvex physical matrix, separate volume/
geometry/quadrature/intrinsic-distance refinements, four-model sensitivity
and per-island redistribution remain required. The current reference fields
must support the original raw global maximum, common-region fields and
singularity diagnosis without smoothing away element differences.

IPC Toolkit1.6.0 is installed only in the separate reference environment.
Its [primary documentation](https://github.com/ipc-sim/ipc-toolkit)
provides barrier, continuous collision detection and lagged friction routines.
It supplies no FEM bulk physics or complete simulation. `contact_reference.py`
now combines independently assembled3D P2 bulk models with its area-weighted
physical barrier, improved-max collision set, continuous collision detection
and lagged friction. A supported two-block normal-compression control solves
in job711916 with2.452e-13N free residual. Its force is0.598596N versus the
zero-gap analytic0.5N, a19.7% force difference. The activation distance10µm
is20% of the50µm imposed material closure, so contact regularization remains unresolved.
The unaveraged6030Pa peak is retained against the5000Pa analytic stress.

The Python collision callback causes the initial cancelled trial to stall.
The library's compiled patch filter preserves its identical component labels
and resolves that issue. The initial universally projected Newton direction
then fails the1e-9N solve gate. Exact Hessians with a checked positive
semi-definite direction fallback solve the same objective without changing
physics or tolerance. Failed/cancelled trials remain preserved. The
12-case3/4/5-grid and activation-distance convergence study is now running.
Friction/shear qualification, free-body grasp, the egg/interior-hotspot studies
and their native load replays remain required.

## Reproduction and data

Source `research/rigid_stress/scripts/cluster_env.sh`, then use the new
`PHYSICAL_VALIDATION_ENV` (default data-side `venvs/rigid-stress-physical`).
The environment inherits the validated base site's dependencies through a
recorded `.pth` and installs pinned reference packages locally. The base
environment is preserved. Runtime root is
`$RIGID_STRESS_DATA_ROOT/runs/20261014-physical-validation/`.

```bash
bash research/rigid_stress/scripts/check_physical_independent.sh cpu "$NEW_CPU_RUN"
bash research/rigid_stress/scripts/check_physical_independent.sh cuda "$NEW_CUDA_RUN"
bash research/rigid_stress/scripts/generate_physical_meshes.sh "$NEW_MESH_RUN"
bash research/rigid_stress/scripts/run_shell_validation.sh "$NEW_SHELL_RUN" \
    --sizes 0.28 0.20 0.15 0.11 0.08
```

The [initial immutable checkpoint](../evidence/20261014-physical/independent-fem-initial-checkpoint-v1/manifest.json)
verifies1303 entries and1303 exact originals. Manifest SHA256:
`9ee806dfcd847d213914a7ff48712ffd3510ee63d899f8fdeee5fd751f5183d0`.
Its [verification](../evidence/20261014-physical/independent-fem-initial-checkpoint-v1/verification.json)
and publisher after-source hash are derived records outside the manifest's
own file list. Live finer and successful later contact trials are excluded
from this initial checkpoint. Initial mesh reproduction uses the archived
generator version. The current generator has explicit size control.

Run fine studies on allocations. `summarize_physical_initial.py` verifies the
initial source/results and formatting equivalence. Retain failed setup-v1,
mesh-smoke-v1 and the duplicate-level/unconverged first spherical sweep alongside
successful records. The next report must include complete curves and matched
stress/traction maps rather than infer physical accuracy from these initial
linear numerical checks.
