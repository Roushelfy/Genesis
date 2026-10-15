# Corner arithmetic, independent stress precision and resolved CAD loads

The [complete physical-validation objective](LOAD_MODEL_PHYSICAL_VALIDATION.md)
remains active. The native corner evaluator now cancels common nodal
translation before multiplying by geometry. Complete CPU/CUDA frozen-field
checks pass the unchanged component budgets against independently verified
96-digit polynomial stresses. The old float64 comparison also has measurable
roundoff above that budget; its original failures remain retained. This
numerical repair does not explain or close the much larger physical spatial
and contact discrepancies.

## Diagnose both the native calculation and its reference

Original native replays716476/716477 on8433/30298 tetrahedra pass complete
equilibrium and whole-field linear tensor checks, while their strict cached
corner comparison fails. Frozen-field diagnostics719196/719197 preserve their
actual wrapper1 and every original displacement value. Newly rebuilt CUDA
cached fields reproduce the stored fields exactly. For all four mesh/load
combinations, complete cached and uncached arrays have byte-identical NPY
contents. The cache shortcut alone therefore does not explain those failures.

The original pointwise budget is
`1e-5Pa +1e-9*abs(reference_component)`. Whole-tensor relative agreement can
coexist with a violation in a small component formed by cancelling large
nodal terms. CPU diagnostics719198/719204 separately retain native/reference
geometry gradients, extended-precision accumulations and an independently
computed extended cofactor geometry. Both geometry and accumulation contribute;
their diagnostics are not an additive decomposition of the final error.

To audit the precision reference,719228/719233 use exact rational inversion
of the ten-monomial quadratic Vandermonde matrix. They verify every monomial's
derivative at all four corners. Pivoted Gauss-Jordan inversion then computes
each physical geometry map with96 decimal digits. Binary float64 inputs are
converted exactly before that arithmetic. Every element/corner/component,
at both full-inertia and rigid-relief frozen states, remains checked and saved.
The largest decimal inverse identity residual is7e-94. Full extended-cofactor
fields agree within0.00283 of the original component budget.

On30298 tetrahedra, the old float64 reference itself differs from the decimal
polynomial field by as much as6.45759/4.12686 budget units for full inertia/
rigid relief. Those are measured reference arithmetic errors, not elastic
equilibrium errors or a reason to relax the component tolerance.

## Equivalent quadratic edge tangents improve the actual native kernels

For cornerc and another vertexj, the outgoing quadratic edge tangent is

\[
t_{cj}=4(u_{cj,\mathrm{mid}}-u_c)-(u_j-u_c),\qquad
\nabla u(x_c)=\sum_{j\ne c}t_{cj}\otimes\nabla\lambda_j.
\]

This is the same affine quadratic displacement field. Forming differences
first preserves translation cancellation before multiplication by large
geometry coefficients. The cached path uses its existing corner nodes and
gradients; the full path uses existing element connectivity and barycentric
gradients. Output modes, four-corner/global unaveraged maxima, material,
elastic operators, loads, force budgets, gauge rows and rigid dynamics remain
unchanged. Production arithmetic stays in Quadrants.

Candidate probes719217/719232 precede the production change. Actual production
checks719248/719249 then rebuild the native model and evaluate both paths on
every original frozen component. Eight complete fields per backend cover
both meshes, both load modes and both paths. They also retain the original
raw-global-peak reconstruction budget.

|Actual production backend|Largest component error /original budget|Complete frozen fields|
|---|---:|---:|
|CPU|0.154571|8|
|CUDA|0.074298|8|

The independent offline helper now differentiates the exact quadratic
polynomial with extended cofactor geometry in scikit-fem's degree-of-freedom
order. Full-field qualification719275 checks this helper against the decimal
oracle: the largest error is0.000052824 of the unchanged component budget.
The replay validator uses this verified reference and retains the legacy
float64 tensor and its measured comparison separately. Historical failures
are preserved rather than relabeled as successful runs.

A small portable regression fixture retains the unchanged worst original
tetrahedron and frozen nodal values. Its old native field differs from the
decimal target by4.83505 budget units. Converting only the test target to
portable float64 consumes at most7.81e-8 of that budget. The complete96-digit
derivation, extended targets and source identities remain in the evidence.
Both CPU719264 and GPU719265 pass13 selected tests, including both corner
paths, independent uniform-strain FEM on U/cavity meshes, full/invalid output,
shared operators, direct/inverse recovery, history invalidation and resets.

Fresh actual native replay719284 closes both modes on8433 tetrahedra at
the unchanged100000 iteration cap and original tolerances. Its maximum complete
force-budget fraction is0.055821; maximum corrected component-budget fraction
is0.0290. Fresh30298-tet replay719285 passes both modes as well: maximum
complete force-budget fraction0.173894 and component-budget fraction0.083248.
The legacy float64 comparisons still exceed their component budget by factors
5.3601/4.6515 on that run, and remain explicit outputs. All four fresh replays
retain full equilibrium, complete fields, raw peak and unchanged tolerances.

## Guard the changed stage's cost without widening its scope

Matched benchmark719295 loads the exact old source in its own module and
compares old/current full-output corner/von-Mises/global-peak kernels on the
same device, original low mesh and manufactured affine displacement. Every
original/current component is retained and checked against the analytic field.
Five warmups precede seven alternating-order samples, each timing ten launches
and synchronization. No elastic solve or contact mapping is timed.

|Batch|Old median per launch|Current median per launch|Current /old|
|---:|---:|---:|---:|
|1024|0.323570ms|0.320819ms|0.99150|
|2048|0.594554ms|0.576046ms|0.96887|

These observations guard against a large regression in the changed stage.
Their small differences are not a warmed application speedup or a physical
accuracy/cost operating range. Raw samples, hardware and complete fields remain.

## Covered CAD states separate linearization from spatial/contact error

Independent audit719205 uses the already balanced31668/64716-tet covered CAD
states at the same6.25ms time and original held closure, friction, material,
density, gravity and jaw actuation. It reconstructs full-inertia resolved loads
and independent linear FEM, rather than treating an unchanged scalar peak as
a model check. Frozen objective-deformation stress and rigid-relief loads are
separate diagnostics. Complete volume tensors, raw corner peaks, energies and
reference wrench-balance corrections remain.

|Tetrahedra|Resolved linear/nonlinear volume tensor change|Frozen-objective constitutive change|Resolved linear raw global peak|
|---:|---:|---:|---:|
|31668|0.143684%|0.313597%|40.363746MPa|
|64716|0.145668%|0.313633%|46.873744MPa|

These conditional linearization errors coexist with the earlier9.68% spatial
tensor change. They do not establish pressure convergence, full-arm feedback,
the complete original approach/lift/support application or a physical tail.
The changed geometry and resolved loads also prevent interpreting this table
as an isolated improvement relative to the old coarse-polyhedron cases.

Fine producer719100 now closes all five original171228-tet dynamic states.
Independent auditor719101 and contact-coverage auditor719102 also close:
the final mapped/body residuals are1.443373e-13/5.426195e-14N, and all129
active final body samples lie inside the declared numerical marking caps.
Final raw global bounds remain[55.382574,55.415414]MPa. Maximum unsmoothed
nodal normal force divided by incident lumped area is344.688MPa, a discrete
diagnostic rather than continuum pressure. These results do not establish
the pending three-level common-field tail. Comparator719103 retains all158340
original probes; subsequent719220 extends the linearization comparison only
after the original independent gates close.

## Reproduction and remaining acceptance

Reusable entry points include the frozen arithmetic diagnostics, difference-form
probe, decimal polynomial verifier, actual native qualifier, focused regression
wrapper and matched corner-stage benchmark. The existing resolved replay now
records both improved and legacy reference comparisons. Actual successful and
failed inputs, complete arrays, source closures, exact Slurm submissions,
versions/hardware and bounded-byte reconstruction remain required evidence.

The [immutable numerical/model publication](../evidence/20261014-physical/corner-arithmetic-model-v1/manifest.json)
retains all17 closed new runs, original/current complete fields, actual fresh
replay results, regression receipts and matched timing samples. Manifest SHA256
is`72cc5526e417b0e4eb7b8d97f73a0c45191e581076db25c1b106c31594a8134f`.
Verification checks955 archive entries against931 actual runtime originals and
reconstructs seven large files from exact ordered byte chunks. Both original
failed-replay/geometry and covered-field parent archives are fully reverified;
every referenced actual input is rehashed without duplicating parent-covered
arrays. The fine producer/audits and pending comparison are execution status
outside this numerical/model packet; they do not supply a physical tail.

All original A/B/C requirements remain: separated bulk, geometry, contact,
quadrature, regularization and history tails; finite-load/model/local-island
sensitivity; meaningful matched full egg/Franka and nonconvex contact references;
resolved native replay versus actual feedback; and warmed costs at demonstrated
physical accuracy. This numerical repair preserves that scope.
