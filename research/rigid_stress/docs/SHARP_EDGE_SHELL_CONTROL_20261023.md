# Independent single-edge pressure on a hollow shell

The unfitted sharp-edge law has **4.185% cell-pressure error** in a new
frictionless polar-land control whose reference independently determines
pressure and support. Adding finite-shell compliance changes pressure by
only **0.00536%**. Coarse-candidate versus fine-reference local tensor,
sampled tensile and sampled von-Mises errors are **1.531% / 2.696% / 1.988%**.
These results support retaining the cheap pressure law in this declared
small-contact problem. They do not close complete-shell stress accuracy,
moving friction or native configuration matching.

## Matched problem and independent pressure

Reuse the physical domain and normal-source responses from the
[independent hollow-shell pressure study](HOLLOW_SHELL_CONTACT_PRESSURE_20261021.md):
500um wall, 300um polar land, E=10GPa, nu=0.3 and consistent-mass inertial
relief. The three unchanged meshes have 60,600 / 94,608 / 139,488 tetrahedra.
This new contact has Fn=5mN and gap

    g(x,y) = 0.02*x + 80*y^2/2,  x >= 0.

The indenter orientation is fixed; indentation is solved from force balance.
Reaction moments are outputs. This differs from the earlier smooth
paraboloid's free tilt and zero-moment conditions. The declared model is
linear elastic, small-slope, frictionless normal contact on the flat land;
it is a compact physical control, not a captured Franka contact.

The candidate uses only load, modulus, slope and along-edge curvature.
Its width is 0.706532um, half-length 30.748284um and edge exponent 0.5.
Those scales size a search window x=[0,1.413064]um and
y=[-43.047597,43.047597]um. The independent active-set solve can choose any
cell in that window. All remote window boundaries remain unloaded.
The required physical edge at x=0 stays available.

The reference uses exact rectangular normal compliance plus the retained
finite-body displacement response, including the cavity. Piecewise bilinear
source interpolation is integrated over each constant-pressure cell, splitting
at table knots. Partition of unity, first moments and a mixed bilinear
control pass. General linear solves retain the small measured nonreciprocity
of the interpolated regular correction; it is not silently symmetrized.
Force, nonnegative pressure/gap and complementarity keep existing checks.
No pressure/support fitting, new elastic factorization or runtime contact
solve is introduced.

## Refinement and positive-depth stress

The coarse-body grid24 pilot and all three bodies at grids36/48 complete.
Every grid compares half-space-only and 3x3/5x5 finite-body source tables.
The fine-body grid48 pressure changes 0.0000100% between source tables;
medium/fine body pressure changes 0.0001198%. Predicted transverse centroid
error is 4.686%; square-root second-moment errors are 3.096% / 1.136%.
These moments are not certified continuum support-edge locations.

| Contact grid / body | Transverse cell width | Along cell width | Maximum cell-average pressure |
| --- | ---: | ---: | ---: |
| 24 / coarse | 58.878nm | 3.587um | 725.88MPa |
| 36 / fine | 39.252nm | 2.392um | 892.71MPa |
| 48 / fine | 29.439nm | 1.794um | 1033.10MPa |

The increasing cell maximum retains the ideal-edge singularity. Cell-pressure
L2 is a finite-resolution comparison; no finite continuum pressure-square norm
or surface maximum is claimed.

Independent displacement-gradient kernels integrate the solved cell pressure.
Adding its distributed regular response gives the reference sampled stress.
The cheap candidate combines the existing composite edge field with one
regular response at its analytically predicted centroid. Both are evaluated
at the same **140 material points**, depths **7.687071--61.496568um**
(0.25--2 half-lengths). There is no surface sample or global peak search.

| Comparison | Tensor | Sample tensile peak | Sample VM peak |
| --- | ---: | ---: | ---: |
| Coarse candidate / fine reference | 1.5313% | 2.6961% | 1.9876% |
| Fine candidate / fine reference | 1.5306% | 3.1132% | 2.0142% |
| Fine reference grid36 / grid48 | 0.14047% | 0.07279% | 0.05558% |
| Medium / fine reference body | 0.00373% | 0.04143% | 0.00239% |
| Reference source3 / source5 | 0.000702% | 0.00308% | 0.000174% |

A separate saved-array inspection reproduces the errors, verifies original
input identities and adds order6 stress integration to the retained order4
rule; tensor change is 6.86e-12 relative. Adding the observed grid, body,
source and integration variations to the coarse candidate errors gives
**1.6763% / 2.8134% / 2.0458%**. This is a deterministic variation sum,
not a certified continuum bound or confidence interval. Original 10% sample
tensor and 15% sample-peak budgets pass. Signed cancellation means the coarse
candidate's smaller tensile error does not establish greater accuracy.

Fine-reference sampled tension/VM maxima are 0.46724 / 9.31275MPa.
Principal strains at these samples range from -0.09356% to +0.04221%.
These finite-depth values do not validate linear geometry/material behavior
arbitrarily close to the loaded singular edge.

## Reproduction and remaining work

The [terminal packet](../evidence/20261019-qualified-ranking/sharp-edge-shell-pressure-v1/README.md)
retains exact captured workers, source inventories, input hashes and jobs
742984 / 742994 / 743005. Runtime arrays remain under the existing ranking
data root in edge-shell-pressure-pilot-v1, edge-shell-pressure-refinement-v1
and edge-shell-pressure-inspection-v1. Run
qipc_comparison.study_edge_shell_pressure with --parent and --grids, then
qipc_comparison.inspect_edge_shell_pressure with the three ordered --studies.
Use captured tasks.json through shell_comparison.submit, CPU Slurm rtx-mid,
image1_26, the existing physical environment and explicit data/private caches.
These offline costs do not enter native throughput.

Next reuse these independently solved pressures for complete-volume,
membrane/bending and spatial peak comparisons, with explicit exclusion/depth
semantics at the ideal edge. This local bank alone cannot establish them.
General curved/nonconvex and changing friction contacts, matching native
accuracy/cost, useful independent batch populations and full-cycle admission
remain requirements of the [active goal](../GOAL.md). The original frame428
refusal, numerical checks, engine baselines and prior failed references remain
unchanged.
