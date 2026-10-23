# Independent single-edge pressure and stress on a hollow shell

The unfitted sharp-edge law has **4.185% cell-pressure error** in a new
frictionless polar-land control whose reference independently determines
pressure and support. Adding finite-shell compliance changes pressure by
only **0.00536%**. Coarse-candidate versus fine-reference local tensor,
sampled tensile and sampled von-Mises errors are **1.531% / 2.696% / 1.988%**.
Complete-domain follow-up gives3.260% volume-tensor error, but its error
integral and pressure-grid reference have not converged to the declared
targets. Refined peak comparisons pass conditionally at depth cutoffs
0.707um and7.687um; the0.177um tensile peak remains unqualified.
These results support further work on the cheap law in this declared
small-contact problem. Moving friction and native configuration matching
remain open.

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

## Complete-domain fields and integration

The follow-up holds the fine-body solved pressure fixed while comparing all
three body responses. Exact uniform-rectangle stress integration supplies
the near field; polynomial source integration supplies the far field, with
orders9/17 checked separately. Independent complex-step quadrature agrees
with the rectangle kernel to4.48e-15 relative, and the original saved local
fields agree to6.87e-12. A batched evaluation of the existing composite edge
candidate agrees with its scalar implementation to2.12e-16 relative.

Two longest-edge-bisection partitions preserve the original material volume
3.7303908383e-6m3 and midsurface area0.007474502743m2 to roundoff. They use
492,300 /638,080 volume points and595,440 /1,526,040 depth points in
37,620 /46,410 material columns. Contact-focused spatial scales halve from
0.5 to0.25 candidate widths, while depth quadrature increases from8 to12.
Volume tensor, membrane and bending integrals include the whole domain;
no contact neighborhood is removed. The minimum quadrature depths are
33.74 /16.87nm in volume and0.06963 /0.01501nm in columns. These are
integration points for the declared ideal linear-elastic model, not claims
of material validity at atomic scales.

Peak searches have separate cutoffs0.176633 /0.706532 /7.687071um,
respectively0.25width /width /0.25half-length. Initial searches combine
volume and column points, three35x121 polar planes and every retained
element's unsmoothed corner stress. A subsequent local3D refinement starts
from each observed tensile/VM maximum, retaining the original maxima and
the same depth cutoffs. The initial central plane spacing is88.32nm across
and768.71nm along the edge. Eight local search passes use four successively
halved boxes; final lattice spacings are11.04nm across,48.04nm along and
max(width,depth/4)/16 through depth. Neither sampled maxima nor local refinement certify
a continuum global maximum; the ideal surface peak remains unbounded.

The first complete-field readouts rejected boundary vertices in the strict
cross-mesh locator. Diagnosis found all lower-bank volume and column points
inside, with rejected vertex distances at most1.05e-17m. The repaired
readout uses each element's own corner stress and exactly matches the
particular-field vertex coordinates. It preserves all one-sided corner
values, the locator thresholds and original failed records. Completed
particular arrays are reused in place; no pressure or stress law changes.

### Measured errors and failed integral gates

All rows below compare against the fine-body grid48 reference. Percentages
are weighted relative L2 errors; membrane/bending are integrated tangential
tensor resultants, not local peak stresses.

| Integration / candidate body | Volume tensor | Membrane | Bending |
| --- | ---: | ---: | ---: |
| Lower / coarse | 2.6290% | 0.9318% | 1.0049% |
| Higher / coarse | 3.2604% | 0.8247% | 0.8301% |
| Higher / medium | 3.0946% | 0.5928% | 0.7669% |
| Higher / fine | 3.0343% | 0.1536% | 0.2585% |

The reference norms change0.5837% /0.0530% /0.0067% under integration
refinement, within the2% norm target. However, the corresponding candidate
error norms change19.84% /12.93% /21.06%, exceeding the separate5% error-norm
target. Small reference-norm changes do not establish a converged error.
On the higher partition, pressure-grid36/48 tensor change is2.365%, above
the2% reference-field target. Medium/fine body changes are0.6104% /0.5750%
/0.7208%; source3/5 changes are0.0241% /0.0182% /0.0535%. Far integration
changes remain below2e-9 relative.

Adding the observed same-query pressure/body/source/far changes to the
coarse candidate errors gives6.260% /1.467% /1.687%. These sums exclude
the separately tested integration discrepancy and are not certified bounds.
**All three complete-field acceptance gates remain false** because the
error integrals are not stable; volume also fails the reference-field gate.
About80.39% of reference volume stress-square energy lies within8 contact
half-lengths. The original lower-partition results remain recorded rather
than being presented as the final error estimate.

A saved-array decomposition separates particular-field, regular-response
and signed cross terms and reproduces every total error. About97.77% of
the volume error-energy change lies shallower than0.176633um; the volume
regular-response error energy changes only1.63e-11 relative. Conversely,
regular-response terms dominate membrane/bending error energy, while their
particular terms change only about1.66% /1.73%. Thus the next checks can be
targeted: resolve the shallow singular/cell-pressure integral and integrate
the piecewise-linear regular stress through each element crossing along
material columns. Simply tightening the elastic solver is not supported
by this diagnosis.

### Refined positive-depth peaks

The local3D check reuses each variant's saved regular displacement and
evaluates the exact cell-pressure or unchanged composite particular field.
It independently refines grid36/grid48, source3/source5, medium/fine body
and coarse candidate peaks. The greatest final-level incremental change
is0.0161%; changes from the original bank are retained separately.

| Minimum depth | Reference tensile / VM | Candidate tensile / VM errors | Observed variation sums | Conditional result |
| --- | ---: | ---: | ---: | --- |
| 0.176633um | 0.77548 /199.709MPa | 4.071% /1.349% | 12.456% /1.440% | Tensile fails; VM passes |
| 0.706532um | 0.62890 /86.174MPa | 3.761% /1.635% | 3.854% /1.711% | Both pass |
| 7.687071um | 0.46724 /9.313MPa | 2.670% /1.988% | 2.790% /2.047% | Both pass |

The sums include observed pressure/body/source peak changes and the last
two local search passes. The declared gates are5% for each reference
change,1% for the last search increments and10% for the summed candidate
comparison. At the shallowest cutoff, pressure-grid tensile change remains
8.378%, although local spatial refinement is stable. Along-edge pressure
cells are1.794um wide, larger than this depth. Additional local peak search
cannot repair an unresolved pressure representation.

At the middle cutoff, maximum tension occurs about3.231um deep, rather
than on the cutoff plane. These are observed maxima over the retained
domain bank plus local searches. They do not certify the global continuum
maximum, material nonlinearity or the native scene configuration, and do
not override the failed complete-field integration gates.

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

The [complete-field packet](../evidence/20261019-qualified-ranking/sharp-edge-shell-fields-v1/README.md)
adds the exact-rectangle controls, two sampling partitions, failed and
repaired complete-field readouts, boundary diagnosis, independent field
inspection, local peak refinement and integral decomposition. Reuse its
captured tasks.json with shell_comparison.submit. The main workers are
qipc_comparison.edge_shell_sampling, study_edge_shell_fields,
inspect_edge_shell_fields, inspect_edge_shell_peaks and
diagnose_edge_shell_integrals; all inputs and data-side destinations are
explicit. No new elastic factorization or production implementation is
introduced. The larger repaired field job takes19m34s on one CPU; this
offline validation cost is separate from native throughput.

Next resolve the error-integral sensitivity and shallow pressure-grid
sensitivity before promoting the complete-field error estimates. Preserve
the conditional finite-depth peak results and every failed gate.
General curved/nonconvex and changing friction contacts, matching native
accuracy/cost, useful independent batch populations and full-cycle admission
remain requirements of the [active goal](../GOAL.md). The original frame428
refusal, numerical checks, engine baselines and prior failed references remain
unchanged.
