# Automatic contact-region study: first measured results

The [new goal](AUTOMATIC_CONTACT_AND_THROUGHPUT_GOAL_20261020.md) remains in
progress. A compact CPU study now separates physical pad support from the
existing reconstruction radius. The retained2mm DPR baseline sends about42%
of the measured normal load onto the disconnected inner boundary. Conservative
transfer of the same outer-face pressure changes the complete linear elastic
stress by about8.70%. This is a measured load-transfer effect, not a newly
qualified physical stress error or an admitted native implementation.

The [evidence packet](../evidence/20261019-qualified-ranking/automatic-contact-support-v1/README.md)
retains executed sources, original reader failures, terminal receipts and the
inspected pressure figure. Full fields, maps and original quadrature remain
hashed on data. [Subsequent native and source-input results](RESOLVED_PAD_AND_SOURCE_INPUTS_20261020.md)
add400-step Quadrants resolved-pressure conformance, completed DPR/SCR1000-step
bounded-roundoff variants and an independent force-only SCR conflict certificate.
These retain their research scope and do not supply a production automatic model.
Existing fixed-radius rankings and their running fine-reference
histories continue independently, without source edits or restarts.

The [subsequent compliance study](PAD_COMPLIANCE_STUDY_20261020.md) measures
eight stiffness/load/asymmetry cases, actual search expansion and source-partition
controls. It distinguishes small nominal rigid-gap error from substantial
strong-compliance pressure error and separately measures contact quadrature.

## Physical support and pressure together

`study_contact_regions` solves the declared independent-column law
`sum(k_A*A*[delta-g]+)=Fn` over positive quadrature areas. Its sorted active
prefix selects the pressure-bearing support; no reconstruction radius is
inserted. Final pairwise force reductions avoid cumulative-sum drift. The
original analytic failure at relative load error3.70e-12 is retained; the
force-balance target remains1e-12.

For a positive quadratic gap with principal curvatures c1,c2, integration gives
`Fn=pi*k_A*delta^2/sqrt(c1*c2)` and axes `ai=sqrt(2*delta/ci)`.
The spherical fourth-root law follows as a special case. Four controls cover
loads0.1–1N, anisotropy, orientation and a factor-two pad stiffness change.
Maximum errors are0.000501% indentation,0.0738% support area and0.00105%
pressure-weighted axes. Splitting each source area in two changes the force
field by at most3.79e-16 relatively. A nonconvex double-minimum gap has two,
one and two connected regions on loading/unloading0.1→10→0.1N. Zero or negative
principal curvature explicitly refuses the local quadratic formula.

On the existing coarse StVK0.1N normal endpoint, the rigid-geometry/full-gap
prediction differs from measured pressure by0.266%/0.411% area-weighted L2 on
upper/lower sides. Using the saved **deformed** gap reproduces its declared
normal law to1.52e-13. The latter is a constitutive reconstruction check using
the answer's deformation; it is not a shell-compliance prediction. These soft
pad/flat-land observations do not validate a universal curvature rule, a hard
shell contact law or a thickness-independent predictor.

The column law is the benchmark's explicitly declared constitutive assumption.
It differs from a laterally coupled elastic layer and from Hertz half-space
contact; see the [pad declaration](COMPLIANT_PAD_DECLARATION.md) and
[spring-foundation convention](https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_theory.06.068.html).
Shell response needs its geometry and compliance, as discussed in
[Sun and Paulose](https://arxiv.org/abs/2103.06307). No radius is fitted to stress.

## Actual finite-radius reconstruction of distributed sources

CPU job731403 reads the previously exported **actual production quadrature**,
whose signed P2 scatter matches the original native nodal force. It does not
replace the native profile with a surrogate Gaussian convolution. Every parent
facet is matched by its P2 corner coordinates to the declared outer or inner
mesh. All nonzero forces, including residues, remain included.

The two saved states are normal0.2s and sustained-friction0.4s, from admitted
full-centroid rigid-pad histories. The following ranges include both endpoints
and both plate sides, retaining the old2mm/manual configurations:

| Actual reconstruction quantity | DPR | SCR |
| --- | ---: | ---: |
| Normal load transferred to the inner boundary |41.584–41.774%|0%|
| Normal load outside originally active outer faces, including inner load |53.054–54.023%|18.802–20.346%|
| Pressure difference on both boundary layers, area-weighted L2 |66.361–66.485%|33.394–33.467%|
| Exact-nonzero support area / original support area |7.465–7.654|3.886–3.942|

Here normal weights are unsigned components along the physical plate direction;
original inner pressure is zero. Cell averages divide integrated force by the
actual projected parent-facet area. They are not resolved pointwise pressures.
Support area is sensitive to the explicit exact-nonzero convention. Complete
force/moment defects remain at most2.64e-11/5.41e-11 relative to unsigned
source-force scales, below the unchanged1e-8 gates. Global wrench conservation
therefore does not identify the pressure distribution. SCR's body-wide fit also
need not preserve each individual plate reaction exactly.

The [four-panel pressure figure](../evidence/20261019-qualified-ranking/contact-region-figure-size-v2/contact-region-figure-v2/group0/figure/contact-pressure.svg)
shows the complete upper polar support with a common0–9kPa scale: measured
outer pressure, DPR outer pressure, DPR inner loading and SCR outer pressure.
No interpolation, smoothing or per-panel normalization is used. At the normal
endpoint, DPR places58.41% on the outer boundary and41.59% on the inner boundary; SCR's upper reaction is
99.96% of the measured upper reaction. The actual paired original inputs agree
within1e-8. The183×145mm editable SVG/PDF,600dpi PNG and150dpi preview pass text
checks and rendered visual inspection. The caption and quantitative argument
are retained separately from the artwork.

The original export's shared style enabled automatic cropping, despite the
requested canvas size. Its actual163.055×139.449mm artwork remains in the first
packet. The [figure companion](../evidence/20261019-qualified-ranking/contact-region-figure-size-v2/README.md)
explicitly disables cropping and verifies actual SVG/PNG dimensions. It uses
the identical pressure/data contract and retains every nonzero parent facet;
only unused zero-pressure polygons are replaced by the matching background.

## Conservative distributed-source transfer and elastic response

CPU job731405 uses the existing independent free-link P2 elasticity adapter.
It applies each already physical outer-face centroid force through its exact
six P2 face functions. Their signed nodal values are permitted; no second finite
disk is added. The paired control uses the original DPR nodal load. Both use
the same native mesh,10GPa/nu0.3 material and recorded body/inertial load.
Free-link mass projection, numerical gauge removal and the complete elastic
equations remain checked; no physical vertex is pinned.

Both normal/friction endpoints pass the predeclared controls: force/moment
defects below1.48e-16/2.41e-18, virtual-work defect below2.25e-17, and full elastic
relative residual below4.87e-12. The same-load CPU control matches original
native complete stress to1.06e-11, against the predeclared1e-6 limit.

| Endpoint | Transfer-induced complete-stress change | Old/new raw VM peak | Old/new raw tensile peak |
| --- | ---: | ---: | ---: |
| Normal |8.69794%|196.434/207.102kPa|119.467/131.071kPa|
| Sustained friction |8.69904%|196.439/207.400kPa|119.477/131.075kPa|

The field change uses the exact physical-volume integral of squared affine
stress, divided by the conservative-transfer field norm. It isolates a change
of load transfer on one linear elastic mesh. It cannot be interpreted as an
8.70% improvement against continuum truth or combined with old2mm accuracy or
throughput values. Complete fields remain available for the existing common
material-column comparisons once the fine reference is admitted.

## Next experiments and remaining work

First test coupled shell/pad compliance against rigid-gap prediction across
thickness, stiffness, asymmetric and moving/nonconvex cases, preserving free
balance and expanding only numerical search regions. Compare support/pressure
before full stress and peaks. Distributed physical pressure should use a
conservative transfer; collider points still require a separately validated
physical region/pressure model and friction history.

For the separate every-step failures, existing DPR37 evidence has substantial
total normal force but eight zero-normal sources carrying about4–5e-22N shear;
SCR38 has zero normal force throughout and1.76e-20N unsigned shear. These are
still failed original histories. Test a bounded treatment derived from physical
force scales, quantify force/moment/stress effects, and rerun the actual path.
Finite lagged-normal inconsistencies and certified diverse-bank incompatibility
must remain rejected; they cannot be repaired by enlarging tolerances or patches.

Prioritize a useful validated DPR integration in Quadrants, then measure its
incremental module and scene cost against matching manual baselines. Current
native throughput evidence remains separate. SCR attribution/repair, coupled
automatic models, native implementation, measured tuning and the final fixed-
radius uncertainty/rank/figure/cost chain are outstanding. These first results
do not close the extended goal.
