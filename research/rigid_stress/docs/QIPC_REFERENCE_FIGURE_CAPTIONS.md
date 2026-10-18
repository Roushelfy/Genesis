# Captions and interpretation for the first QIPC figures

This inspected diagnostic checkpoint does not close the active reference goal.
Data/exports are in`20261018-qipc-reference/figures-v1`; actual export QA is in
`figure-exports-v1`. The[figure contracts](QIPC_REFERENCE_FIGURE_SPEC.md)record
claims and source evidence.

## New full-field cost

Full-field cost on RTX PRO6000 Blackwell, FP64, homogeneous case0 inputs.
Panels a/b measure complete10ms scene/control env-steps with10ms/1ms physical
steps. Native curves use the common host controller,900 warmup and three900-step
repeats. Error bars show observed minimum/maximum rates, not confidence intervals.
Shell default uses surface0/one substep/PCG50/tolerance1e-4/coarse10; fine uses
surface1/ten substeps/PCG300/1e-8/coarse1. Shell rates are diagnostic, without
physical reference qualification or an explicit invalid-step counter. Panel c
subtracts the applicable rigid median scene latency. Panel d uses separate
intrusive sampled process-memory runs, including measured8192-env capacity;
pending fine-shell points are not imputed. Normal timing excludes host field
exports, memory polling and profiling. Max-only and cold activation are separate.

Interpretation: batch scaling, physical substeps and small-batch latency all
affect practical cost. Tuned DPR helps at the largest measured batch; SCR uses
more time/memory. These conditional computational choices do not identify a
fastest physically accurate method.

## Complete prescribed-motion diagnostic

Complete6s common motion with force caps overridden. Black is the pinned
original-main SNH QIPC candidate plus the separately recorded16-epsilon numerical
line-search patch, DBC1e7,1ms step, velocity tolerance1e-9 and linear tolerance
1e-8. Blue/green are DPR/SCR; orange is nominal500um stock Shell. Panel a uses
physical mass-weighted COM: QIPC nodal masses, native authored URDF COM and
Shell area/thickness vertex masses. Panel b retains raw global VM at each1ms
physical step; zero has no logarithmic coordinate. Panels c/d use the same
material face-centroid column domain and exact one-sided inner/outer samples.
Relative denominators below1Pa reference tensor RMS remain undefined gaps.
Inner surfaces are dashed, outer solid. Tangent Cauchy discrepancies and full3D
raw peaks are distinct observables. These are deterministic time series, not
independent statistical samples. Spatial/contact uncertainty remains unresolved;
the figure does not establish a three-way physical accuracy ranking.

Interpretation: similar lift outcomes coexist with different stress histories
and release/recontact responses. DPR/SCR motion nearly coincides because stress
recovery does not feed back into rigid contact. Large low-load discrepancies
must be assessed with absolute tensors/reference RMS and state-history controls.

## Shared outer tangent maps

Outer tangent VM at1.2s squeeze and4.2s weak-grip onset for the same complete
histories. All panels share undeformed azimuth/height samples and one symmetric-
log colour scale, including zero, a1Pa linear region and the actual maximum
displayed value. No smoothing, percentile clipping or fitted footprint is used.
One-sided fields on1280 material columns are display samples, rather than a
full-volume norm or raw global3D extremum search. Source raw peak fields and
element/fiber locations remain separate. Reference qualification is open.

Interpretation: the shared scale exposes absolute field differences without
rescaling each method to its own maximum. It shows where load is distributed
and how weak grip changes it while preserving the difference between a plotted
surface invariant and an unresolved physical peak.
