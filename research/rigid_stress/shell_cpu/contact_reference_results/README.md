# Independent halfspace contact control

`../contact_reference.py` solves axisymmetric contact of a rigid paraboloid
against an elastic halfspace using ring-constant pressure, midpoint collocation,
an elliptic-integral Green operator, and a primal/dual active set. Both pressure
and indentation are unknown. The total force is prescribed. No analytical
Hertz radius or pressure is supplied to the solver; analytical Hertz expressions
are used only after convergence for comparison.

This is an offline validation control, not a runtime pressure solver, and it
**does not establish thin-shell contact accuracy**.

Run with one BLAS thread:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m shell_cpu.contact_reference
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m shell_cpu.contact_reference --n 1280 --forces .005 .05 .5 --out shell_cpu/contact_reference_results/refinement1280
```

`results.json` retains N=160/320/640 at three loads. `refinement1280` contains
additional finest-grid results. The radial domain is 0.3 mm for all loads,
E=10 GPa, nu=0.3, rigid indenter radius=20 mm. The initial active set is a fixed
fraction of the computational domain, independent of Hertz prediction.

At N=1280, contact-edge radius differences are 0.0834%, 0.3649%, 0.0917% for
0.005/0.05/0.5 N. Edge errors are quantized by the ring spacing and are not
monotonic at every refinement. Continuous area-weighted pressure L2 differences
are 1.270%, 0.631%, 0.326%; these decrease from 2.498%, 1.052%, 0.618% at N=640.
The separately stored midpoint pressure error is smaller and should not be
mistaken for a continuous field norm.

Total-force residuals and discrete complementarity are approximately 1e-15.
A separate between-collocation displacement evaluation records the finite-grid
contact-gap defect; finest-grid negative gap / indentation is at most 7.26e-5.
It is not silently treated as zero continuum error.

`checks.json` doubles the outer domain at unchanged radial spacing: pressure
changes 1.04e-14 and indentation 1.59e-16 relatively, with unloaded outer edges.
Gauss orders 12/24/48 change pressure by at most 1.4e-9 relatively. The diagonal
Green singularity is integrated using analytical logarithm subtraction.
`initial_split_gauss` preserves the earlier direct split-Gauss integration run.

The companion NPZ files contain radius, pressure, gaps and analytical comparison
values. `halfspace_contact_control.png` visualizes the independently recovered
pressure and support. Main results and control summaries record the executed
source hash. This evidence says Hertz is recovered in its halfspace/parabolic
contact regime; it supplies no validation of shell thickness effects, arbitrary
nonconvex contacts, friction, or egg stress fields.
