# Fixed-Newton PCG stopping contrast

This supplements the [first sparse-keyframe window packet](../short-window-v1/README.md)
in response to the user's question: why require PCG1e-6 when Newton stopping
should control final nonlinear accuracy? The previous1e-6 setting was a passing
candidate; its necessity had not been established.

## Meaning of the tolerances

The native implementation sets the PCG threshold to
`tol_rate * abs(r_0^T z_0)` and stops when `abs(r_k^T z_k)` reaches it, or when
the separate iteration budget is exhausted. Here `z=M^-1 r` is the applied
preconditioned residual. With a fixed SPD preconditioner, the relative
preconditioned residual **norm** target is therefore `sqrt(tol_rate)`:

| Native `tol_rate` | Relative preconditioned residual norm |
| ---: | ---: |
|1e-6|1e-3|
|1e-4, native default|1e-2|
|1e-3|about0.0316|
|1e-2|0.1|

This is a recursive linear residual criterion, not a direct nonlinear force
or stress error. The configuration's short residual/rhs description is
imprecise. The exact executed native kernels and configuration source are
copied in `code-evidence/`; unchanged source/build identities remain in the
receipts and original inputs. The standard path starts from zero linear
solution, so its initial residual is the current right-hand side.

For an inexact Newton step, an inner residual condition controls how accurately
the Newton equation is solved; the outer nonlinear stopping criterion
determines whether the resulting state is accepted. Adaptive forcing can
avoid oversolving early Newton systems, as discussed by
[Eisenstat and Walker](https://epubs.siam.org/doi/10.1137/0917003).
That general condition and this implementation's preconditioned squared norm
must not be conflated. The current outer criterion compares the mapped vertex
direction before CCD against `velocity_tol * dt`, rather than directly testing
force balance. The retained actual residual/reaction/field checks are still
necessary. No Eisenstat–Walker policy or native numerical modification is
implemented in this experiment. The previous partition-PCG entry gate is a
different mechanism and had zero accepts.

## Controlled experiment

Job726725 runs four fresh processes sequentially in one allocation. Each starts
from the same original state and advances continuously to1.16s. Newton velocity
tolerance remains1e-9m/s, dt1ms, standard linear-PCG, double MAS and cap4096.
Only PCG stopping varies. The timed interval is(1.05,1.16]s; full spatial
stress observations remain the predeclared1.10/1.13/1.16s. Geometry, material,
contact/friction, finite-strength pose DBC and target history are unchanged.
This is prescribed motion, not the original capped-actuator protocol.

All four use whole-window boundary timing, one separately measured initial
read-only elastic activation, actual per-step robot tracking and native/
finite/J checks. Complete spatial host transfer stays inside the window;
compression/write is deferred. One run per setting is exploratory timing.
The companion job726726 compares full fields, physical finger reactions,
force residuals and actual iteration distributions against the same stricter
budget8192 candidate and the previously declared bands. All three keyframes,
including near-zero force at onset, remain in the comparisons.

## Measured result

All four settings pass the previously declared numerical/observable bands at
all three keyframes and complete1160 steps with no native invalid-step/ET/
Newton/line-search forced exit. Counts below cover the110-step timed window;
PCG-cap steps cover the full1160-step history.

| PCG `tol_rate` | Window seconds, n1 | Total PCG iterations | Total Newton iterations | PCG-cap steps | Three-keyframe bands |
| ---: | ---: | ---: | ---: | ---: | --- |
|1e-6|59.8877|1008832|1285|0|Pass|
|1e-4|50.5423|810215|1740|1|Pass|
|1e-3|47.4926|729291|2172|0|Pass|
|1e-2|34.9529|500367|2221|0|Pass|

The outer solve compensates for less accurate inner directions with more
Newton iterations, while total PCG work and measured window time decrease.
This supports using PCG1e-2 as the next bounded validation candidate instead
of treating1e-6 as necessary. It does not establish a universal optimal
tolerance or a repeat-averaged speed ratio.

At PCG1e-2, common tangent-field contrasts against the stricter candidate are
6.359e-6/4.322e-5/4.949e-5 at1.10/1.13/1.16s. Physical finger-force contrasts
are0/2.606e-5/2.351e-5, and net residual/egg weight is
8.846e-9/1.313e-5/3.033e-6. Nodal residual/contact norm is8.950e-5 at onset
and8.237e-9/8.963e-9 at the two loaded keys. Exact vectors, normalizations,
spatial peaks and all Newton/PCG/CCD/line-search distributions are in the
retained analysis; onset is not masked. One1e-4 step reaches the PCG budget
but the later outer state passes the declared checks; the cap is retained
rather than silently reclassified as linear convergence.

The next checks are fresh-process repeats of1e-2, capped-coupling short-window
validation and complete-grasp sparse observables. The present data justify
loosening the inner solve for that validation, with Newton and actual residual
checks retained. An adaptive residual policy is optional further work if the
remaining cost/robustness evidence calls for it.

## Reproduction and scope

The original [packet](../short-window-v1/README.md) holds the shared stricter
reference fields, surface/layout and independent zero-step target binding.
This packet retains all12 new complete spatial fields, full journals/timings,
analysis results, exact tasks/submissions/return codes, lossless logs and native
source evidence. Reconstruct each worker's Python source from that packet's
`source/` plus this packet's `source-variants/<campaign>/` overlay. The capture
helper verifies every executed analysis-file hash against that reconstruction.
`capture.py` only copies/hashes bytes and losslessly compresses journals; all
scientific computation runs in the captured Slurm/container workers.

Passing this window establishes only these numerical contrasts. It does not
qualify spatial/contact/timestep uncertainty, complete-grasp dynamics, capped
coupling, sparse DPR/SCR/Shell primary cost, or the full active goal.
