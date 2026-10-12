# One final traction projection for acceptance and scatter

The CUDA cooperative native route evaluates each final traction once for
original-point wrench, local inward Coulomb-cone checks and signed P2 nodal
scatter. Six resultant components and cone validity are cached per constraint;
the subsequent contact publication retains its unchanged1e-8 force/moment
acceptance and specific provenance/integration/fitting diagnostics. The
cache is typed current-step scratch, rewritten before each publication and
included in native enumeration. No per-sample force-cloud buffer is added.
CPU production and noncooperative routes retain their existing scheduling.

Componentwise loads now add each constraint's nodal contribution directly,
rather than first summing source components before atomics. Independent CPU
FP64 fields qualify this rounding difference. CPU23/CUDA23 cases pass in
296.500/422.860s, including source-prior full fields, arbitrary actual friction,
dynamic capacity, release/rebirth, partial reset, invalid-member recovery and
environment isolation. Their48 recorded source-prior fields have maximum
normalized wrench error7.386597889e-10, complete equilibrium/gauge residual
budget fraction0.045856225 and peak difference7.909635315e-6Pa. Six further
componentwise U/cavity cases pass on each backend in70.740/82.560s, checking
36 cached-scatter fields with independent P2Shell oracles, original wrench,
local cones and the complete residual including gauge rows. Dense, anchor-fan
and packed quadrature routes are exercised. The committed tests select the
same constructor/scatter calls that the archived test adapter exercised.

B2048/623102 uses the low complete hollow shell
(480P2 tetrahedra/810P2 nodes/2430 DOFs), FP64/Q10/s8, explicit
`body_source_prior`, maximum output, mesh64/GJK patch-on, dt0.01/substep1.
Both routes contain qualified EPA and selected seed/sample-major/port-bound
optimizations; exact capacity and device frontier count are absent in this
isolated comparison. Hardware is RTX PRO6000 Blackwell Server Edition,
97887MiB, driver595.71.05, container1_26, Genesis1.4.3/Quadrants1.3.3.

On GPU13316917,600-step warmup plus128 frozen profile passes give
16.977832→16.279721ms contact wall. Snapshot repeat differences remain below
6.661338148e-16N nodal force and1.269654604e-8Pa peak, and final normalized
wrench errors stay below1e-8. Native buffers increase6,111,942,287→
6,112,745,054 bytes,802,767 extra bytes for this workspace. The disabled
new cache keeps49 scalar bytes compared with the prior source.

The sequential same-GPU live pair runs600 warmup and three1280-step repeats
per route, with2323 selected-environment resets per repeat. Every repeat
includes controller/load updates, contact repair, reset, history and fallback.

| Final projection route | Repeat valid env-step/s | Aggregate valid env-step/s | Batch step/s |
|---|---|---:|---:|
| Separate acceptance/scatter |57170.411,57487.405,57408.514|57355.126229|28.005433|
| Fused projection |58050.828,58365.922,58243.724|58219.868707|28.427670|

Zero invalid environment steps and scatter overflows occur throughout. The
measured1.015076987x gain selects the candidate. These absolute rates belong
to their GPU pair and are not comparable as incremental gains against rates
from other GPUs. Subsequent selected-capacity/device-count combinations and
broader current-source scopes need their own qualification and measurement.

Recreate the archived `probe_fused_traction_acceptance_v1.py`, profile/live
pair wrappers and regression wrappers. The extra componentwise adapter v1
passes a NumPy scalar to Quadrants and triggers one CPU template-cache warning
treated as failure; that raw run and the cancelled, unexecuted pending v1
CUDA allocation are preserved. Adapter v2 passes the required Python float
and both backends pass, without changing any numerical assertion.

Archive `native-fused-traction-acceptance-checkpoint-v1` has782 entries/
141 unique blobs/29,557,011 stored bytes. An allocated CPU publisher verifies
all originals/stored/decompressed hashes, source closures, reports, recorded
fields, extra componentwise checks, snapshot budgets and complete live rows.
Manifest SHA256:
`08a4c4e2d589de1ee13c6c9c60c2a8937980588c149480f7211f5c285512a172`.

The goal remains active: dynamic profiling, actual tail costs, broad batches/
seeds and four scopes, remaining applicable optimization and a measured
plateau are pending. Physical convergence and RL training are separate work.
