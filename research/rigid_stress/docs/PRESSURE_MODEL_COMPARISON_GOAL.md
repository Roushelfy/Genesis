# Goal: qualify pressure and compare practical configuration choices

User direction, 2026-10-17, on `Roushelfy/Genesis:rigid-stress-recovery`.
Build sufficient current-source evidence for choosing the default pressure
route versus intrinsic body_source_prior. The branch cleanup is complete;
preserve its compact active layout and reuse existing accepted references.

## Outcome

Give users a defensible answer to: which configuration should I use, how much
does it cost over pure rigid dynamics, what accuracy/model limitations am I
accepting, and when should I switch? Pressure needs a current characterization,
not just older rates or the assumption that it is always cheaper enough.
Do not infer a pressure overhead from the recent intrinsic curve.

## Compare the actual options fairly

The main alternatives are `contact_model="pressure"` and
`contact_model="intrinsic", intrinsic_load_partition="body_source_prior"`.
Both retain normal and tangential contact forces and the same native elastic
problem. Pressure uses a supplied-pad-frame law; intrinsic enforces local
material-normal constraints and topology-aware support. Source-prior permits
measured local wrench redistribution while conserving the body aggregate.
Neither is an experimental contact-pressure ground truth.

Include a clearly labeled public-default stress configuration (only required
mesh and declared physical inputs supplied), the recommended intrinsic setting,
and worthwhile model-specific tuning. Record requested and effective options,
including history, integration, precision, fusion and output. Some intrinsic
parameters do not affect pressure; body_source_prior does not depend on
intrinsic_support. Explain these combinations rather than treating every
parameter as a separate solver or forcing irrelevant flags to match.

Freeze the current Franka/full hollow egg bank and scene as the main paired
workload:480 P2 tets/2430 DOF, material/mass, normal plus frictional contacts,
radius schedule, collider/contact solver, dt/substeps, controller/policy,
physical duration, observations and resets. Change the load model explicitly.
Use comparable or paired GPU allocations and repeated warmed timings.
Allow each mode's valid native fast paths; distinguish common-input model
comparison from additional tuning and fidelity changes.

## Measure performance, scaling and memory

Measure rigid, rigid+stress, policy+rigid and policy+rigid+stress for both routes
under the same conditions. Cover a useful small-batch point, the historical
around-B2048 regime, the recommended B16384 regime and the practical large-batch
shoulder/capacity. Choose intermediate widths and repeat lengths sensibly;
the existing intrinsic curve can be reused where source/options match.
For an efficient headline comparison, start with max output and history0.
Add bounded matched max/full costs so output-mode choices also have evidence.

Report raw repeats, valid aggregate env-step/s, batch latency, incremental
stress latency, memory and failures. State both the step-time ratio and
throughput loss relative to that mode's matched rigid baseline; distinguish
"50% more time" from "50% less throughput". Count all attempted environments
and reset/control/policy work in the declared scope. A rejected rollout has no
successful full-rollout rate; preserve its failure instead of shortening the
trajectory, discarding difficult environments or clipping loads.

Profile representative changing-contact workloads to explain differences:
support/anchor sampling, fitting, scatter, elastic recovery, checks, peak,
history, dispatch/synchronization and memory traffic. Separate cold JIT,
intrusive observers/profilers and capacity diagnostics from repeated rates.
If pressure has a concrete defect or material tuning opportunity, address and
remeasure it with focused checks; do not turn this into another exhaustive
optimization campaign or assume the intrinsic path is optimal for pressure.

## Establish useful accuracy and applicability evidence

Reuse independent FEM and the existing four-state R/L/M comparison. Verify
changed code with complete equilibrium/gauge, original wrench budgets, full
tensor/raw global maximum and lifecycle checks. Numerical agreement on mapped
loads remains separate from whether those loads represent contact adequately.

Compare both models on a compact representative set: the principal egg grasp,
a smooth convex control and a nonconvex/internal-hotspot or close-sheet case.
Include asymmetric/frictional, near-sliding and multiple-contact states,
self-balanced gripping, transitions/release and partial resets. Keep physical
footprints fixed for model comparisons and study a small declared radius/profile
variation separately. Neither convexity nor a successful solve alone qualifies
the pressure approximation.

Quantify peak/tensor differences, hotspot location, footprint support and
per-source force/moment redistribution. For pressure, report constraints in its
own supplied-pad frame and the separate diagnostic relative to local material
normals. Preserve known cone violations and wrong-sheet/topology risks.
For source-prior, quantify local redistribution and retain incompatible or
near-cancelling-wrench failures. Label these as model limitations, numerical
failures or unsupported inputs as appropriate; do not silently change the law,
radius, friction or tolerance to pass.

Retain bounded complete-grasp raw-stress/reward comparisons, with whole-episode
and named-phase results separated. Assess whether model/radius choices change
practical ranking or termination decisions. Treat source-prior as another model,
not truth, and preserve the existing reference uncertainty and peak bias.
Do not require universal high-mesh convergence, full RL training or physical
fracture calibration for this stage.

## Delivery and completion

Deliver one concise comparison report, scaling/memory and error/cost plots,
runnable commands and reusable summaries with source/configuration/hardware
identity. Include an easy-to-read decision table for pressure versus intrinsic,
default versus tuned settings and max versus full output. State where pressure
is a useful fast proxy, where the extra intrinsic cost is justified and where
neither option supports a trustworthy absolute damage claim. Recommendations
must follow the evidence; pressure may have a restricted operating range.

Keep entrypoints short and evidence compact. Preserve relevant failed results
and historical identities through the archive index rather than reviving old
campaigns or copying giant archives. Commit code, measurements and guidance on
this branch. Completion requires current paired data and a usable configuration
selection guide, not just a new document, unrelated historical rates, or a claim
that pressure is fast. Label unavailable runs and unqualified cases explicitly.
