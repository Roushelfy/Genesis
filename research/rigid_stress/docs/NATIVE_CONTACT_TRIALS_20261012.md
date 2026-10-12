# Parallel contact clearing and bounded diagnostic trials

The native contact packer now clears raw status, errors, work counts and
certificate metadata in parallel over every original contact slot and
environment. The following per-environment scan still packs valid indices
in exactly the same order. Nodal initialization, all contact inputs,
numerical arithmetic and final acceptance remain unchanged. This adds one
Quadrants parallel loop and no buffer. The native function AST exactly
matches the independently measured candidate.

Actual native qualification passes **9 CPU cases in102.83 s and9 CUDA
cases in124.56 s**. These check complete shared-pair streaming, release,
partial reset, source-failure isolation, continuous certificates and
first-failure persistence. The numerical fitter is unchanged from the
55-case native CUDA epoch in `5510f21e` and the qualified canonical CPU
epoch in `c560c5e4`; no new55-case suite is claimed for metadata clearing.
The separately measured candidate also passes9 CUDA cases in101.05 s.

## Matched live cost

Each comparison uses one RTX PRO6000 Blackwell Server Edition GPU,
driver595.71.05,97,887 MiB, image `genesis:1_26`, Quadrants1.3.3 and FP64.
Hardware UUIDs are in the manifest. The low complete hollow shell has480
P2 tetrahedra,810 nodes,2430 DOFs and80 selected contact faces. Q10
squared-coordinate fans, s8 validated surface paths, verified single-face
disks and failed-disk nine-panel refinement remain fixed, with GJK mesh64,
contact patches, componentwise pair-union tractions, block4/effective8,
dt0.01/one substep, maximum output and warm cooperative fitting.

The condition bank SHA256 is
`156d956bc5fc3827da8df236cf1695a4bf30a9a1e1d939c672f6a57d69d10930`.
Both variants use B32/seed623101,600 warmup steps and three1280-step
repetitions, with37 saved-state resets in each repetition. Every environment
and all input updates, retries, controller work and resets remain counted.

| Variant | Repeated effective env-step/s | Aggregate env-step/s | Aggregate batch step/s |
|---|---|---:|---:|
| Native serial metadata clearing | 3009.073,3017.809,3019.262 | 3015.375 | 94.230 |
| Parallel metadata clearing | 3121.086,3133.556,3134.667 | 3129.758 | 97.805 |

Total counted steps divided by total elapsed time gives **1.037933x gain**.
Every repeat has zero invalid environment steps, overflow and per-environment
failure counts. Logged physical trajectories/contact quantities match exactly;
maximum logged peak difference is1.021e-6 Pa and the largest complete
residual budget fraction is0.006621. Both variants use167,466,930 native
bytes. This measures live rigid-plus-stress without a policy, not full RL
training or large-batch acceptance. It is not multiplied by gains from other
physical GPUs into a new end-to-end claim.

Separate enabled kernel profiling uses128 snapshot repetitions. Intrinsic
stage wall time is5.4338→5.1701 ms. The former serial metadata/validity
scan is0.644 ms; parallel metadata clearing is0.014 ms and the remaining
stable scan is0.274 ms. The added loop/scheduling cost and other stages
explain why rollout gain is smaller than that local saving. Snapshot
original points are unchanged, normalized wrench errors remain below1e-8,
nodal replay differences are below1.2e-16 N and global peak replay differences
below1.1e-8 Pa. Profiling wall times are intrusive and are not throughput.

## Rejected gradient-only trial evaluation

This FP64 candidate avoids the21-entry Hessian during line-search trials.
A nonconverged accepted trial must be revisited with its Hessian before
another Newton direction; that extra evaluation and all cold retry work
count against the same work budget. The candidate passes **55 CUDA cases
in654.34 s**, but its complete trajectory cost is worse.

| Variant, separate matched GPU | Repeated effective env-step/s | Aggregate env-step/s |
|---|---|---:|
| Native warm fitter | 2970.246,2939.519,2974.169 | 2961.230 |
| Gradient-only trials | 2711.868,2710.274,2713.831 | 2711.990 |

The resulting ratio is **0.915832x**, an8.42% regression; it is rejected.
The workload/model/budgets are identical to those above, all repetitions
retain every environment,37 resets, zero invalid steps and zero overflow.
Warmup peak and complete-residual checks pass. No extra intrusive live
CPU-oracle capture is needed to select a candidate that is not retained.
Snapshot fitting also changes0.471→0.476 ms on this GPU; it provides no
evidence to override the complete live result.

## On-demand boundary diagnostic prototype

A separate FP64 prototype generates triangle vertices, edge/disk roots,
disk centers and64 boundary-circle angles on demand for each original
source footprint. It evaluates a bounded six-dimensional separating-ray
search without allocating a diagnostic point/force cloud. These witnesses
only guide search; a finite witness cloud cannot establish continuous
incompatibility. The existing conservative subdivision checker supplies
the actual proof, with depth12/65,536-node bounds in this experiment.

For the exact policy tick876/environment24 component1, CPU and GPU find
a strict continuous certificate in101 search evaluations/116 visited nodes.
The continuous polar upper bound is about−3.495897e-4 and the normalized
target margin1.467016e-3, above propagated original-point error and roundoff
budgets. An independent CPU geometric audit includes every selected triangle
intersecting either unchanged Euclidean disk, without branch or orientation
pruning, and obtains the same negative upper bound. The successfully fitted
component0 retains its status and exact wrench errors. Original positions,
forces, moments, R=0.006465712642 m and actual μ=0.5196625772 are unchanged.

This checkpoint's diagnostic uses the previously audited clipped integration
prototype and remains unselected. Follow-up qualification with unmodified
native integration is separate. The strict proof identifies a limitation
of the declared componentwise law; it does not make normal-grasp failures
acceptable or authorize dropping contacts, changing loads or weakening
acceptance. Aggregate wrench partitioning, general legal boundary-crossing
integration, complete policy and1024/2048/larger four-scope measurements
remain open. No plateau or overall completion is claimed.

## Evidence

The [contact-trials manifest](../evidence/20261011-nonconvex/contact-trials-checkpoint-v1/manifest.json)
contains475 artifacts/11,486,756 bytes, SHA256
`72f8a7e80281a2a9f0cf3a057159dc652f6273813bcda0620a97886c957bbc5a`.
Allocated CPU publication verifies every source closure, original/published
file hash, matched physical metadata, numerical budgets and complete repeat
counters. Native qualification and rejected-candidate logs/source are
included. The archive contains the exact benchmark/qualification/diagnostic
launchers with explicit data-side cache/output paths. Runtime assets remain
under the established project data root. No high-resolution mesh or physical
integration-convergence study ran.
