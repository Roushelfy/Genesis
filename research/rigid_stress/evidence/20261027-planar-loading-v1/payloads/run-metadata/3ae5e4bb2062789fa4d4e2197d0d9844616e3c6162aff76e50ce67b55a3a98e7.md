# Planar loading surfaces: native DPR integration and throughput optimization

**Active user goal, 2026-10-27.** Make planar triangular loading surfaces a
usable, optimized path in the existing rigid stress feature. Start with the
validated CPU construction, preserve automatic-radius DPR and the existing Q4
elastic recovery, and maximize useful multi-environment throughput. Deliver
native code, accuracy evidence, detailed profiles and complete Franka results.

## Reference and intended change

Use [the CPU source](../triangular_load_cpu/README.md) and
[measured report](../triangular_load_cpu/REPORT_ZH.md), pinned at
[`79720355`](https://github.com/Roushelfy/Genesis/tree/797203553f246ca155bbef909c1bb33e7cd7b90c/research/rigid_stress/triangular_load_cpu).
The 75-case shell study keeps 1,452 elastic DOF. Loading subdivision k8/k16
gives worst field errors 1.546%/0.445%, worst sampled-peak errors
0.908%/0.244%, and paired complete CPU module speedups 1.56x/1.49x.
These are same-Q4 CPU load/solve/stress results, not GPU or RL rates.

The change concerns the physical loading/integration surface. Keep the Q4
shell stiffness, mass, gauge, inertia relief, precomputed displacement response
and whole-object stress recovery. Provide a small public configuration with
qualified defaults and a validated original-route fallback. Rewriting the elastic backend as a triangular
shell is not a prerequisite. Keep volume FEM available as a matched reference,
and apply shared optimizations to either backend where measured useful.

Prepare fixed inner/outer triangle surfaces and their sparse displacement maps
offline. For proxy motion u=Jq, transfer loads with f=J^T g. Preserve virtual
work, rigid motion, original-point force/moment and eccentric friction couples.
Loading refinement adds no elastic DOF. Match the CPU k8/k16 controls first,
then tune a cheaper compact or geometry-aware construction where justified.
Coplanarity alone cannot justify coarse interpolation: CPU k1 has large field
errors even on planar faces.

Keep the cheap Hertz radius estimator tied to the original geometry and
validated curvature information. A planar loading facet does not imply zero
physical curvature. Preserve source positions, normal/friction forces, sizing
inputs and the declared surface-ball Hertz pressure law. Reanchoring, radius
inflation or another pressure law would require a separately labelled study.

## Native correctness and contact robustness

Production floating numerical preparation and runtime stay in Quadrants,
inside Genesis's rigid stress feature. The CPU implementation is an independent
oracle. Reuse existing native assembly, response, acceptance and lifecycle.

Implement the exact contained-section moment path with its pressure-positivity
conditions. Retain accurate clipped/multiface integration and the nonnegative
pressure fit when those conditions fail. Map all finite-footprint contributions
back through the consistent shell load map, including both surface sides.

Preserve arbitrary moving and asymmetric contacts, normal force and friction,
convex/nonconvex material identity and continuous grasp histories. Keep current
collision provenance, actual anchors and original application-point moments.
Respect connected material sheets when nearby surfaces are spatially close.
Handle triangle seams, Q4 edges, junctions, release/recontact, slot changes,
partial reset, checkpoint restore and heterogeneous environments.

The CPU warped-surface miss is a required guard case. Use a qualified original
loading path or another validated safeguard when the proxy misses or its
coverage/geometry is unsupported. Record fallback frequency and cost. Force
and moment conservation alone do not certify the stress field.

Separate same-space numerical consistency from the proxy's loading/spatial
error. Keep original exact residual and load/lifecycle gates. Compare full
tensor fields, global sampled maxima and shell N/M/Q against the matching
original Q4 route, with an independently resolved integration control.
Freeze task error budgets before tuning, retain existing budgets, and target
the fidelity demonstrated by the CPU k8/k16 candidates. Test egg and bounded
convex/nonconvex controls, including remote hotspots. Keep every rejection and
uncertainty explicit. Physical continuum/contact convergence retains its prior
scope and is not a prerequisite for this performance phase.

## Complete optimization

Profile the integrated path before choosing kernels to optimize. Carry every
applicable optimization family through implementation, targeted tuning and
matched measurement, or record why existing evidence makes it negative or
inapplicable. Use engineering judgment on algorithms and scheduling. Cover:

- Fixed-geometry preparation, compact proxy/map storage, offline refinement
  and bounds, material-aware candidate queries and useful witness reuse.
- Contained analytic moments, early empty/zero-work removal, reuse across
  pressure fitting and scatter, and efficient clipped-boundary integration.
- Contact/environment packing, compatible work grouping, batched response,
  cooperative execution, memory layout and suitable batch scaling.
- Kernel/rigid-step fusion, CUDA-graph-compatible execution, device-resident
  scheduling, avoided copies/synchronization, global max/full output and
  fallback buffers. Preserve per-contact support and pressure semantics when
  aggregating loads.

Measure the actual bottleneck and combinations, rather than stopping after an
isolated fast integration kernel. Reuse current winners. Earlier scoped
Atlas/H/FP32/screen/history negatives remain recorded. Revisit one only when a
changed workload or measured bottleneck supplies a concrete reason. Low-model
numerical consistency and throughput work take priority over expensive fine
physical-convergence sweeps.

## Evidence and completion

Compare the planar path with the already optimized Q4 loading path, volume DPR
and pure rigid under matched geometry, material, controls, source bank, cadence
and output. Measure frozen-contact modules and complete Franka/hollow-egg
workflows across small and useful large batches, including the throughput
plateau where practical. Report max/full separately. Policy inference and
simulation throughput retain their exact measured scope.

Keep source/config/input identities, paired repeats, original runtime checks,
invalid steps and fallback counts. Separate primary wall timings from intrusive
profiles and oracle exports. Report load queries, integration/fit, scatter,
response/solve, stress/output, acceptance and scheduling. Include preparation,
shared/per-environment storage and sampled process/device memory with clear
scope. CPU speedups do not establish native GPU gains.

Completion requires the integrated planar loading feature, focused numerical
and contact/lifecycle qualification, a per-family optimization ledger, tuned
module and end-to-end accuracy/time/memory decisions, runnable evidence and
recommended configurations. Keep useful production changes, remove or archive
unhelpful experiments with restoration pins, and report remaining limitations.
A numerical port or documentation update alone does not complete this goal.
