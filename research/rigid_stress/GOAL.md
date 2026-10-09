# Agent goal: finish high-throughput rigid egg stress recovery in Genesis

Work in `Roushelfy/Genesis`, branch `rigid-stress-recovery`, based on upstream
`e9e1214d192914ddec85ca28014c459cfbc6c860`. Implement and validate a complete
Franka Panda grasp of a hollow egg-shaped rigid entity with auxiliary
fixed-geometry linear-elastic recovery of its global maximum von Mises stress.
Maximize **measured aggregate environment transitions/s** on the available GPU,
including many parallel scenes. Deliver working code and measured evidence;
do not stop at a proposal or isolated synthetic-load microbenchmark.

Read `research/rigid_stress/AGENTS.md`, its README and all docs first. Use the
included CPU reference/evidence as the correctness oracle and starting point.
Preserve the upstream engine's code conventions and the user's other branches.

Hard constraints:

1. Only the reference shape is fixed relative to contacts. Contact positions,
   forces/directions, counts, footprints and symmetry may all change. Use the
   **full shell**, no quarter/half symmetry or fixed finite contact basis.
2. Include actual normal and tangential friction forces and all support/table
   contacts. If spin/rolling friction is enabled, preserve its pure moments.
3. Smooth contact evolution over a grasp may be exploited for speed, but never
   required for correctness. Abrupt changes, zero contacts, slip, release and
   partial environment resets need independent correction/fallback.
4. Egg motion/collision stays rigid. Use auxiliary small-strain quasi-static
   elastic stress recovery; do not substitute deformable FEM dynamics. Fixed
   material/mass are explicit declared inputs for shared factors.
5. Return only a global scalar maximum per env (optional argmax for QA). No
   preselected hotspots; no full stress-field transfers in the GPU hot path.
6. Count all rigid steps, contact extraction/mapping, inertia relief, residuals,
   pack/scatter, correction, history maintenance, resets and policy work in
   their stated benchmark scope. No CPU solve or host transfer disguised as GPU.

Execute end to end:

A. Reproduce the CPU checks and inspect the original raw measurements. Port
   the shared-factor P2 math cleanly; keep full residual and gauge invariance.
   Add robust rank/conditioning handling and per-env reset tests.

B. Run and complete `franka_egg.py`: real approach/clamp/lift/hold/slide/release,
   shell-matched mass/COM/inertia, varied poses/friction/footprints and multiple
   independently phased environments. Verify actual lift/hold success and
   nonzero tangential force. Validate the egg-as-A/B force signs, authored
   transforms and matching substep contact snapshots.

C. Resolve the seed mapper's finite-patch/point-wrench discrepancy. Use a
   declared physically meaningful finite-footprint law accepting changing
   areas; preserve required resultant force/moment and test positivity/friction
   admissibility. Diagnose approximation error explicitly. Do not mask it via
   inertia relief. Include body forces and centrifugal inertia; compare
   inferred versus measured acceleration with complete Newton-Euler checks.

D. Build a real device-resident sparse direct baseline: one factor/operator
   group, many RHS, no per-env factor duplication. Compare cuDSS multi-RHS or
   exported sparse factors + cuSPARSE SpSM, with tested permutations/scaling.
   The dense debug Torch factor must not become the scalable backend.

E. Implement and benchmark fused full-domain P2 four-corner VM reduction,
   GPU contact mapping/scatter, previous-frame prediction, full residual,
   failed-env compaction/correction, circular Q/KQ/Gram history updates, and
   adaptive direct fallback. Optimize chunk/layout, FP32/scaling/refinement,
   stream synchronization and graphs where supported. TF32/Tensor Cores are
   candidates only after actual hardware accuracy/utilization measurement.
   Geometry-derived bounds/coarse spaces are optional and must retain proper
   residual/error validation; a fixed contact-space surrogate is insufficient.

F. Pass the complete suite in `docs/VALIDATION.md`: arbitrary/asymmetric
   frictional loads, live grasp replays, zero/all failed masks, rank-deficient
   history, resets, rigid-frame/gauge tests, precision and at least three full
   mesh levels with fixed physical footprint law plus quadrature convergence.
   Current coarse meshes/old quarter convergence do not satisfy this.

G. Report strict and throughput profiles from `ASSUMPTIONS.md` separately.
   Proposed targets: strict full relative residual <=1e-6 and max peak error
   <=0.01%; throughput residual <=1e-3 and max peak error <=1%, compared with
   identical same-mesh FP64 direct RHS. Report absolute Pa errors near zero,
   empirical limits, refinement/fallback count and physical mesh error. Do not
   relax tolerances or fidelity silently to claim speedup.

H. On the actual GPU, sweep B/chunks/history/precision and select the fastest
   validated configuration on calibration seeds, then freeze and validate on
   held-out seeds. Benchmark warmed >=10-second full-grasp runs, >=3 repeats:
   rigid baseline, stress-only, live rigid+stress, and matched batched
   policy-inference+rigid+stress. Report env transitions/s, batch steps/s,
   policy transitions, dt/substeps/sampling, GPU model/VRAM, stages and errors.
   Differentiate RTX 6000 Ada from RTX PRO 6000 Blackwell. Do not invent FPS.

I. Commit runnable one-command setup/demo/tests/benchmarks, configurations,
   raw CSV/JSON, a report with ablations and the chosen throughput/accuracy
   Pareto point, and one real-scene visualization/video or reproducible viewer
   command. Preserve baseline/direct paths and optional-dependency usability.
   Push the completed work to this task branch. Do not touch upstream main or
   the user's unrelated branches. No routine confirmation pauses are needed.

If target hardware/native libraries are unavailable, finish all implementable
CPU integration and buildable GPU work, leave reproducible device commands,
state the exact untested parts and blocker, and never label GPU acceptance or
throughput as measured. Otherwise continue through device tests and reporting.
