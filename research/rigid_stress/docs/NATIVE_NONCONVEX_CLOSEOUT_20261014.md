# Nonconvex extension closeout audit

The active brief is `../GOAL.md`, `NATIVE_QUADRANTS_PLAN.md`,
`NONCONVEX_CONTACT_REQUIREMENTS.md` and `OUTPUT_MODES.md`. This audit preserves
that scope. Numerical, lifecycle, performance and publication acceptance is
complete. [The final report](NATIVE_NONCONVEX_RESULTS_20261014.md) gives the
measured configuration, errors, rates, output costs and resource limits.
This closeout is delivered to `Roushelfy/Genesis:rigid-stress-recovery`.

Numerical source is `76537c2fb64cd469e83ba9921b0724b5c82a5f36`.
The closeout changes validation, documentation and evidence. Its engine and
examples remain byte-identical to that numerical source. The normal example
and benchmark use the production engine
without importing the research numerical solver. All production elastic
arithmetic, contact mapping/fitting/scatter, load balance, residuals and global
stress reduction execute in Quadrants. Build-time NumPy topology and offline
CPU FP64 oracles have separate roles.

## Requirement-to-evidence map

| Requirement | Implementation and direct checks | Accepted evidence |
|---|---|---|
| Boundary provenance, collision geom/pair, reference face/barycentric and original application point | `provenance.py`, `source_vertices.py`; `test_stress_witness.py`, `test_stress_witness_manifold.py`, source frame/transition and ambiguity tests | Complete current-source CPU/CUDA regression |
| Owner-oriented material normals, cavities and arbitrary surface winding | `topology.py`, `surface.py`; `test_native_nonconvex_material_normals` and boundary-validation tests | Complete current-source CPU/CUDA regression |
| Finite topology-aware support, narrow-gap/branch/hole exclusion and declared distance approximation | `paths.py`, `surface_frontier.py`, `footprint.py`, `fan_refinement.py`; nonconvex path/profile, sliding, small-disk and refined-fan cases | Complete regression plus same-cloud field oracles; s8/Q10 is a declared approximation, not continuous geodesic or physical convergence |
| Original-point six-component wrench after projection and signed P2 scatter | `provenance.py`, `traction.py`, `cooperative_scatter.py`; rotated/offset/friction/torque fixtures and final dynamic original-wrench audit | Fixture gates and160 final-source independent original-wrench/inertia-relief fields |
| Physical compression/local Coulomb cones and explicit aggregate partition | `BODY_SOURCE_PRIOR_MODEL.md`, source-prior kernels; independent convex-dual CPU fit, self-balanced compression, different actual friction coefficients, multiple sources and workspace growth | Complete regression;48 inherited CPU/48 clean CUDA final-source fixture fields |
| Distinct ambiguity, unresolved integration, numerical failure and physical incompatibility; no clipping/truncation or silent radius changes | Source/footprint/traction statuses, continuous certificates and retained first-failure record; `test_stress_failure_diagnostics.py`, compatibility, disk-refinement, shared-lifecycle and source-prior rejection cases | Complete current-source regression and zero invalid/overflow counts in accepted dynamic workloads |
| Existing rigid lifecycle, shared immutable operators, independent environments, release/slip and partial reset/restoration | `recovery.py`, `model.py`, typed native data; ordinary rigid scene tests, shared-support/lifecycle and intrinsic checkpoint tests | Complete current-source regression; max/full observations invalidate and recompute through ordinary state notices |
| Full/max field API, global unaveraged peak and complete equilibrium including six gauge rows | `get_max_stress`, `get_stress_field`, full affine-P2 corner fields, complete K/gauge residual; full-field/unbatched/substep and independent-field tests | Complete regression,464 final-source captured dynamic fields and final-source matched output-cost pairs |
| Enabled-moment/support boundaries and optional-feature isolation | Unsupported rolling/torsional contact moments, joint supports, autodiff and undeclared applied/coupling loads raise explicitly; free-moment fitting is tested independently | Build/error tests, disabled native allocations and matched rigid-only scopes |
| Actual Franka/full hollow-egg workloads and matched four scopes | Original condition banks and exact source closures; six widths through B32768,600 warmup and three1280-step repeats; every environment, reset, observation and outcome counted |24 final-source scope results plus completed copied curve and edge audit |
| Detailed profiling, tail work, memory and all material optimization decisions | Final B16384 stage fences, per-environment work arrays, native buffer inventory, assigned-device telemetry and candidate/ablation publications | Final bottleneck/decision report; profile timings remain separate from headline throughput |
| Performance endpoint without changing fidelity | FP64/Q10/s8/body-source-prior law and unchanged wrench/residual/stress budgets; current-source batch curve and unchanged Jacobian guard | Diminishing copied-curve returns, B36864 accepted trials, B38912 memory failure and B40960 address guard; device/workload-specific endpoint |
| Reproducible source/tests/example/commands/evidence and task-branch delivery | Normal engine/example/test paths; archived launchers, banks, JUnit reports, numerical properties and hash/chunk verifier | Final artifact verification, documented reproduction, clean scoped commit and remote-head verification |

## Verified existing numerical evidence

`native-sampler-warp-qualification-checkpoint-v1` contains35 inherited CPU,
35 prototype CUDA and35 clean-source CUDA cases. Its clean fixture fields
have normalized original wrench error at most7.394051316e-10, complete
residual budget fraction0.04585622523 and peak difference7.967217243e-6Pa.
The complete clean-source CPU numerical path is unchanged by the CUDA sampler.

`native-final-sampler-dynamic-checkpoint-v1` independently checks464 complete
fields, including the complete K/gauge system and full tensor/von Mises/global
peak. Worst residual budget fraction is0.00563800285 and peak difference
1.860404154e-6Pa. `native-final-sampler-wrench-oracle-checkpoint-v1` adds160
fields from original forces/application points/free moments, its own mass and
centrifugal/inertia-relief calculation: original aggregate wrench error
8.627857791e-9, residual budget fraction0.00551764678 and peak difference
1.540160156e-6Pa. Final wrench1e-8, residual
`max(1e-11N,1e-8||rhs||)` and stressRTOL1e-4/ATOL1e-3Pa remain fixed.

## Current-source regression and source provenance

The complete initial regression collects 135 cases per backend. CPU passes
114 with 21 explicitly CUDA-only skips. CUDA passes 133 and fails the two
saved-edge fixtures at tick147. The complete reports and failures are retained.
The failed test compared native elastic displacement against loads from a
separate CPU traction fit, combining contact-fitting and elastic error budgets.

The corrected `tests/rigid/test_stress_surface_fan.py` independently balances
the actual native nodal loads before evaluating the full CPU K residual,
including every gauge row. Independent CPU traction fitting, original force/
moment, local cones, signed nodal loads, complete tensor/von Mises and global
peak comparisons retain their original gates. Its 10 cases pass again on both
backends (CPU94.181s, CUDA217.616s). The two problematic CUDA fixtures have
complete residual budget fractions at most0.002480. JUnit properties retain
the separately fitted RHS difference and actual equilibrium error.

Combined current-source coverage is therefore 135 CUDA passes and 114 CPU
passes/21 CUDA-only skips. The final collector matches every case identity and
proves every other recorded source byte-identical. It preserves the original
failed suite. The production numerical engine, wrench1e-8, residual
`max(1e-11N,1e-8||rhs||)` and stressRTOL1e-4/ATOL1e-3Pa remain unchanged.

The initial launcher omitted code copies because its container lacked `rg`.
Its recorded Git revision, empty source patch and actual import path are
retained. Exact initial Git sources were subsequently copied and explicitly
labeled as a post-start provenance supplement. The corrected launcher records
524 source/fixture paths before and after the fresh module runs. Final result
and publication launchers also retain identical before/after script hashes.

## Current-source measurements and resource boundary

Six final matched four-scope widths through B32768 and the copied curve
finish. B32768/623102 live/policy rates are145256.946/145431.227 env-step/s.
The same-GPU copied2048-condition curve reaches150133.237 at B32768.
B36864 edge trials pass source/bank/residual/outcome checks and give
154343.359/154176.223 live/policy with zero invalid steps, overflows or failed
lifts. Its separate GPU allocation prevents a controlled incremental ratio.

B40960 fails during scene build at the existing Jacobian guard:
3678 constraints×15 DOFs×40960 environments exceeds signed-i32 indexing.
The exact guard limit is38924. B38912, the last convenient complete
2048-condition multiple, fails during warmup when growing the complete
sample-position buffer with CUDA_ERROR_OUT_OF_MEMORY. Its failure/source/
telemetry records are retained without a valid rate. All guards/capacities
remain unchanged. B36864 sampled setup/workload peaks are92607MiB live and
94633MiB policy on a97887MiB device. These200ms samples are not exact
transient peaks. B36864 is the largest successfully measured complete bank,
with no claim to the largest feasible individual integer batch.

Final-source max/full/max live/policy measurements finish at B1024 and B8192
on one GPU per width, with600 warmup and three1280-step repeats per route.
Exact banks/configuration/network and per-environment rigid/contact/grasp/
descent outcomes are verified. Live positions are exact. Full output adds
107520 bytes/environment. Pooled-control live/policy rate costs are
0.017%/0.772% at B1024 and0.739%/0.745% at B8192.

The policy consumes stress observations and evolves independently on CUDA.
Its cross-process final positions and stress peaks describe different states.
The initial1e-12m cross-process assertion and subsequent single-control-floor
assertion fail and are preserved. B8192 full/max positions differ by223µm,
with65µm in the same-mode max/max control. Peak differences are9.496Pa and
199.072Pa respectively. The final audit retains these as rollout diagnostics.
Same-state field accuracy remains established by the strict tests and
independent CPU field oracles. Policy measurements report complete rollout
cost rather than isolated output-write overhead.

Separate intrusive50-step peak/output traces rise from0.244007 to0.324171ms
at B1024 and1.750063 to2.287547ms at B8192. Profile/trace/setup work is
excluded from headline wall timing. The final bottleneck/candidate inventory
reviews fit/geometry, complete inverse/scatter, rigid work, packing, sampling
storage, history, residuals, precision and scheduling. The curve, resource
limits and measured candidate decisions establish a practical endpoint for
this device/low-mesh/law. Further selective algorithms or backend memory/
addressing changes are separate studies.

## Publication audit and retained failures

The first verifier checks 97 existing checkpoints, 43120 artifact entries
and 36 complete large-JSON reconstructions. 43119 runtime originals match
exactly. One allocation log acquired an appended Pyxis attachment message.
Its archived36 bytes are an exact prefix of the current109-byte log.
The verifier records both hashes/sizes separately.

389 missing identity files totaling7707118 bytes were restored from retained
runtime originals only after recorded SHA256/size matched. This includes
logs and sources covered by ignore patterns. Immutable manifests remain
unchanged. The restoration journal and full verification result are retained
under `20261014-closeout` and in the final publication. Those restored files
were explicitly committed in `0cd0419b`.

The final Git-inclusion audit also finds105 already present archival files
covered by ignore patterns (588783 bytes). They are explicitly included in
the closeout commit after hash verification. Every artifact path from all98
manifest schemas is now checked against the Git index.

The final
[closeout manifest](../evidence/20261011-nonconvex/native-nonconvex-closeout-20261014-v1/manifest.json)
has 6997 entries. All 6997 stored/decompressed files and runtime originals
verify exactly, with two complete large-JSON reconstructions. Manifest SHA256:
`cf30881e3185fa62d05d0460dd67bb34d6de77a6ad52fb8760cbf8d7a762798e`.
It preserves 12 output routes, both initial regressions and corrected-module
reruns, B36864 live/policy, both diagnosed capacity failures, numerical JUnit
properties, exact source copies/patches, analysis/publication launchers and
the failed preliminary audits/launches. Large runtime JSON is published as
ordered32MiB byte ranges with whole-file reconstruction hashes. Raw data and
staging stay under the declared data root.

The final allocated audit of all98 publications passes50117 entries,
50116 exact originals, one exact archived allocation-log prefix and38
complete large-JSON reconstructions (job711619, exit0).
The [complete verification report](../evidence/20261011-nonconvex/native-nonconvex-closeout-20261014-v1/all-publications-verification.json)
and [new checkpoint verification](../evidence/20261011-nonconvex/native-nonconvex-closeout-20261014-v1/verification.json)
are derived closeout records outside the immutable manifest's own file list.
Archive bytes are preserved even where historical source/log whitespace
differs from current formatting rules.

## Reproduction and delivery

`docs/NATIVE_API.md`, `docs/RUNNING.md` and the final report document normal
feature use and exact low-mesh benchmark/test protocols. The source includes
portable test/output launchers, the result collector, bounded publisher and
hash/reconstruction verifier. Runtime paths are configured through
`RIGID_STRESS_DATA_ROOT` and `RIGID_STRESS_ENV`.

The closeout commit contains the tested validation change, documentation and
all final archive files, including artifacts covered by ignore patterns.
Engine/example source is unchanged. Delivery uses a normal push to the task
branch, followed by matching `git rev-parse HEAD` to
`git ls-remote fork refs/heads/rigid-stress-recovery`. The existing primary
checkout and unrelated jobs/branches are preserved.

Physical mesh/quadrature convergence remains a separate later study. These
measurements validate the inexpensive complete low shell on identical inputs.
Policy throughput is inference rollout, with the scripted grasp controller
and a small residual network. Complete RL training remains separate.
