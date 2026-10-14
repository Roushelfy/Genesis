# Nonconvex extension closeout audit

The active brief is `../GOAL.md`, `NATIVE_QUADRANTS_PLAN.md`,
`NONCONVEX_CONTACT_REQUIREMENTS.md` and `OUTPUT_MODES.md`. This audit preserves
that scope. It is a working acceptance record; the extension remains active
until the final regression, output-cost and boundary publications below pass
and the completed deliverables are committed and pushed.

Numerical source is `76537c2fb64cd469e83ba9921b0724b5c82a5f36`.
`4ae1d688` changes its documentation/evidence, with no changes to engine,
tests or examples. The normal example and benchmark use the production engine
without importing the research numerical solver. All production elastic
arithmetic, contact mapping/fitting/scatter, load balance, residuals and global
stress reduction execute in Quadrants. Build-time NumPy topology and offline
CPU FP64 oracles have separate roles.

## Requirement-to-evidence map

| Requirement | Implementation and direct checks | Closeout gate |
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
| Performance endpoint without changing fidelity | FP64/Q10/s8/body-source-prior law and unchanged wrench/residual/stress budgets; current-source batch curve and unchanged Jacobian guard | Final B38912 endpoint result or diagnosed resource limit, with no global-optimum claim |
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

## Current-source measurements and remaining checks

The six final matched four-scope widths and copied curve have finished.
B32768/623102 live/policy are145256.946/145431.227 valid env-step/s.
The same-GPU copied2048-condition curve reaches150133.237 at B32768.
The B36864 edge records pass source/bank/residual/outcome checks and give
154343.359/154176.223 live/policy with zero invalid steps, overflows or failed
lifts. Its separate GPU allocation is not a controlled incremental speedup.

B40960 fails during scene build at the existing Jacobian guard:
3678 constraints×15 DOFs×40960 environments exceeds signed-i32 indexing.
The exact guard limit is38924; B38912 is the last convenient complete
2048-condition multiple. Its actual live run fails during warmup when growing
the complete sample-position buffer, with CUDA_ERROR_OUT_OF_MEMORY. Its
failure/source/telemetry records are retained; it has no valid rate. The guard
and all contact capacities remain unchanged. Observed B36864 setup/workload peaks are92607MiB
live and94633MiB policy on a97887MiB device; these200ms samples are not exact
transient peaks and do not prove an out-of-memory failure.

Fresh full stress regression collects135 cases on each backend. CPU passes
114 with21 CUDA-only skips; CUDA is still running and has two saved-edge
fixture failures requiring diagnosis. Current-source max/full/max live/policy
measurements have finished at B1024 and B8192 on one GPU per width.
The first B1024 output-pair audit's1e-12m cross-process rigid-position
assertion fails with a largest9.802e-10m difference despite exactly matching
per-environment rigid/contact/grasp/descent counters. Its independent policy
max/max control differs by2.262269950e-6m, while the full/max difference is
9.801587389e-10m. The corrected postprocessing compares the mode difference
against that retained control plus transform roundoff, with exact integer
outcomes and the original peak-error gate. The live pair has exact positions.
The original failed audit and control data are preserved; force, wrench,
complete residual and stress gates remain unchanged. B1024 full/max ratios
are0.999829800 live/0.992281893 policy; full fields add110100480 bytes and the
separate peak/output trace rises from0.244007 to0.324171ms per step. The
policy comparison is a closed-loop rollout, not a pure output-kernel overhead.

## Publication audit

The current verifier checks97 existing checkpoint manifests,43120 artifact
entries and36 complete large-JSON reconstructions.43119 runtime originals
match exactly. One allocation log acquired an appended Pyxis attachment
message; its archived36 bytes are an exact prefix of the current109-byte log.
The verifier records both hashes/sizes and does not rewrite the original
manifest or treat the current file as unchanged.

389 missing identity files totaling7707118 bytes were restored from retained
runtime originals only after their recorded SHA256/size matched. This includes
logs and source files under ignored directory patterns. The immutable manifests
remain unchanged. `20261014-closeout/restored-artifacts.json` records each
restored path/hash; the scoped publication commit must explicitly include
those files. The complete verification result is
`20261014-closeout/publication-verification-v3/result.json` under the declared
runtime data root. Fresh measurements and the final publication need their
own subsequent verification.

Physical mesh/quadrature convergence remains a separate later study. These
measurements validate the inexpensive complete low shell on identical inputs.
Policy throughput is inference rollout, with the scripted grasp controller
and a small residual network, rather than complete RL training throughput.
