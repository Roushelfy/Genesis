# Parallel-plate requirement audit, 2026-10-18

Specification: [remote goal](PARALLEL_PLATE_COMPARISON_GOAL.md), introduced
at fork commit `914a558dfa0257636e6d7af0af370928d952ff16`.
Delivery: [results](PARALLEL_PLATE_RESULTS_20261018.md),
[commands](RUNNING_PARALLEL_PLATES.md),
[closeout evidence](../evidence/20261018-parallel-plates/closeout-v1/README.md).

**Disposition: complete as a bounded comparison with explicitly diagnosed
unsupported physical-accuracy quantities.** The specification permits
“actual comparable, qualified simple-case results or an explicitly diagnosed
unsupported quantity.” No continuum stress ranking, successful SCR field,
settled original-Shell result or qualified cross-method speedup is claimed.
This is a negative qualification result supported by completed experiments,
not a promise that a newly submitted sweep will resolve the limitations.

Campaign paths below are relative to the data run root recorded in the guide.
Exact tasks, input/source hashes, receipts and raw-field references are retained.

| Requirement | Evidence and disposition |
| --- | --- |
| Full free egg, two flat plates, no robot/support pinning | `plate_*` scene adapters; actual initial geometry assertions; full egg and no fixed FEM nodes. Normal plate force is imposed and separation measured. |
| Verified force/motion mapping | Empty-plate backward-Euler checks in original/native backends; serial shear check passes after explicitly declared finite-guide stiffness control. Earlier unsupported API/geometry/armature/guide failures preserved. |
| Common material/rest geometry/mass and disclosed differences | `final-analysis-v1/group0/summary`: exact solid/native mass6.529848g; original Shell6.946250g,500um wall,2mm contact softness. Native friction floor0.01 and QIPC control are explicit. No fitted pressure radius. |
| Few small common-force equilibria and continuous history | `reference-loads-v1` and `native-equilibria-v2`:0.1/0.3/1N, all final50ms mechanical windows and endpoint equation checks pass. Different normal settling durations disclosed. Strict QIPC cap-free exit flag remains false where caps occurred. |
| Newton/PCG profiling and bounded tolerance study | `offdiagonal-controls-v1`, `localized-controls-v1`, earlier checkpoint: independent residual checks reject both tight and loose cap-sensitive cases. Linear quadratic residual criterion is distinguished from outer displacement correction. |
| Surface/thickness/settling/dt/contact uncertainty | Surface1->2 changes tangent61.27%, one loaded polar node per plate on both; wall2 long hold loses contact; wall8 equation fails; dt/2,mu0.01 and d_hat/2 controls audited. Physical contact field and raw peak convergence explicitly unsupported. |
| Independent same-load elasticity / rigid modes | `plate_audit` CPU P1 replay of actual nodal tractions; complete tensor differences2.42e-6/5.96e-6 on the two admitted surface grids. Six-mode gauge corrections, residuals, rigid motion and contact support retained. |
| Sparse complete fields and compatible metrics | `load-analysis-v1` and `closeout-analysis-v1`: complete endpoint arrays; common material tangent/membrane/bending/inner/outer fields; separate solid-only full3D metrics, absolute and relative differences, raw extrema and locations. Missing Shell transverse components and different peak sample locations explicit. |
| Preserve native/Shell rejection | Original SCR normal status40 and shear status33 dumps retained; no successful subset or acceptance relaxation. Original Shell zero-load/tighter-PCG/dt/longer-history controls isolate unresolved coupled equilibrium. |
| Friction/tangential discriminating control | `shear-reference-v1` and `shear-native-shell-v1`: identical2.5s history,mu0.3,10um motion. QIPC/DPR final windows pass measured gates; motion-time residuals and PCG caps remain failures, SCR fit rejects, Shell unsettled. No sustained-shear accuracy claim. |
| Off-axis/oriented case where useful | Not run. Optional here: it would not resolve the already diagnosed primary contact support/admission/coupling failures. General off-axis accuracy remains unsupported. |
| Cost, observation, scaling and memory | `native-cost-v1`: three warmed repeats atB1/B8/B64 with all-environment contact checks, sparse full-field transfers and engine admission. Separate native/QIPC profiles retain stage-name/kernel data and memory peaks. Failed QIPC primary cost retained, never used as accepted speedup. |
| Repeatability, load/unload and parameter stability | Repeated QIPC endpoint/tolerances; `native-stability-v1` six-load cycle and3/6/9mm radius controls. Zero-load residual force/peak and force floor explicit. No fitted selection. |
| Maps, curves, tables, commands and recommendations | Three common-scale diagnostic figures, color/grayscale/SVG/PDF; three-load field tables, uncertainty/stability/cost sections; exact captured task lists and runnable submission guide. Practical limits are stated per method. |
| Native architecture/original baselines preserved | Production numerical code unchanged by this plate extension. Native Quadrants integration and old accepted evidence retained. Shell observer evaluates frozen cached operator into owned buffers; plasticity/fracture disabled. QIPC is explicitly the inherited16-epsilon line-search roundoff diagnostic, not unmodified main; its patch/binary are pinned and prior original-main failures remain available. |
| Storage and execution | All new runtime data/logs under `/mnt/data/zhaofeng`; original home fallback unchanged. Scientific computation/plotting in rtx-mid Slurm via GENESIS_IMAGE_VER=1_26 gs-srun. Small source/docs/publication packet remain in Git. |
| Final validation | All11 plate modules pass Ruff checks/format, links and whitespace checks pass;185 copied artifacts and173 external identities verified. One explicitly superseded interim cost dependency retained. Final analysis/figures and two focused software smoke jobs exit0; visual color/grayscale QA passed. |

The requested physical-error and error-versus-cost ranking is unsupported for
three independently evidenced reasons: the candidate QIPC pressure footprint
is not spatially resolved, SCR rejects the balanced load reconstruction, and
original Shell does not attain coupled common-force equilibrium. The bounded
controls localize these failures; a global mesh sweep, force/radius fitting,
clipping peaks or dropping rejected methods would not complete that ranking.
Engineering cost and discrete elasticity conclusions remain useful within
their stated scope. Repairing the engines or establishing a continuum contact
reference is follow-on work, not an unreported success of this comparison.
