# Native scene and module performance, 2026-10-19

The [current source-bound summary](NATIVE_PERFORMANCE_SUMMARY_20261019.md)
joins61 admitted repaired-source scene/cadence configurations with their exact
conformance and20 separate memory profiles. This page retains the original
engine's measurements and failures; its rates are not the current recommendation.

The [subsequent MPR contact report](NATIVE_CONTACT_CONSISTENCY_20261019.md)
localizes and repairs the motion discrepancy below:349 B1/B8/B32 checks and
three focused regressions pass on engine1ff39b077. This report retains the
original engine timings/failures. Repaired-source timings and larger memory/
conformance are separate; older rates are not relabeled as measurements of
the repaired engine.

All15 fixed-horizon production timing processes completed, and their independent
timing/source audit passes. Complete motion/output conformance does not:
32 of259 checks fail, all involving egg motion relative to pure rigid at B8/B32.
Retain these measured baseline rates with their unresolved admission. B1 covers
only the first frozen condition; its passing path is not an eight-condition
production recommendation. No physical-pad error is attached to these speeds.

## Production scene measurements

Each mode uses900 warmup steps and three1200-step repeats, dt0.01s/substeps1,
with native control, actual evolving rigid contacts and every-step max/full
GPU buffers. Each repeat covers12s of physical time. Full host field dumps,
research audits and trajectory observers are outside this primary timer.
Each batch's five modes share one GPU allocation; different batches use
separate RTX PRO6000 Blackwell Server Edition GPUs. Warm repeats are not fresh
process repeats. The eight-condition bank repeats by environment index:
B8 covers all eight conditions; B32 runs four independently simulated copies
of each. B1 exercises condition0 alone.

Values below are median valid env scene-steps/s. All45 timed intervals have
zero recorded invalid environment-steps and identical1200-step horizons.
These are scene steps, not module updates or policy transitions.

| Batch | Pure rigid | DPR max | DPR full | SCR max | SCR full |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 546.98 | 338.48 | 338.91 | 114.78 | 113.92 |
| 8 | 2,982.31 | 2,239.26 | 2,226.52 | 725.06 | 725.38 |
| 32 | 11,518.89 | 8,457.44 | 8,449.68 | 2,672.28 | 2,745.12 |

| Batch | Mode | Minimum–maximum env-steps/s | Median batch-step ms | Added ms versus paired pure rigid |
| ---: | --- | ---: | ---: | ---: |
| 1 | rigid | 544.29–547.85 | 1.828236 | 0.000000 |
| 1 | dpr-max | 338.09–339.14 | 2.954374 | 1.126137 |
| 1 | dpr-full | 338.68–339.04 | 2.950611 | 1.122374 |
| 1 | scr-max | 114.22–115.24 | 8.712039 | 6.883803 |
| 1 | scr-full | 113.83–114.56 | 8.777791 | 6.949555 |
| 8 | rigid | 2,964.80–2,986.73 | 2.682487 | 0.000000 |
| 8 | dpr-max | 2,236.05–2,239.32 | 3.572606 | 0.890119 |
| 8 | dpr-full | 2,219.37–2,228.20 | 3.593048 | 0.910561 |
| 8 | scr-max | 722.58–730.92 | 11.033645 | 8.351158 |
| 8 | scr-full | 720.48–725.78 | 11.028659 | 8.346172 |
| 32 | rigid | 11,436.20–11,523.35 | 2.778046 | 0.000000 |
| 32 | dpr-max | 8,447.93–8,459.22 | 3.783651 | 1.005605 |
| 32 | dpr-full | 8,443.03–8,461.65 | 3.787124 | 1.009078 |
| 32 | scr-max | 2,623.81–2,682.56 | 11.974798 | 9.196752 |
| 32 | scr-full | 2,686.00–2,802.38 | 11.657070 | 8.879024 |

Setup/JIT, exact source identities, all raw repeats, reset counts and logical
native/Torch allocations remain in the receipts. Logical buffer bytes and Torch
peak allocation are not total process GPU peaks; dedicated memory sampling
and larger-batch operating points remain required.

## Complete conformance and unresolved motion

Job728991 independently checks all1200 steps in15 cases,11 full-field snapshots
per full-output case, pure-rigid motion, max/full peaks and condition-matched
batch outputs. All recorded errno/validity, finite full tensors, independent
von Mises calculations and full-to-max reductions pass. Robot DOFs, contact
counts, max/full output agreement and matching-condition batch comparisons
also pass. The32 failures are the four egg-motion quantities for each of four
stress modes at B8/B32. The unchanged absolute limits are1e-10m/1e-10 quaternion
component and1e-9m/s or rad/s, with quaternion sign alignment.

Maximum discrepancies are3.9821794e-7m position,6.9215111e-6 quaternion
component,6.1357183e-6m/s velocity and3.0101747e-4rad/s angular velocity.
The first bit-level difference occurs at step109 for velocity/orientation,
then position at110; thresholds are crossed later at steps550–577. These
facts are consistent with amplified small differences but do not by themselves
prove harmless roundoff. All stress modes share the measured differences,
including SCR with fused rigid post disabled. Do not attribute the result to
fusion alone or relax thresholds to admit the timings.

Jobs728988/728996 complete fresh pure-rigid, stress-built/manual-unrequested
and DPR-fused-post-off controls and their full-path comparisons. The fresh rigid
repeat is bitwise identical to the original rigid path. Both manual-unrequested
and fused-post-off paths are bitwise identical to original DPR, reproducing the
same discrepancy from rigid. Recovery execution is therefore not necessary for
this difference; construction/control differences remain to be localized. Controller-only job728998 now verifies bitwise-identical joint/phase/grip/limit
outputs over all1200 steps with stress-radius emission off/on. This rejects
the controller-output hypothesis. Construction state, initial physical arrays
and nondeterministic constraint arithmetic still require targeted controls.
All original failures and unchanged thresholds remain retained.

## Changing-contact resident module replay

Job728981 and its independent audit728990 pass four B8 module modes. Each run
records a complete1200-frame actual scene history, then restores24 GPU input
buffers and the exact stress reset schedule through the unchanged native
association/reconstruction/recovery/acceptance path. All1200 replay peaks and,
for full output,11 complete tensor snapshots match their own recorded scene
within the unchanged rtol1e-4/atol1e-3Pa numerical replay criterion. This is an
internal numerical consistency criterion, not a physical stress-error bound.
All runtime invalid-update counters remain zero over the three timed histories.

| Mode | Median env-updates/s | Minimum–maximum | Median batch-update ms | Recorded input-buffer bytes |
| --- | ---: | ---: | ---: | ---: |
| dpr-max | 8,308.08 | 8,291.63–8,333.25 | 0.962918 | 1,240,128,000 |
| dpr-full | 8,200.35 | 8,148.85–8,217.64 | 0.975568 | 1,240,128,000 |
| scr-max | 974.49 | 973.91–977.06 | 8.209390 | 1,240,128,000 |
| scr-full | 980.56 | 979.79–981.74 | 8.158615 | 1,240,128,000 |

All four modes share one GPU allocation. Timing includes device-buffer restore,
its dispatch and native numerical acceptance; it excludes scene dynamics,
original recording, compression and independent validation. The histories
retain zero-load states, changing inputs and partial environment resets.
The independent input audit finds35201 loaded collider slots, maximum force
1.26667058N and maximum tangential component0.21117604N. These counts cover
all valid collider slots and are not an egg-only footprint or physical-pad
traction map. Complete compressed input arrays remain on data with exact hashes.

The earlier v1 replay was preliminary: it omitted explicit stress resets and
did not retain its raw replay buffers. Its scene-conformance recordings remain
useful, but its timing is not used in the module table. The v2 numerical module
admission does not resolve the separate pure-rigid motion difference.

## Measured bottleneck and profile-observer repair

Separate B8 frozen-snapshot profiles identify SCR intrinsic contact processing
at9.70751ms versus0.18542ms for its linear solve. DPR's corresponding pressure
weights/constraints stage is0.48787ms, scatter0.31643ms and linear solve0.18659ms.
These are intrusive, synchronized phase measurements of one actual snapshot,
not complete-history throughput. They must not be subtracted from, or summed
with, the primary scene/module timings. Kernel profiler output is retained;
large dispatch/host-synchronization overhead remains a hypothesis requiring
an appropriately scoped trace, not an already measured speedup.

The first SCR profile stopped at an obsolete observer normalization: it divided
group wrench error by net force after opposing sources cancelled. The production
solver already uses original source-force and moment-arm scales. The repair
changes only prepare_recovery_snapshot, independently reconstructs those scales
in NumPy, checks them against native values, and retains the same1e-8 gate.
An AST control proves every other benchmark function unchanged; no engine or
primary live timing code changes. Correct force/moment relative errors are
5.72563e-9 and1.48134e-10, while the obsolete force ratio is5.33928e-8 and
reproduces the original false rejection. Snapshot peak difference is4.59113e-9Pa.

The corrected profile uses the captured a465afbaf checkout plus the exact
observer patch; its inherited source-revision argument names the92692aa61
engine baseline. The captured source manifest/patch and per-file hashes,
not that inherited label alone, identify the executed observer.

## Evidence and remaining delivery

The [production packet](../evidence/20261019-qualified-ranking/native-production-measurements-v1/README.md)
retains full timing/conformance receipts, source manifests, exact worker copies,
all failures and both profile versions. The
[module packet](../evidence/20261019-qualified-ranking/native-module-replay-v1/README.md)
retains the admitted replay and independent input audit. Large JSON/logs are
losslessly gzip-encoded with both raw and encoded hashes. Full trajectories and
source histories remain on data; task JSON/workload scripts reproduce each run.

These native-collider runs use the original nominal6mm radius schedule and a
few collision sources, unlike the physical compliant-pad full-centroid2mm
configuration. Their physical-error cells remain pending. Next resolve motion
conformance, trace and optimize measured contact processing without bypassing
acceptance, extend practical batch/memory and sparse/output modes, and complete
matched configuration error/throughput delivery. Fine physical-reference and
friction qualification continue independently under the original full goal.
