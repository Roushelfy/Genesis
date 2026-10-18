# Sparse stress fields during complete mechanics

This checkpoint implements the user's few-keyframe observation requirement.
It retains complete spatial simulator fields, continuous mechanical journals,
actual CUDA traces, failed physical histories and failed equivalence checks.
It does not establish a qualified QIPC physical reference, a three-way accuracy
ranking or primary sparse throughput.

The keyframe plan was declared before the new sparse comparisons. From each
recorded controller history, it selects the nearest10ms phase observation to
loaded squeeze, lift, lateral motion, weak grip and release/recontact, with
the earliest tie. Prescribed times are1.33/2.77/4.02/4.38/5.52s. Original capped
cases0/2/7 additionally have a post-reset key, with actual delay0/0.74/2.59s.
Historical dense evidence was already available; this is not a blind-data claim.

## Complete prescribed candidates and numerical failures

| Worker | Continuous physical history | Full spatial fields | Finding |
| --- | --- | ---: | --- |
| DPR,726746_0 |6000×1ms,6s|5|Sparse fields match the old dense run at all five keys; actual robot/egg poses are identical.|
| SCR,726746_1 |6000×1ms,6s|5|All five same-method field/pose equivalence checks pass.|
| Stock Shell,726749_0 |6000×1ms,6s|5|Finite complete history; old-dense/new-sparse equivalence fails.|
| Volume-input surface2 Shell,726749_1 |6000×1ms,6s|5|Finite complete history; old-dense/new-sparse equivalence fails.|
| QIPC PCG1e-2,726748 |2113×1ms,2.113s|1|Newton reaches256; rejected and stopped, with static/native failure state.|
| QIPC PCG1e-3,726751 |2365×1ms,2.365s|1|Newton reaches256; rejected and stopped, with static/native failure state.|

Both QIPC runs hold Newton velocity tolerance1e-9m/s, PCG cap4096 and Newton
cap256 fixed. Neither has a detected ET intersection at its final step, but
each exhausts Newton. Last mapped directions are4.268063e-5m and2.049392e-3m,
far above the1e-12m displacement criterion. Finite geometry or short-window
success does not admit these histories. PCG1e-4 is a separately declared next
bounded contrast; it is outside this frozen packet.

The [earlier fixed-Newton contrast](../short-window-pcg-v1/README.md) and
[fresh repeats](../short-window-pcg-repeats-v1/README.md) explain why1e-2 was
tested: the1.05–1.16s window drops from59.888s at1e-6 to34.953s at1e-2, with
all three numerical keyframe bands passing. These new complete-history
failures supersede any assumption that the short-window candidate is ready
for complete-grasp use. The1e-6 baseline has not been shown necessary.

Local standard-PCG source tests
`abs(r_k^T M^-1 r_k) <= tol_rate * abs(r_0^T M^-1 r_0)`; with a fixed SPD
preconditioner the norm ratio is the square root of `tol_rate`. Newton tests
the mapped direction against `velocity_tol * dt` before CCD, rather than
checking a direct force residual. The two tolerances control different nested
solves. Residual-based adaptive inner stopping can avoid oversolving early
Newton systems, as described by [Eisenstat and Walker](https://epubs.siam.org/doi/10.1137/0917003).
The existing native Newton-aware composition entry gate is a different policy.
No new adaptive solver is claimed here.

Job726752 computes actual common-column/surface discrepancies at all five
keys against the historical complete QIPC candidate. That reference remains
physically unqualified and lacks the new full per-step ET receipt. At the
first four keys DPR/SCR tangent discrepancies remain near100%; stock Shell
is132.9–817.9%, and the volume/surface2 variant452.6–4458.8%. The release key
has a near-zero reference: relative errors are null and absolute errors are
retained. No five-key aggregate is called a whole-history norm. Raw stresses,
including large finite Shell values, remain unchanged.

Job726754 tests every new key against the corresponding old dense output,
with unchanged pose/field thresholds. DPR/SCR pass. Both Shell variants fail
all five; robot coordinates match but egg vertices and Cauchy fields do not.
This old/new comparison alone cannot attribute the difference to observation
cadence. Separately declared same-source control726772 completes six original-
state processes to1.16s, retaining18 complete spatial fields. Each setting runs
sparse-r0, dense-compute/sparse-export, then sparse-r1 in one allocation. Actual
stress-update counts are3/1160/3; all export the same three keys. Audit726776
fails every dense comparison and every unchanged-sparse fresh repeat. Stock
sparse repeats differ37.2–41.8% in the retained same-mesh tensor norm; the
volume/surface2 repeats differ101.1–107.3%. This demonstrates repeat variability
without changing sampling and prevents attributing the old/new discrepancy
to cadence alone. Large finite fields and all failed checks remain. Original
Shell stays unmodified; these are diagnostic outcomes, not accuracy rankings.

Job726777 repeats both five-key common-field analyses with snapshot payloads
disabled. The encoded history/coverage metrics exactly match726752, while no
large derived arrays are written. Exact comparison receipts are in `plans/`.

The captured instrumented Shell rows named force/moment averages over a
control interval actually divide impulses by1ms scene time. Their NPZ impulses
and values are correct; the correction JSON records the mislabeled interval.
Current source uses scene-step labels and explicit interval metadata. No raw
receipt was rewritten.

## Actual native recovery work and capped rejection

Job726738 warms each native setting over its original1.05s capped history,
checks the public mechanical reset exactly, then profiles a second continuous
history only over(0.85,1.05]s. It saves fields at0.89/0.95/1.05s. Native CUDA
recovery kernels execute200 times in every-step mode and3 in manual mode.
Job726744 verifies all three fields for both methods, with volume-weighted
four-corner tensor contrasts below1e-11 and exactly equal joint/egg poses.
Single intrusive profiled windows are0.739448/0.476063s for DPR and
1.345419/0.478180s for SCR, dense/manual. They are not primary throughput rates.
Complete CUPTI traces and kernel counts are retained.

Capped QIPC jobs726731/726735 run six bounded configurations from the original
state, preserving servo/passive/equality/contact/friction history and the
same0.85–1.05s window. The original partition setting takes392.992s; five
linear/looser-partition alternatives take2.567–2.936s. All six fail the final
provisional audit726741. The fast alternatives have zero physical finger
reaction at all three keys, while the original partition run has loaded keys
and itself exceeds the nodal-residual band at0.95s. The fast paths are not
qualified accelerations. No long tuned capped run follows from these timings.

Actual robot FK checks726737 pass for all four initial settings. At0.89s,
the original partition egg tilts0.646863rad while the standard linear-PCG egg
is nearly upright. Close robot tracking does not imply the same free-egg and
contact history. The cause of that divergence is unresolved.

## Exact evidence and source reconstruction

`campaigns/` retains22 terminal campaigns, including actual task/group return
codes, raw fields, journals, numerical audits, logs and original source receipts.
Expected negative outcomes are not overwritten as passes. The first native
analysis726742 failed only on a scalar `.item()` call; corrected726744 uses
unchanged physics/data/thresholds. Prescribed runner regression726736 failed
on an old facade's missing public-count attribute; its corrected726740 run
completes1160 steps with three fields. Both failures and corrections remain.

`selection-provenance.json` maps every copy to its original SHA256. Gzip
decodes byte-for-byte to the original. For each campaign, reconstruct every
analysis file from the frozen
[short-window source](../short-window-v1/source/) when `stored` begins
`prior/source/`; otherwise use the indicated content-addressed `source-blobs/`
file. Every reconstructed hash was checked. Backend Git revisions, original
patches/inventories, tasks, environments and hardware are retained separately.
`shared/` includes the exact common domain and selected historical reference
states needed to reproduce the new five-key diagnostic comparison.

The authorized runtime home fallback remains bounded to4GB with private
node-local caches. Ten derived common-alignment exports,129240196 bytes,
were retired after successful analysis with a hash/rebuild manifest; all
original full spatial simulator fields, rests, inputs and metrics remain.
The retired arrays are reproducible intermediates, not unique observations.
`plans/` records that retirement and all predeclared/correction metadata.
`capture.py` only copies, hashes and losslessly compresses bytes; scientific
calculations ran in the captured Slurm/container workers.
