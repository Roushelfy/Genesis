# Valid-source-only capture initialization

The current-contact provenance kernel initializes source attributes only for
valid contacts. It preserves all valid capture, original application points,
collision pair/geom identity, physical radius/friction and error semantics.
Inactive source attributes are scratch and consumers already guard them with
the corresponding validity. Their stale values are not used for a new contact;
every newly valid source receives complete initialization before capture.
CPU and CUDA use this same rule. No numerical law or acceptance is changed.

The candidate is selected for clean production integration, which remains
pending with the other final candidates. Two same-GPU changing-contact pairs
use separate seed cohorts, FP64/Q10/s8/body_source_prior/max, dt0.01/substep1,
mesh64/GJK patch-on,600warmup and three1280-step repeats per route. All
observations, contact parameter updates, resets, metrics, history, fallback
and environments are included on RTX PRO6000 Blackwell Server Edition.

| Batch / seed | Baseline env-step/s | Candidate env-step/s | Ratio | Baseline repeat seconds | Candidate repeat seconds |
|---|---:|---:|---:|---|---|
|8192 /734001|101277.095|102234.301|1.009451355|103.649448 /103.385810 /103.570807|102.734098 /102.357539 /102.606266|
|16384 /991909|104658.347|105877.909|1.011652788|200.642769 /200.227887 /200.271650|198.351134 /197.927234 /197.939641|

The repeat ranges do not overlap in either cohort. Native buffers are unchanged:
7,656,775,242 and16,138,591,818bytes. Sampled device peaks including setup
are24245→23281MiB and46471→45521MiB; these200ms samples are not exact
transient maxima or evidence of a native allocation reduction.
Each paired bank agrees fieldwise, all per-environment rigid/contact/grasp/
descent counters agree, and final positions agree exactly. All repeats have
zero invalid steps, overflow or failed lift outcomes. Full metadata and all
per-environment records remain in the archive.

CPU31 cases pass in480.087s and CUDA31 in684.724s. The96 independently
checked full fields retain original wrench/local cones, signed P2 nodal load,
K and all six gauge rows, full stress/von Mises and global maximum acceptance.
Worst wrench error is7.386598484e-10, complete residual budget fraction
0.0458562252 and global peak difference8.301016351e-6Pa. Partial release,
reset, workspace growth, invalid sources and environment isolation pass.

The private source snapshot is based on production92ec3e0c, with a temporary
`_CAPTURE_VALID_SOURCES_ONLY` switch used solely for off/on comparison.
Production cleanup will remove the switch after combined qualification.
The gains are separate paired decisions and cannot be multiplied with the
sample-layout, Hessian or diagnostic-clearing experiments.

Archive `native-valid-source-capture-qualification-checkpoint-v1` preserves
both CPU/CUDA reports, independent fields, immutable source closures/patches,
the four complete live results, banks, telemetry, launchers and checker.
Original/stored/decompressed hashes and ordered large-JSON reconstruction
pass. Manifest SHA256:
`982dcf3d44065c9d36286aaef80257340cdc086aa1961b2aa543eb0416275cef`.

The extension remains active: clean combined qualification, remaining candidate
decisions, final-source dynamic profiling and the final four-scope/scaling
report are still required. Physical convergence and RL training are not claimed.
