# Validated terminal saved-state continuation for the physical solid reference

The coarse control completes the original400-step physical history as200 saved
steps plus200 newly integrated steps. Independent aggregation730363 passes33
checks and the strengthened input controls730366 pass15. This establishes an
endpoint-continuation control, not a qualified fine reference or completion of
[the active goal](QUALIFIED_STRESS_RANKING_GOAL.md).

## Restore physical state and keep lineage

The affine P2/StVK reference stores full nodal displacement and velocity,
current point forces and fixed projected reference areas in each completed
endpoint. The matching journal stores plate displacement/velocity and global
step/time/load. These quantities retain all free body modes, the plate state
and the lagged normal force used by the next friction step. The model is
hyperelastic; its implementation caches are rebuilt from the restored state.
The modal solver warm impulse is reconstructed as the physical point force
multiplied by dt. Its possible floating-point warm-start difference remains
subject to the original complete-compliance convergence checks.

`solid_pad_checkpoint --run TERMINAL_PARENT --level 0 --output NEW_RECEIPT_DIR`
is stdlib-only scheduler/file inspection. It queries actual Slurm accounting
and rejects a still-running or ambiguous parent before reading checkpoint
state. The original half-dt729314_1 is an executed live-parent negative. Parent
files and its original submission stay intact. The receipt binds complete
journal bytes, selected endpoint fields, original arguments and ten numerical
source modules, including independent FEM assembly. Subsequent validation
requires all original prefix mechanical gates and complete equilibrium windows.
All constitutive, solver, mesh, contact and load-history arguments match.
Only module documentation is ignored in semantic comparisons; every executable
AST node is retained, and each signature is bound to its actual captured file.
Missing or aliased numerical modules are rejected.

`plate_solid_pad` accepts `--resume-checkpoint NEW_RECEIPT_DIR/result.json` in
addition to the unchanged full original physical arguments. Run it only through
the captured Slurm workflow, at a fresh data-side destination. Full float64
body/plate state and lagged normal point force are restored. Saved geometry,
quadrature, material and force/moment totals are checked. Prior endpoint files
and journal prefix are copied byte for byte; global time, level and force indices
continue rather than restarting the ramp. `start.json` records the separate
resume provenance. A continued run still needs every new physical step, both
complete windows and its own independent momentum/field audits.

Result metadata states inherited and newly executed steps separately.
`elapsed_seconds` covers the newly executed segment only and cannot be divided
by the combined history length as a whole-trajectory performance measurement.
This validation infrastructure is separate from native production timing.

## Executed numerical control

The original canonical-force coarse reference729267 remains the uninterrupted
comparison. The new730362 restores its0.2s endpoint and executes through0.4s,
including the actual tangential ramp and sustained support. All400 journal
records pass the original equation, normal/friction law, boundary and current
moment gates. Both equilibrium windows and the own independent current-momentum
audit pass. Maximum equation/friction residuals are1.11535e-9/1.61325e-16.

| Complete comparison | Measured change | Predeclared limit |
| --- | ---: | ---: |
| Full nodal displacement, relative L2 | 3.97801e-10 | 1e-8 |
| Full raw stress tensors, relative L2 | 4.67486e-10 | 1e-8 |
| All point forces, relative L2 | 9.88606e-10 | 1e-8 |
| Maximum nodal velocity difference | 2.46566e-12m/s | 1e-10m/s |
| Largest of five volume/area-weighted compatible-field changes | 6.07596e-10 | 1e-8 |
| Largest raw Mises/tensile peak change | 4.25107e-10 | 1e-8 |
| Largest common Mises/tensile peak change | 4.25254e-10 | 1e-8 |

Common fields retain q9 angular/depth4 physical-volume integration and complete
inner/outer, membrane and bending tensors. Raw and common peak definitions are
separate. No polar trimming, force removal, state clearing, contact change or
tolerance relaxation is introduced. The1e-8 field/peak limits were fixed before
execution. The saved-prefix SHA and copied normal-frame SHA remain exact.

Initial input controls pass13 checks. The later15-check version additionally
requires complete numerical-module coverage and binds each semantic signature
to its captured source; it rejects changed dt, gravity, radius, load schedule,
frame/prefix hashes, prefix length, numerical implementation, missing/aliased
modules, live parents and a checkpoint with no remaining steps. The state
restoration function has the identical AST and the reference driver identical
bytes in both versions. The source difference/identity journal preserves this
provenance strengthening without rerunning unchanged integration arithmetic.

## Fine-reference status at the original continuation checkpoint

No live fine task is cancelled, restarted or modified by this control. The
half-dt729314_1 remains subject to its original15h limit after the denied36h
extension request. If it terminates before completing800 steps, continuation
can be considered only after scheduler-terminal verification and availability
of its complete normal endpoint. Any replacement must retain original failure
records, explicit saved-prefix lineage, the full800-step physical history,
unchanged original gates and independent final spatial/time/contact tests.
Pending field/qualification plans must then be replaced explicitly with the
new source/frame identities. A successful coarse continuation cannot itself
admit a partial fine history, waive a missing endpoint or qualify resolution.

`audit_solid_pad_resume --run CONTINUED_RUN --parent UNINTERRUPTED_CONTROL
--comparison COMMON_FIELD_RESULT --contract PREDECLARED_CONTRACT --output
NEW_DATA_PATH` reproduces the coarse equivalence aggregation. Source snapshots,
receipt preparation, original live refusal and both control versions are
retained. Large original fields and copied endpoints remain under the ranking
data root; small receipts/source/metadata may be published.


The [endpoint-continuation packet](../evidence/20261019-qualified-ranking/solid-pad-endpoint-continuation-v1/README.md) retains99 compact payloads/496,923 bytes with exact original/encoded hashes, executed control variants and source/proof lineage. Three full state/column arrays stay on data with external hashes. No live fine field is frozen or admitted by this packet.


## Actual fine preemptions and periodic state output, 2026-10-20

Slurm729312_1 is PREEMPTED after321 complete steps. The other two original
parents729314_0/_1 then stop after342/418 records during further preemption
on the same node. The full saved normal endpoints provide200/200/400 steps;
all original later records remain preserved. Separate terminal-only receipts
bind these endpoints and all ten numerical modules. The first revised launch
730868 was cancelled during initialization to remove an invalid in-group
terminal check; it has no reference start/journal. Corrected730903 is later
preempted during initialization, also before integration. Both attempts remain
visible rather than being counted as complete fine trajectories.

The current CPU-only continuations730978/730979/730980 use fresh data outputs
and inherit200/200/400 steps. Required new integration is200/200/400 steps.
They use three disjoint recorded node pools and exclude the repeatedly preempted
node through the runner's job-specific --exclude-nodes option. Shared cluster
configuration, numerical parameters, environments and existing links stay intact.
rtx-mid remains a preemptible execution environment.

plate_solid_pad adds --checkpoint-stride, default0. The current runs use25.
Each periodic file retains FP64 body displacement/velocity, point forces, fixed
geometry/quadrature, physical body/inertial force arrays, plate state and global
clock/load metadata. Complete stress fields retain their existing endpoint
outputs. The saved-state restoration function has the same AST as the validated
coarse implementation. Removing only the new CLI/validation/directory/write
nodes gives the identical prior reference-driver AST. No integration arithmetic
or physical acceptance threshold changes.

Actual coarse control730971 completes the original400-step history as200
inherited+200 new steps with periodic output and passes all33 full-array/common-
field equivalence checks. Three fine input controls pass15 each, including the
half-dt400-step prefix. Input controls now accept --expected-prefix-steps;
physical and source validation are unchanged. Continued-parent metadata passes26
controls. Eight periodic states cover225–400 in steps of25. Independent inspection
checks finite complete body states, exact clock/plate/point-force records, current
force/moment balance and final endpoint-array identity.

The initial periodic reader finishes its computations but fails at JSON encoding
of NumPy booleans. Its group0 returncode1 and original log/source remain intact.
Read-only730990 converts booleans at the output boundary, rechecks identical
arrays and all original scientific-control receipts, and passes99 checks. No
reference integration is rerun for this metadata correction. The fine jobs
depend on this independent successful inspection; captured task/source bytes
remain unchanged when their scheduler dependencies are updated.

Current continuation still uses the validated complete-phase endpoints. The
periodic-state files retain data for later recovery; non-endpoint preparation
and restoration require a separate provenance/trajectory control before use.
Segment elapsed time includes its own output work and excludes inherited
history/setup. Preserve original failed/retried costs; no whole-trajectory speed
claim follows from a shorter resumed segment.

[The complete replacement queue](FRICTION_REFERENCE_QUALIFICATION_PIPELINE_20261019.md)
retains seven-parent admission,36 comparisons,18 envelopes,54 final error rows/
18 metric ranks or ties, guarded figures and matching costs. Current source/
CLI/path/placement/dependency checks pass55. The
[recovery packet](../evidence/20261019-qualified-ranking/canonical125-resilient-recovery-v1/README.md)
retains794 payloads/5,326,422 bytes and18 external arrays. All live fine outputs
stay on data; continued-parent admission, fine qualification and final delivery
remain incomplete.

## Partial-level recovery and repeated-continuation ancestry

Terminal-only preparation additionally accepts `--step N` or `--latest`, mutually
exclusive with `--level`. The latest selector checks compressed archive integrity,
skips an interrupted newest write and excludes the final step. A state receipt
binds the exact selected file, full parent journal, inherited prefix, completed
endpoint frames and the same ten numerical modules. Original endpoint receipts
remain valid. Restoration checks saved step/time/level, load and plate state
against the journal before restoring full body displacement/velocity and lagged
normal force. Only completed phases need a full equilibrium window; partial
phases retain every original per-step mechanical gate and global clock/load.

Actual Slurm731095 restores step300 of the independently admitted periodic
coarse control and executes100 new steps. Both equilibrium windows and own
current-momentum audit pass. All34 conformance checks pass against the original
uninterrupted400-step baseline, retaining the prior1e-8 field/point-force/peak
limits and1e-10m/s nodal-velocity limit. Full displacement/stress/point-force
differences are5.84197e-10/3.04605e-10/8.68371e-10; maximum nodal velocity
difference is2.28258e-12m/s. The largest compatible-field change is4.70562e-10.

Input controls pass15 and partial-state selection controls14, including changed
state bindings, missing completed frames, shifted steps, live-state refusal,
private truncated-write fallback and final-state refusal. Legacy endpoint
lineage passes30 controls. The actual300+100 run passes49 ancestry controls
through its saved200-step ancestor, including changed ancestor provenance,
mass/output/job/frame/prefix and a cycle. Each inherited prefix must advance
strictly. Intermediate prefix validation does not require a completed parent
result; the final continued trajectory still needs complete admission and
honest inherited/new segment accounting. An explicit `--baseline` keeps the
uninterrupted comparison separate from the immediate saved-state parent.

The physical driver and all ten numerical modules have identical bytes to the
previous checkpoint. Live fine job captures remain unchanged. This extension
changes state selection, validation and ancestry; it does not admit the fine
reference or alter its pending spatial/time/contact gates.

Slurm731101 additionally restores step225 during the tangential-loading phase
and integrates175 new steps. It also passes all34 conformance and49 ancestry
checks against the uninterrupted400-step baseline. Complete displacement/stress/
point-force changes are4.30784e-10/2.75502e-10/4.98506e-10, maximum velocity
difference1.05699e-12m/s and maximum compatible-field change4.96540e-10.
Periodic packing passes32 checks for the four new step300-control states and44
for the seven new step225-control states. Both own independent momentum audits
pass; all wrappers exit0. The original periodic reader's returncode1 remains
unchanged and its independent99-check admission is rehashed before publication.

The [partial-step packet](../evidence/20261019-qualified-ranking/solid-pad-partial-step-recovery-v1/README.md)
preserves both complete controls, exact source variants, terminal receipts,
private damaged-write inputs and external array identities. New runtime outputs
stay under the established data run root. Existing fine continuations still
start from their original complete-phase endpoints and retain their captured
sources; this validation does not restart or relabel them.
