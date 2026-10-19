# Validated terminal-endpoint continuation for the physical solid reference

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

## Fine-reference status and use

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
