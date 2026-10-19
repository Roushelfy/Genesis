# Pad refinement, linear audit and friction controls

This is a partial evidence checkpoint, not a qualified stress ranking.
[manifest.json](manifest.json) records exact original paths, byte counts and
SHA256 hashes of unchanged selected receipts. Runtime fields and full logs
remain under the established data-side `20261019-qualified-ranking` run root.

The fixed-point500/250um P2 references pass their discrete equations and
quadrature controls, but their complete-field spatial difference exceeds2%.
The500um continuous Shell cycle passes all five windows;250um fails its PCG
cap. The corrected step40 diagnostic and independent joint sparse solve
establish a real unresolved linear error, with the old frozen-system control
retained. These facts do not qualify the failed trajectory.

Physical-pad friction pin `4ce5542f9` keeps smoothing speed independent of dt.
Its14 analytical/contact regressions pass in `compliant-pad-friction-controls-v2`;
the two old-engine negative controls fail as intended. Version1 retains the
earlier four unloaded-force assertion failures. No assertion was loosened:
only the analytic shear test's velocity tolerance was tightened. Continuous
egg friction histories and their reference fields remain separate obligations.

Reproduction uses each campaign's `tasks.json` and `submission.json`, with
the source identity and patches listed by `source.json`. Apply
[`shell_pad_friction.patch`](../../../shell_comparison/patches/shell_pad_friction.patch)
to `41b2a57be` for the friction variant. The separately truncated diagnostic
uses [`shell_pad_linear_capture.patch`](../../../shell_comparison/patches/shell_pad_linear_capture.patch)
on that same normal-pad base; it must never be used as a continuous solver.
Use the retained diagnostic SHA check in `plate_shell_coupled`, then
`audit_shell_linear` on its frozen NPZ. New run directories must be fresh and
under data; preserve the original unsuccessful attempts.
