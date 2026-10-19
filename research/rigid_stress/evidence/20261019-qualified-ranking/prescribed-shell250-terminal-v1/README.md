# Admitted250um Shell prescribed-traction replay

This companion updates the replay's status after the earlier
[reference/control packet](../prescribed-reference-controls-v1/README.md),
which retained its submitted inputs only. Job728888 completes in4m52s and
passes the static equations in3 iterations, with no solver flags and final
force residual2.10101472e-8. The original OOM remains in that earlier packet.

`manifest.json` verifies25 payloads totaling455,148 bytes: actual start/result,
all3 iteration records, source, terminal logs and audited frame/rest identities.
The full arrays remain on data. The new three-field Shell comparison (728918)
contributes immutable submitted inputs only; its results remain pending.

This is velocity-cleared, fixed-traction static elasticity on the primary250um
mesh with12 coarse patches. It is not a continuous contact history. The first
Shell/control tensor difference combines thin-shell formulation/kinematics and
force-transfer effects; it does not isolate contact error. See the
[report](../../../docs/PRESCRIBED_REFERENCE_QUALIFICATION_20261019.md).
