# Automatic DPR evidence

See [the implementation report](../../docs/AUTOMATIC_DPR_IMPLEMENTATION_20261023.md)
for the API, command, measured scope and limitations. Production engine
`40775c1d0` is the final measured version; later commits only add audits/docs.

- `example-tasks.json` and the unchanged 3,754-byte `conditions.npz` reproduce
  the B4 Hertz/full 130-frame benchmark using the report's submit command.
- `timing.json` retains all three repeats, configuration, GPU/source identity
  and original-result hashes for the before/after scene and module measurements.
  `profiles.json` contains separately measured stages at frame 129.
- `results/automatic-dpr-controls-v11` contains final CPU/GPU numerical controls,
  strict edge rejection and explicit planar edge/vertex approximation controls.
- `results/automatic-dpr-controls-v9` contains the independent finite-box
  comparison. Its fine direct reference is not continuum-converged; coarse FEM
  differences of about 86% remain visible.
- `results/automatic-dpr-controls-v8` retains the six selected original DPR
  regressions per backend, from the earlier small-support implementation.
- `results/automatic-dpr-scene-planar-v2` contains 130-frame admission for both
  automatic profiles; `results/automatic-dpr-lifecycle-planar-v1` checks partial
  reset and unfused execution before the empty-integration-work optimization.
- `results/automatic-dpr-scene-comparison-v3` retains the passing same-input
  audit and the **failed exact before/after radius comparison**, including the
  full difference report and original traceback. No assertion was relaxed.
- `results/automatic-dpr-fit-replay-v1` and `results/automatic-dpr-scene-q16-v5`
  preserve the quadrature failure/replay and correctly timed sharp-edge witness.
- `campaigns.json` hashes original artifacts from all 39 campaigns, including
  compiler, setup and numerical failures. `campaigns/` retains submitted task
  arguments and job receipts. These historical task files keep their original
  absolute input paths; use `example-tasks.json` for the portable entrypoint.
- `preservation.json` records frozen sources and untouched user-file hashes.
  `manifest.json` hashes the packet files; large arrays and full benchmark
  buffer inventories remain at the data paths recorded in the receipts.

Final timing uses the original level-1 mesh (480 tetrahedra, 2,430 DOFs), fp64,
quadrature 16, inverse FEM, four conditions tiled for B64 and an RTX PRO 6000
Blackwell Server Edition GPU. Full fields stay on device; production timing
excludes research exports. All 20 final configurations have zero invalid steps.
The live window ends at initial gripping, not completion of the 600-frame cycle.

The collision boxes remain sharp. `box_edge_policy="planar"` is a declared
footprint surrogate, not a physical sharp-edge Hertz solution. The default
strict policy still rejects unsupported sharp primitive curvature. Numerical
admission and the same-coarse-mesh comparison do not certify shell stress or
surface maxima.
