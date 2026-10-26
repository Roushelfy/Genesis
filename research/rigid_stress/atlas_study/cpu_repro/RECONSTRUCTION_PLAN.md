# CPU atlas reconstruction protocol

This reconstructs the unavailable cpu_next experiment from its retained report
and the published shell_cpu finite-element core. Source identity, undocumented
fixture seeds and exact old timings cannot be recovered from the report.
The old values are historical comparison targets, not inputs to parameter fitting.

## Fixed definitions

- Frozen bilinear egg: 54 coarse Q4 macros, each subdivided 4 by 4, 864 fine
  Q4 elements, 866 fine nodes, 5,196 total shell DOFs and 56 coarse nodes.
- Authored axes 22/22/30 mm, asymmetry -0.18, thickness 0.5 mm, E=10 GPa,
  nu=0.3 and density 1000 kg/m^3. Smooth directors are sampled and frozen.
- Shared coarse interpolation P plus local coordinate bubbles B_i. Each
  selected feature uses T_i=[P,B_i], K_i=T_i.T K_m T_i and the same mother
  load relief. Six common coarse translation gauge rows remove rigid modes.
- Local enrichment covers conservative macro caps plus one adjacency halo.
  Fine nodes qualify only when every incident macro is active. Coarse nodes
  are excluded from bubbles so coordinates are independent.
- The same output S gives all 6,912 two-sided stress samples. Full output
  reconstructs all body-frame tensors. Max covers all samples.
- Fixed material-sphere cap angle is nominal radius / 22 mm, with a normalized
  Hertz-shaped radial rule of 4*q*q points. This is the declared CPU load
  adapter, separate from native DPR and from independent contact physics.

## Main comparison

Freeze a new explicit 42-loaded-case and release bank before evaluating fields.
It includes 24 whole-surface directions, six seams/poles/corners, two double
contacts, one triple contact and nine moving frames. Keep normal and frictional
loads, finite footprints and source-wise assignment without snapping loads to
library centres. New fixture details are recorded rather than tuned to match
the old 4.02/4.14/4.08 percent means.

Compare a 12-feature icosahedral library with continuous weights, nearest-one
and a narrow continuous transition. Merge equal (feature, environment) RHSs.
Retain hard top-two and whole-load averaging as explicitly labeled controls.
Use whole-field area-weighted tensor differences and sampled maxima separately.
Test pair, second/third and triple selector boundaries with decreasing offsets.

## Other quoted claims

Rebuild the full exact top-node force response H and FP32 storage, both exact
Schur directions, global POD and actual batched history/failure compaction.
Retain all positive and negative outcomes. No large H matrix is committed.
The full H payload target is 20736*2598 scalars, or 430,977,024 FP64 bytes and
215,488,512 FP32 bytes, conditional on the declared top-force basis.

## Timing protocol

Numerical experiments can be developed independently. Formal timings run in
exclusive single-process windows after all other workers stop numerical work.
Use one BLAS/OMP thread. Record wall and process CPU time, raw repetitions,
source hashes and cold setup separately. No result-oracle cache is timed as a
live source/recovery path. Checksums and exact-reference evaluation stay outside
the timed region. Include all required current loads, gates and history updates.

Use fixed B1/B16/B64 static cases with 15 loaded and one release case, and
separate sustained B16/B64 history trajectories. Report batching/tiling honestly.
CPU seconds are measured anew, not converted to GPU or Franka RL performance.

## Numerical interpretation

Check numerical solve residual inside each selected Galerkin space. The mother
residual of a spatial approximation is a separate diagnostic. Do not always
correct to the full mother merely to pass the exact mother's residual gate.
Sum-of-source feature recovery is not automatically the joint Galerkin solve.
The combined tensor/peak errors must be measured against matching mother loads.

Publish reconstructed source, frozen cases, all summary values, failures,
provenance and one-command reproduction. Clearly separate reproduced numerical
mechanisms, newly measured quantities and unrecovered historical details.

## Geometry interpretation correction before formal timing

The initial pilot borrowed the older egg_supplement convention in which the
authored axes describe the outer surface and the middle surface is shifted
inward by h/2. The retained cpu_next report explicitly describes the authored
frozen bilinear geometry as the midsurface. The final reconstruction therefore
uses authored coarse egg positions directly as the midsurface, with top/bottom
loads offset by +/-h/2. Geometry.position_convention records this choice.

This decision follows the written model specification, rather than selecting
the geometry that matches historical errors. The previous offset-model pilot
outputs/logs are retained under initial_outer_offset or equivalent explicit
diagnostic paths. Final numerical controls are rerun after this correction.
No formal performance window had started when the convention was corrected.
