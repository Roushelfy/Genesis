# Qualified prescribed-C2 method comparison, 2026-10-19

The independently qualified C2 reference now supports a complete three-method
static comparison. Jobs728893/728918/728933 completed successfully. All21
aggregation checks,27 error rows,15 signed tensor decompositions and12
independent full-resultant-array checks pass;62 source/input identities were
rehashed before publication.

This prescribed0.1N-per-pole C2 pressure,2mm radius case differs from the
physical compliant-pad problem. Native DPR/SCR use the declared two-source
coarse mesh and fixed2mm reconstruction radius; Shell uses the250um surface,
12 coarse patches and admitted static relaxation. Its static velocity clearing
is not a physical history. Do not join these errors with full-centroid-source
physical-pad throughput or the default-collider production benchmark.

The [reference qualification](PRESCRIBED_REFERENCE_QUALIFICATION_20261019.md)
retains the25 spatial/load-integration gates, q9/q25 sampling controls and
measured conditional envelopes. These are observed changes, not certified
continuum bounds or statistical confidence intervals.

## Complete relative errors

All values below are percent. Field metrics cover the complete declared domain;
raw3D/native and common tangential peaks have different definitions.

| Method | Tangent | Membrane | Bending | Outer | Inner | Raw Mises | Common Mises | Raw tensile | Common tensile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DPR | 9.798884 | 7.536518 | 9.617933 | 7.156081 | 8.804194 | 7.198117 | 4.899589 | 6.738050 | 6.639394 |
| SCR | 9.779428 | 7.593035 | 9.550592 | 7.178557 | 8.730941 | 2.103309 | 5.580918 | 4.854804 | 4.761882 |
| SHELL | 30.757306 | 6.921956 | 41.239030 | 35.259353 | 37.407864 | 4.163989 | 5.638545 | 10.259018 | 10.272116 |

Conditional observed envelopes leave DPR/SCR unresolved for tangent, bending,
outer and inner fields, with both below Shell. All three membrane intervals
overlap. Raw Mises ranks SCR < Shell < DPR; common Mises puts DPR below both
others and leaves SCR/Shell unresolved. Raw/common tensile ranks SCR < DPR <
Shell. There is no definition-independent peak winner, and no radius tuning.

## Dimensional errors

| Method | Metric | Absolute RMS or peak error | Unit | Reference RMS or peak |
| --- | --- | ---: | --- | ---: |
| DPR | tangent | 1112.175154 | Pa | 11350.018 |
| DPR | membrane | 0.2921263857 | N/m | 3.876145254 |
| DPR | bending | 5.791417267e-05 | N | 0.0006021477838 |
| DPR | outer_tangent | 1226.816881 | Pa | 17143.69802 |
| DPR | inner_tangent | 1417.224509 | Pa | 16097.1519 |
| DPR | raw_mises | 23709.51914 | Pa | 329385.0105 |
| DPR | common_mises | 17815.02605 | Pa | 363602.4553 |
| DPR | raw_tension | 17895.43791 | Pa | 265587.7838 |
| DPR | common_tension | 17631.32601 | Pa | 265556.2385 |
| SCR | tangent | 1109.966809 | Pa | 11350.018 |
| SCR | membrane | 0.2943170755 | N/m | 3.876145254 |
| SCR | bending | 5.750867882e-05 | N | 0.0006021477838 |
| SCR | outer_tangent | 1230.670124 | Pa | 17143.69802 |
| SCR | inner_tangent | 1405.432835 | Pa | 16097.1519 |
| SCR | raw_mises | 6927.983304 | Pa | 329385.0105 |
| SCR | common_mises | 20292.35501 | Pa | 363602.4553 |
| SCR | raw_tension | 12893.76725 | Pa | 265587.7838 |
| SCR | common_tension | 12645.47563 | Pa | 265556.2385 |
| SHELL | tangent | 3490.959718 | Pa | 11350.018 |
| SHELL | membrane | 0.2683050529 | N/m | 3.876145254 |
| SHELL | bending | 0.0002483199065 | N | 0.0006021477838 |
| SHELL | outer_tangent | 6044.757081 | Pa | 17143.69802 |
| SHELL | inner_tangent | 6021.60066 | Pa | 16097.1519 |
| SHELL | raw_mises | 13715.55519 | Pa | 329385.0105 |
| SHELL | common_mises | 20501.8896 | Pa | 363602.4553 |
| SHELL | raw_tension | 27246.69883 | Pa | 265587.7838 |
| SHELL | common_tension | 27278.24408 | Pa | 265556.2385 |

## Direct tensor differences and signed cross terms

For each common integration point, A=S0-S1, B=S1-S2 and T=S0-S2.
The audit integrates ||T||² = ||A||² + ||B||² + 2<A,B> directly. It does not
subtract scalar error norms. Membrane and bending arrays are also independently
reloaded and recomputed over the full area; the12 checks pass and shared-control
integration differences are zero.

For native methods, A measures the reconstructed-profile versus prescribed-C2
load difference on the same coarse elastic domain; B measures that coarse C2
control versus the final125um three-wall reference. For Shell, A contains the
shell formulation/kinematics and load-transfer difference against the250um
two-wall solid control; B is that solid's refinement difference. These are
explicit control differences, not a unique causal attribution of the physical
pad's contact error. Negative cross terms represent measured cancellation.

| Method | Metric | A / reference | B / reference | T / reference | Cross correlation | 2<A,B> (unit squared) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DPR | tangent | 0.0302997242 | 0.0948718421 | 0.097988845 | -0.0551254102 | -40827.3263 |
| DPR | membrane | 0.0112804434 | 0.0725604269 | 0.0753651802 | 0.175712643 | 0.00432174457 |
| DPR | bending | 0.0397025401 | 0.093571065 | 0.0961793338 | -0.145540922 | -3.92085849e-10 |
| DPR | outer_tangent | 0.0317941506 | 0.0721578627 | 0.0715608079 | -0.239010747 | -322320.055 |
| DPR | inner_tangent | 0.0387110774 | 0.0897817149 | 0.0880419417 | -0.260091487 | -468465.24 |
| SCR | tangent | 0.0276395166 | 0.0948718421 | 0.0977942774 | -0.0383051843 | -25879.0559 |
| SCR | membrane | 0.00721118609 | 0.0725604269 | 0.0759303525 | 0.428480149 | 0.006737012 |
| SCR | bending | 0.0368680841 | 0.093571065 | 0.0955059213 | -0.143982681 | -3.60195744e-10 |
| SCR | outer_tangent | 0.0318567683 | 0.0721578627 | 0.0717855694 | -0.232399895 | -314022.175 |
| SCR | inner_tangent | 0.0330592032 | 0.0897817149 | 0.08730941 | -0.257863278 | -396641.232 |
| SHELL | tangent | 0.307506444 | 0.019032905 | 0.307573055 | -0.0274469771 | -41388.2361 |
| SHELL | membrane | 0.0706748656 | 0.0140820743 | 0.0692195558 | -0.201906587 | -0.00603826532 |
| SHELL | bending | 0.411628839 | 0.00851866244 | 0.412390302 | 0.0791228697 | 2.0119418e-10 |
| SHELL | outer_tangent | 0.352375998 | 0.00736962119 | 0.352593535 | 0.0190700872 | 29110.065 |
| SHELL | inner_tangent | 0.374172448 | 0.010166868 | 0.374078638 | -0.0228117018 | -44972.2644 |

## Evidence and reproduction

The [immutable packet](../evidence/20261019-qualified-ranking/prescribed-method-ranking-v1/README.md)
retains terminal source/configuration identities, full JSON/CSV tables, separate
raw Mises/tensile peak positions, common peak locations,3D-only stress components
and q9/q25 comparisons. Full NPZ columns remain on data. Its recorded task files
reproduce the individual field comparisons; aggregate with:

```bash
python -m research.rigid_stress.qipc_comparison.prescribed_method_comparison \
  --data-root "$RANKING_DATA" --output "$NEW_OUTPUT"
```

Run scientific work through the documented Slurm/container runner with explicit
data-side output paths. This closes the prescribed-load comparison only;
physical-pad load/contact separation, final friction qualification and the new
production throughput protocol remain open.
