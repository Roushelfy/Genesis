# Three-wall250um friction reference, 2026-10-19

Job728464 completed all400 StVK steps and both requested mechanical windows.
The independent current-configuration momentum audit passes. A separate check
of every stored step confirms the original six mechanical gates without
relaxation. Runtime is11419.849s for the recorded dynamics scope; this is
reference cost, not native/Shell production throughput.

The largest complete-history equation, normal-law, friction-law and current
moment relative defects are2.3183e-9,5.5684e-9,1.1634e-8 and2.6727e-9.
Point-velocity prediction error is at most2.2372e-13m/s; the minimum omitted
boundary gap is9.5801e-5m. Final tangential support is approximately-0.00748085224N
against+0.00748085213N body forcing. Full history and frame hashes are retained.

## Measured thickness changes

Jobs728950_0/1 compare the admitted250um two-wall and three-wall fields with
complete physical-volume weighting, q9/q25 angular sampling and depth4.
The primary q25 changes below are percent, normalized by the finer field.

| Endpoint | Tangent | Membrane | Bending | Outer | Inner | Raw Mises | Raw tensile | Common Mises | Common tensile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal | 1.310360 | 0.303391 | 0.426220 | 0.246777 | 0.419646 | 0.202772 | 0.032886 | 0.023481 | 0.039123 |
| Sustained friction | 1.310291 | 0.303361 | 0.426229 | 0.246787 | 0.419633 | 0.202403 | 0.041779 | 0.024211 | 0.031021 |

This wall contrast passes the2% field/5% peak targets. It does not resolve the
previous500-to250um surface change of2.749%; the125um surface/wall/joint
comparisons remain required. No final friction rank is assigned here.

## Fine controls and outstanding comparisons

Job728930 runs two exact125um three-wall controls: q6 at1ms for400 steps and
q10 at0.5ms for800 steps. Both cover the full0.4s with the same material,
loads, friction law, geometry and mechanical tolerances as the q10/1ms final
reference. Their independent momentum audits follow each history. The worker
changes only allow CPU-only runtime identity collection; a normalized AST
comparison verifies that scientific code is unchanged.

The prepared12 comparison tasks cover both endpoints, both controls,
q9/depth2, q9/depth4 and q25/depth4, normalized by the final q10/1ms reference.
Submit only after complete400/400/800-step histories, windows, momentum and
frame identities pass. Targets remain0.5% fields and1% peaks. Existing coarse
controls are not treated as measured fine-grid uncertainty.

Job728945 separately attempted250um Shell replay/solved-pad controls at0.1,
0.3 and1N. The0.1/1N replay tasks fail the static zero-resultant assertion
(with8.176e-12N and2.225e-12N residuals); their downstream solved-pad tasks did
not execute. The0.3N pair completes but has not yet passed a consolidated
comparison. These are unresolved controls; no acceptance threshold is relaxed.

The [packet](../evidence/20261019-qualified-ranking/friction-wall250-reference-v1/README.md)
retains terminal250um history/momentum/field receipts and immutable fine-control
plans. Full arrays and all live fine-reference outputs remain on data. The
[completion checklist](QUALIFIED_STRESS_COMPLETION_CHECKLIST.md) remains open.
