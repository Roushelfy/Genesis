# Friction keyframe cost, source feasibility and reference response reuse

The existing200-step full-field observation mode now has matched, independently
admitted friction costs for DPR, SCR and the repaired250um Shell. This does not
admit native every-step observation: DPR fails at observation37 and SCR at38.
Optional reference response reuse passes analytic, complete-history and field
equivalence controls. Fine nonlinear reference qualification and the final
friction comparison remain in progress under the
[active goal](QUALIFIED_STRESS_RANKING_GOAL.md).
Exact terminal receipts and sources are in the
[checkpoint](../evidence/20261019-qualified-ranking/friction-keyframe-reuse-v1/README.md).

## Matched keyframe costs

`pad-friction-keyframe-cost-v1` (728543) completes1000 continuous1ms steps per
method, with five0.1N levels, mu0.3 and the declared lagged-normal smooth
friction law. The first normal level and second tangential-ramp level are
warmup. The last three200-step intervals measure the complete research workflow,
with full-field observation every200 steps. Every physical step retains the
mechanical checks and journal. Each method passes its own complete-history,
equilibrium-window, requested-observation and primary-cost audit; independent
normal compatibility also passes. No per-phase timer is enabled.

| Method, B1 full output | Median ms/step | Minimum | Maximum |
| --- | ---: | ---: | ---: |
| DPR,2mm | 75.925 | 75.109 | 76.163 |
| SCR,2mm | 88.373 | 86.576 | 90.789 |
| Repaired250um Shell | 295.081 | 290.305 | 296.622 |

Hardware is RTX PRO6000 Blackwell Server Edition in the established rtx-mid
image. These are averages over whole intervals including mechanics, host output,
admission calculations and I/O, not single stress-call or pure-kernel latency.
Consecutive warm blocks are not independent fresh-process replicates. Keep this
friction-mode table separate from the
[normal full/max, B1/B4, every-step/sparse matrix](COMPLETE_PAD_COST_20261019.md).

At1s, field drift from the admitted0.4s friction frame reaches0.070997% DPR,
0.070721% SCR and0.107431% Shell; raw VM changes are0.003890%,0.003948% and
0.042070%, respectively. All remain below the existing0.5% field/1% raw-peak
conformance gates. The free egg continues creeping, without velocity clearing.
Complete mechanical history and valid requested keyframes do not establish
valid stress at unobserved steps.

## Every-step native failures and actual inputs

The original `pad-primary-friction-cost-v1` (728468) SCR group terminates
with failure after37 written steps, at observation38. Its nonfinite rejected
peak causes the original strict JSON serialization to stop, as in the original
DPR failure. `completed=false`; no every-step SCR cost or error rank is admitted.
The previous [checkpoint](FRICTION_REFERENCE_OBSERVATION_20261019.md) correctly
recorded SCR as running at its publication cutoff; its packet is not rewritten.

`capture_native_pad_input` preserves the actual call sequence and warm state,
saving the caller-owned input immediately before observation38. This intrusive
diagnostic is not a primary timing. The immutable ready receipt permits
read-only inspection while the costly recovery finishes. All46 captured nonzero
sources have zero normal force. Their unsigned total tangential force is
1.76411488e-20N; individual cone excesses reach4.33397e-22N. These tiny loads
are floating-point-scale residues of the lagged-normal law at initial detachment,
not significant sustained shear. They nevertheless violate the instantaneous
unilateral Coulomb input contract. Later requested equilibrium fields have
actual sustained tangential reaction about0.00748N.

The independent source-cone inspection (728575) uses actual P1 outer triangles,
2mm radius and inward face normals. It includes a conservative superset of
every face that could carry the source forces. On228 selected SCR faces,
every admissible traction obeys `f_z <= -a p`, with
`a = min(-n_z - mu sqrt(1-n_z^2)) = 0.7001040441`.
Since `|f| <= sqrt(1+mu^2) p`, a necessary aggregate condition is
`-F_z >= a/sqrt(1+mu^2) * |F_xy|`. The captured pure-tangential resultant
violates it by1.18297673e-20N, versus a32-epsilon accumulation allowance of
1.25347902e-34N. Even the conservative support cannot reproduce this exact force
with unilateral surface cones. This certificate does not identify the native
solver's internal status. The DPR step37 packet has substantial normal force
and does not fail this aggregate condition; its eight observed local source
rejections are separate. Failure to prove infeasibility is not proof of feasibility.

The capture subsequently terminates after38 journal rows (728529), preserving
the2,808,291-byte rejected NPZ. Its independent final inspection (728586)
observes46 status34 values: `30 + SAMPLED_WRENCH_INFEASIBLE`. It reproduces
the same force-cone certificate from the final rejected packet. Stress is
nonfinite; scattered force, right-hand side and equation residual arrays are
finite, but the reported relative elastic residual is1.034e17. Finiteness alone
does not admit this field. The capture is a failed diagnostic, not a successful
elastic solve or primary cost. Both exact packets and their distinct hashes
are retained; the small before-call packet is published directly.

No source is removed, cone enlarged, threshold retuned, or failed observation
replaced by a passing keyframe. The independently admitted every-step Shell
friction time remains301.428ms median. Native failures describe this particular
lagged-law/input-contract combination, not all frictional native workloads.

## Optional reference response reuse

`plate_solid_pad --contact-solver modal --reuse-contact-velocity` reuses
`A^-1 J^T impulse`, already computed for the full contact-compliance check.
Adding it to the free velocity avoids a second solve for the same body response.
The flag defaults off and requires the modal solver. It discards no elastic
mode and changes no physical equation. Independent full-Piola force and
current-position momentum checks remain mandatory.

`solid-pad-stvk-response-reuse-control-v1` (728491) passes24 manufactured contact
controls, including redundant rows and direct full-velocity checks, plus StVK
analytic controls. All400 steps, both final windows and the independent momentum
audit pass. Against the admitted cached reference, maximum relative changes over
normal/friction endpoints are:

| Quantity | Maximum relative change |
| --- | ---: |
| Complete tangent/membrane/bending/inner/outer fields | 5.118e-10 |
| Separate full3D-only normal/shear components | 1.224e-9 |
| Raw VM/tensile peaks | 1.711e-10 |
| Common tangent VM/tensile peaks | 1.093e-10 |

The retained metadata-only audit verifies predeclared1e-8 field/raw-peak targets,
also applying them to common peaks and full3D-only components. Dynamics time is
635.432s versus913.413s for the prior cached execution; single runs do not
establish a statistically repeated speedup.

After these controls passed,125um/two- and three-wall histories (728537) were
submitted with optional reuse enabled. Existing250um/two-wall (728389) and
250um/three-wall (728464) histories keep their captured sources. All retain400
steps, q10 contact, direct StVK correction, exact-state caching and unchanged
physical/convergence gates. Complete histories and independent surface/wall/peak
comparisons are required before reference qualification.

Both125um allocations were subsequently **preempted** onrtx-209-187 after
about16 minutes. Automatic requeue hit the existing-output guard and failed;
Slurm's duplicate accounting records preserve both events. Only16 steps exist
for the two-wall run, none for the three-wall run, and neither has a final
result or momentum audit. Legacy shell returncode files contain zero because
they did not explicitly handle termination; those files do not establish
completion. The [interruption packet](../evidence/20261019-qualified-ranking/reference-preemption-v1/README.md)
retains59 verified payloads totaling473,402 bytes, including the partial journals,
scheduler records and source receipts.

The submission helper now records TERM/INT/HUP as143/130/129 and passes
`--no-requeue`, consistent with its fresh-output-only protocol. Seven stdlib/shell
controls pass, including actual foreground-process-group TERM and preservation
of an existing destination. The initial control's parent-readiness race is
retained separately; the final test waits for the foreground child's readiness.
Fresh retry728589 uses exactly the same scientific tasks at new data paths;
both allocations explicitly show `Requeue=0`. This is an infrastructure retry,
not a change to the physical model, tolerances, resource allocation or admission.
Its complete scientific results remain pending. The250um jobs are unaffected.

## Reproduction and storage

Reuse captured `tasks.json` through Slurm, with fresh data-side outputs and the
existing environments. `inspect_pad_rejection --frame INPUT_OR_REJECTED_NPZ
--mesh NATIVE_MESH --radius 0.002 --output NEW_DATA_PATH` reproduces the read-only
certificate. Unavailable native statuses/field-finiteness values remain null
for before-call inputs.

The packet keeps329 size/SHA256-verified payloads totaling6,028,562 bytes:
bounded terminal receipts, executed sources, the small captured
input, predeclared targets and reproducible metadata gate. Large fields/journals,
active references and dependencies stay on data. No active writer, environment
or compatibility link is migrated. Remaining friction refinement, error ranks
and actual traction/response figures keep the overall goal open.
