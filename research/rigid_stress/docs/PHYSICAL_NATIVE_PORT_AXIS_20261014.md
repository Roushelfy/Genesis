# Native intrinsic-port resolution and the finer U operating range

The [physical-validation goal](LOAD_MODEL_PHYSICAL_VALIDATION.md) remains active.
Six additional numerical configurations complete at unchanged material,
physical3mm footprints, friction and complete acceptance tolerances. They add
204 qualified fields and six explicitly unsupported pressure moments. Together
with two previously published histories, the independent audit contains282
qualified fields and eight unsupported moments. All75 fixed physical probes
lie inside every compared U mesh. Resolved contact remains a separate requirement.

## Same-mesh distance resolution

The opt-in native-history runner exposes `--ports` and `--models`. It constructs
the intrinsic-distance cache when an intrinsic model first needs it. Pure
pressure completes without that cache. Pressure fields are also retained before
a later cache initialization could fail. Production algorithms and defaults
are unchanged by this research-runner change.

Every same-mesh port comparison verifies byte-identical original rigid-input
NPZ files. Independent complete equilibrium, gauge rows, raw global corner peak
and warm/fresh fields remain checked. The maximum complete residual uses23.33%
of its original budget; maximum history/fresh tensor difference is2.74e-9.

| Tetrahedra | Port subdivisions | Graph vertices | Dense FP64 distance matrix | Initialization |
|---:|---:|---:|---:|---:|
|5669|2|4946|0.182GiB|8.22s|
|5669|4|12362|1.139GiB|54.91s|
|16146|2|10102|0.760GiB|36.94s|
|16146|4|25252|4.751GiB|250.83s|
|40824|2|19278|2.769GiB|151.48s|

These are the distance array's actual shape/byte counts, excluding adjacency,
visited flags and other solver memory. The earlier8-port failures are retained
in the [preceding report](PHYSICAL_PROFILE_HISTORY_20261014.md). The new lower-port
settings extend the observed numerical range without establishing a universal
GPU capacity endpoint or replacing the missing finest-level port bound.

At5669 tetrahedra, the largest full-tensor volume-L2 change is0.9383% for2-to4
ports and0.2994% for4-to8. At16146 tetrahedra the largest2-to4 change is0.4370%.
The largest raw-peak changes across these comparisons are0.8451% and0.1819%
for2-to4 and4-to8 respectively. Pressure is independent of port resolution
up to numerical roundoff. The40824 level has only the qualified2-port field;
its own port tail is unverified. A smaller matrix is a declared numerical
setting, with its measured accuracy consequences retained.

## Combined remeshing and the remaining physical distinctions

At fixed2-port settings, opposing componentwise raw interior peaks are
92.255/92.182/89.759kPa at5669/16146/40824 tetrahedra. The final change is2.629%;
common-probe tensor, displacement and exact linear-field energy change2.673%,
0.775% and0.775%. Its raw maximum approaches the inner bend near
`(0.244,-4.998,0.154)mm`, away from the arm contact centres.

These are combined remeshing changes. Surface geometry and independent
traction integration also change: the final opposing force-point L2 change
is10.16micrometres and the original-point wrenches' fixed-origin moment change
is1.319e-9Nm, with unchanged forces. The audit records these changes explicitly.
They do not constitute a separated volume-only experiment. The largest final
2-port tensor tail across the histories is3.293%, for source-prior near-slip.

For the finest opposing problem, full-tensor volume-L2 differences from the
independently prescribed traction reference are0.000482% pressure,0.2105%
componentwise/pair-union and0.2762% source-prior. That selected U reference
traction remains locally Coulomb-inadmissible. These numbers describe elasticity
and declared load-law differences, with the reference's physical limitation.
They cannot establish a validated frictional-contact result or model superiority.
The raw peak, full field and traction-admissibility labels all remain retained.

## Warmed one-environment scope

| Tetrahedra / ports | Pressure | Componentwise | Pair-union | Source-prior |
|---|---:|---:|---:|---:|
|2339 /8, previous|6.72ms|18.25ms|18.26ms|20.32ms|
|5669 /2|10.28ms|29.07ms|29.04ms|32.24ms|
|16146 /2|18.19ms|53.51ms|53.82ms|59.20ms|
|40824 /2|40.94ms|101.70ms|102.11ms|112.60ms|

Each median uses four warmups and eight repeats on recorded RTX PRO6000
Blackwell96GB devices. Scope includes uploads, mapping/status reads, fitting,
recovery, history, global full-domain peak and synchronization. Bulk validation,
independent FEM, rigid dynamics and policy are excluded. Device UUIDs differ;
these are measured operating points rather than paired implementation speedups
or aggregate environment-transition rates.

```bash
bash research/rigid_stress/scripts/run_native_physical_sequence.sh "$RUN_DIR" \
  --mesh "$MESH" --geometry u --max-iterations 20000 --ports 2
bash research/rigid_stress/scripts/run_native_physical_sequence.sh "$PRESSURE_RUN" \
  --mesh "$MESH" --geometry u --max-iterations 20000 --models pressure
bash research/rigid_stress/scripts/summarize_native_port_axis.sh "$AUDIT_RUN" \
  --run "$BASELINE_RUN" "$PORT2_RUN" "$PORT4_RUN" "$FINER_RUN"
```

The first launch array713572 fails in shared environment loading because
`nounset` exposes an unset `SLURM_USER`, before any numerical wrapper starts.
Its original scripts/logs and manual pre-execution audit remain retained.
The corrected array713581, dependent audit713607 and publication713626 all
complete with exit0. Shared cluster configuration is unchanged.

The [immutable port-axis archive](../evidence/20261014-physical/native-port-axis-checkpoint-v1/manifest.json)
verifies1799 entries/1799 exact originals. Manifest SHA256:
`359bc24a85e32b51a6736d6a862e0b3839ce25bfd3f9d2d0c527e235f211eff0`.
Earlier baseline fields remain in their existing archives. Every new successful
field, launch failure, source closure and complete derived audit is preserved.
Credible remaining physical tails, per-island maps and converged full egg/Franka
and interior-hotspot contact/replay remain required for stage completion.
