# Friction reference refinement and every-step observation admission

The500um/two-wall nonlinear reference completes its full history and physical
momentum checks. Its measured change from the coarse reference is still above
the spatial targets, so friction accuracy remains unqualified. A new every-step
cost test also exposes a DPR source-admission failure that endpoint-only stress
observations did not exercise. Neither result closes the
[active goal](QUALIFIED_STRESS_RANKING_GOAL.md). The
[packet](../evidence/20261019-qualified-ranking/friction-reference-observation-v1/README.md)
retains terminal successes, failures and exact executed sources separately.

## Completed nonlinear reference and measured changes

`solid-pad-stvk-spatial500-v2` (728256) completes400 continuous1ms steps,
with0.1N normal loading followed by sustained mu0.3 friction and horizontal
acceleration1m/s2. All complete-equation, normal/friction law, point-velocity,
omitted-boundary and current-moment history checks pass, as do both final50ms
windows. No velocity clearing or vertex pinning is used. The independent
endpoint audit passes; current-position moment residuals are9.12e-10 and
7.72e-10 against the unchanged1e-7 target. The shear material-position moment
is5.21e-6, illustrating why current and rest moments must not be conflated.
Dynamics time is5641.13s. This run uses the previously validated direct
constitutive correction without the subsequently introduced evaluation cache.

After verified parent completion, `stvk-spatial500-fields-v2` (728478)
computes complete common-column comparisons with physical reference-volume
weights, exact constitutive Cauchy evaluation, depth quadrature4 and surface
quadrature9/25. Against the coarse one-wall reference, the25-point changes
in percent are:

| Endpoint | Tangent | Membrane | Bending | Outer | Inner | Raw VM | Raw tension |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal | 8.2062 | 5.5394 | 9.6221 | 6.9978 | 7.8323 | 4.5411 | 8.3442 |
| Sustained friction | 8.2058 | 5.5390 | 9.6223 | 6.9979 | 7.8321 | 4.5451 | 8.3501 |

This changes both surface and wall resolution; it is not an isolated surface
control. The9-point tangent/bending changes are8.2098/9.6156% normal and
8.2094/9.6158% friction. The small angular-quadrature difference does not
remove the large spatial change. Finer surface and independent wall controls
are required; the2% field/5% peak targets are unchanged.

The same500um mesh also permits a separate normal comparison against the
previous linear static reference at0.1N. The maximum complete-field change
is0.025452% (bending), raw VM change0.021518% and raw tension change0.042115%.
This quantifies this one normal load's transient/constitutive contrast; it is
not an unmeasured bound for the other normal loads or for friction.

The250um/two-wall cached trajectory728389 remains in progress. The additional
250um/three-wall trajectory728464 holds the surface resolution fixed and
isolates wall resolution. Both retain all original constitutive/contact and
physical-equation gates. A separate optional response-reuse control728491
tests whether the already computed full compliance response can avoid a
redundant linear solve; its24 manufactured controls pass, but complete-history
and field equivalence remain pending and it is not used to qualify these runs.

## DPR every-step rejection is retained

`pad-primary-friction-cost-v1` (728468) requests B1 full stress at every step
for five0.1N levels. The initial normal level and the level containing the
tangential ramp are warmup; the final three levels are measured. The existing
normal-cost audit still passes for all three methods before the new trajectories
start. Every new cost run must supply its own complete history and field check.

DPR stops at observation37 during initial contact. The original journal
serialization refuses a nonfinite peak after36 written steps. The diagnostic
repeat728476 captures the complete rejected field/source packet and writes
step37 with `native_valid=false`, errno128 and a missing JSON peak, then
raises the existing rejection. It does not replace the nonfinite value by zero,
continue the trajectory or mark the result accepted. Raw nonfinite values
remain in the retained NPZ.

The read-only source inspection728487 reproduces eight status1 input
rejections. Each source has zero current normal force but retains a tangential
force of4.33e-22 or5.17e-22N from the declared lagged-normal friction law.
The unsigned total source force is0.00116359N. Each local Coulomb cone has
zero tangential capacity, and the existing per-source16-epsilon allowances
are only1.54e-36–1.84e-36N. The source inputs therefore fail that contract.
Scattered forces, the elastic right-hand side and equation residual are all
finite; the nonfinite output accompanies rejection and does not demonstrate
an elastic factorization breakdown.

No input forces are dropped, no friction cone or native tolerance is relaxed,
and the physical law is unchanged. The observer now retains a rejected packet
even between requested endpoint files, keeps JSON valid with explicit missing
nonfinite diagnostics, and still stops on rejection. Its complete-force and
residual finiteness checks are strengthened. The failure has **no admitted
every-step DPR friction timing or error rank**. Prior endpoint-only friction
fields cannot admit this different observation cadence or its failed cost run.
The completed normal-cost matrix is a separate, friction-free configuration.

The SCR every-step trajectory is still running at publication; no completed
cost or failure conclusion is assigned to it. Its active writer is excluded
from the selected packet, except for the already completed normal-compatibility
audit leaf. The original failing DPR group and successful Shell group are
included with their actual exit codes.

## Independently admitted Shell friction cost

The repaired250um Shell completes1000 steps and all five mechanical windows.
Its own primary audit passes whole-history equations, solver exits, pad/friction
laws, sustained tangential balance, observation cadence, disabled phase timers,
interval accounting and endpoint field/peak agreement. Measured blocks begin
after the tangential ramp and two full warmup levels.

The three warm batch-step latencies are summarized as median301.428ms,
range293.071–305.976ms. These are B1 complete research-workflow times,
including mechanics, requested full stress, host transfers, mechanical checks
and journals; they are not isolated kernel latency or independent-process
statistics. They do not admit the failed DPR run or the unfinished SCR run.

Against the separately admitted0.4s friction frame, the later endpoints show
real continuing creep: complete-field differences reach0.107431% and raw VM
changes0.0420705% by1s. They remain below the existing0.5%/1% conformance gates;
the result is not bitwise steady state or motion artificially reset to zero.

## Reproduction and storage

The packet contains205 size/SHA256-verified payloads totaling2,289,881 bytes.
Large reference fields and the2,432,914-byte rejected source/field NPZ remain
at the manifest's data-side paths. Selected leaves from the active primary-cost
array are identified explicitly; no live writer, environment or compatibility
link is moved.

Use captured `tasks.json` and analysis-source files through the established
rtx-mid / GENESIS_IMAGE_VER=1_26 workflow. `inspect_pad_rejection --frame
REJECTED_NPZ --output NEW_DATA_PATH` reproduces the source inspection.
`audit_pad_primary_cost` now distinguishes normal from sustained-friction
baselines and requires two warmup levels before friction measurements.
Its normal compatibility and the successful Shell friction audit are retained.
Fine spatial qualification and the final case-specific reliability/error/cost
comparison remain in progress.
