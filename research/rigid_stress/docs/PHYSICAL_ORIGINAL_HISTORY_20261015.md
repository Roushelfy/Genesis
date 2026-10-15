# Original history independently checked through2.23s

This advances the complete [physical-validation objective](LOAD_MODEL_PHYSICAL_VALIDATION.md).
The original ground-supported approach/lift protocol now has1784 independently
checked physical states through2.23s. It still lacks856 of the requested2640
states through3.3s. These are the original controls and physical initial state,
not the stationary fixed-arm control used in the spatial study.

## Original1783-state prefix

715815_0 preserves every869 previously audited state and advances to1783 before
failing on1784.715817_0 independently checks all1783 controls, previous-state
integration, unaveraged contact pullback, real ground support, consistent body
inertia and the actual nine-joint robot step. Maximum body and robot-effort
residuals are9.99503e-11N and9.91075e-11 in the declared effort units, below
their original1e-9 budgets. The producer is FAILED, not a completed3.3s result.

715856_0 independently audits raw global stress/whole-volume determinant bounds
on all1783 states. The maximum raw stress lower bound over the history is
1.068730697MPa at step866,t1.0825s. The last retained lower bound is
0.156448414MPa. Global field bounds remain distinct from physical accuracy or
continuum error estimates. Near-zero precontact fields retain their absolute
rounding differences rather than being interpreted through relative error.

715878_0 completes12 actual native CUDA replays: full-inertia and rigid-relief
loads at steps11,80,866,869,1440,1783. All original complete elastic budgets
pass; the maximum independently checked budget fraction is0.0342594. Complete
corner tensors agree with independently assembled linear FEM to1.34970e-10
relative L2 difference. These are numerical nodal-load checks at one unresolved
coarse physical setting. They do not validate finite-footprint mapping or a
converged full-application stress field.

715879_0 checks7518240 independently evaluated interior/corner samples,69 per
tetrahedron, over227 saved snapshots. Its upper stress and lower determinant
certificates retain their explicitly reported arithmetic allowances. This is
not an interior sampling audit of every1783 state or a continuum-error bound.

## The rejected direct candidate already satisfies the original physical gates

The saved fraction-one direct Newton candidate for1784 has body residual
1.07373011377e-10N and actual/fresh robot-step effort residual1.56526e-11.
The strict internal Newton target is1e-10, whereas the original physical
body/actual-robot/fresh-step acceptance gates are1e-9. The distinction already
exists in the original solver and independent auditor; no gate is enlarged.

After rejecting that direct candidate, the original fallback reaches only
artificial residual fraction0.288151 and fails. That artificial stage is never
used as a physical state.716034_0 and716039_0 instead rebuild the original
equation and re-evaluate the original saved direct candidate without taking any
Newton or continuation step. Both retain its exact full displacement. The
second report adds the existing runner's recovery-record schema, explicitly
identifying the operation as a re-evaluation rather than an artificial stage.

716040_0 independently checks all1783 preceding states and the recovered1784th
state using separately reconstructed original controls, contact forces, body
inertia and actual MuJoCo robot stepping. The recovered body's residual is
1.07373011376e-10N; contact pullback error is zero. Original dt1.25ms, physical
history, geometry/material/mass, friction, controller gains and caps are fixed.
Its source macro-state differences are diagnostics: source contacts and macro
states have different integration timing, and Genesis/IPC contact-law
equivalence is still not established.

## Continuing the original equation

`budget_checked_original.py` keeps the strict Newton target. If the direct
solve rejects a candidate, it independently re-evaluates the unchanged original
body/joint equations and requires the original1e-9 physical gates plus a fresh
robot step before accepting. Otherwise it retains residual-defect globalization
and permits only its final unmodified equation as a physical state.716041_0
starts from the exact1784-state audited prefix.

At1785 its body norm repeats exactly above the strict internal target despite
already satisfying the original physical gates. The separately snapshotted
`stagnation_checked_original.py` checks this condition after three consecutive
identical body norms. It can return only after the original body/joint/fresh-step
gates pass. It changes neither the original residual nor a tolerance. The strict
Newton and residual-globalization paths remain available.716061_0 runs this
numerical continuation from the same immutable audited1784-state prefix.
716065_0 qualifies an immutable read-only observation of both live producers:
all1784 original prefix files are byte-identical, and complete arrays plus all
corner/quadrature tensors on1785–1787 meet the previously declared numerical
comparison budgets. Maximum corner/quadrature relative changes are
7.01976e-12/5.92974e-12, and raw peak change is6.43264e-13. This does not claim
equivalence of the full future histories. After that result, the redundant
716041_0 baseline is cancelled; its explicit partial-state audit is retained.
716042_0 independently verifies all1789 saved baseline states. Its actual wrapper
code is zero and its summary is absent; the corrected classifier and all-state
audit correctly retain `complete_requested_history=false`. A zero code therefore
does not mislabel this real cancelled prefix as complete. Its source before/after
identities are closed; its original force and robot budgets all pass.
The remaining716061_0 continuation is live and unverified beyond the stated
observation. Subsequent716066/716067/716068/716069 independent all-state audits,
raw-field bounds, native replay and interior checks remain necessary. No physical
trajectory state is accepted from an artificial defect-continuation fraction.

The [complete fine spatial checkpoint](PHYSICAL_FULL_SPATIAL_20261015.md)
independently retains23.94% tensor spatial change and5.17% linear/nonlinear
tensor discrepancy at the finest held-control setting. The original-history
progress here does not resolve those physical axes. Full original history,
contact/volume/geometry convergence, nonlinear U, remaining load models and
feedback comparisons, and warmed cost at verified accuracy remain required.

Exact successful/failed job commands, input/source before/after identities,
package/hardware records and every saved raw state are preserved in the
`original-history-checkpoint-v1` publication. The full-spatial checkpoint
preserves its separate numerical/model comparisons and all failed/cancelled
replay attempts. Live producers are excluded from both publications.
`original-stagnation-checkpoint-v1` separately preserves the sealed observation,
completed field-equivalence qualification, actual cancelled baseline and its
independent1789-state audit. The continuing716061 producer is excluded.
