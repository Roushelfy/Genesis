# Closed boundary, aggregate-model and sparse-packing trials

This records closed, unselected experiments after `cd76762e`. Production
model selection and the newer boundary-fallback qualification are separate.
All final wrench, complete equilibrium/gauge and stress budgets are unchanged.

## Boundary witnesses on the original native samples

The on-demand FP64 diagnostic also works with unmodified native Q10/s8
surface-fan integration. For the original policy876/environment24 input,
CPU/GPU preserve component0 success and prove component1 continuously
incompatible without changing its point, force, moment, physical radius or
actual friction. Both searches use101 evaluations and116 proof nodes.
The CPU/GPU rays differ; their respective conservative continuous polar
upper bounds are−1.557415e-4/−4.660338e-4, with target margins
1.638305e-3/1.360779e-3. An independent all-selected-triangle/disk boundary
and arc-Lipschitz CPU audit confirms the CPU ray's negative upper bound.
Finite witnesses guide search; continuous-domain verification supplies the
proof. This diagnoses the declared componentwise law, not a general
unsupported-grasp category or permission to discard a contact.

Directly replacing the old sampled search then passes **56 CUDA cases
in655.89 s**, but the CPU epoch finishes **55 passed,1 failed in468.32 s**.
The old step602 fixture still receives a valid physical certificate; its
new polar upper bound−3.179774e-5 is weaker than the fixture's existing
−1e-4 assertion. The replacement is not selected. The follow-up retains
the original search/proof first and adds on-demand boundary search only
for unresolved contacts, preserving already established stronger proofs.
The original fixture assertion and physical/numerical budgets stay fixed.

The attempted old/new timing baseline imports a separate frozen dataclass
and fails Quadrants' typed-workspace identity check. Its failures are
archived. No matched diagnostic speedup or overhead is claimed from that
epoch; the subsequent baseline uses the actual native workspace class.

## Explicit aggregate-pair field experiment

A separate model constrains the complete pair wrench over the union of
unchanged source footprints. About anchor0, the declared target is

$$F=\sum_iF_i,\qquad
M=\sum_i\{M_{a_i}+(a_i-a_0)\times F_i\}.$$

The experiment fits the existing weighted vector-traction objective to
that target with each destination's actual material normal, radius and μ.
Original input fixtures remain immutable. This changes the conserved
object from every original point component to the aggregate pair wrench;
it remains **unselected** and is not a silent replacement of existing
componentwise acceptance.

Both the exact B1024 step602 and policy876 inputs pass native Quadrants
fitting/scatter/full recovery on CPU and GPU. Each fixture uses three actual
friction inputs: original μ,1.1μ and1.2μ, giving **12 independent full-field
comparisons** across the two backends. The original settings are
0.4473954717 and0.5196625772, respectively. Every cone and aggregate wrench
check passes. The independent fit oracle uses a convex-dual trust-region
solve with no native forces/duals; independent CPU elasticity assembly and
full tensor/von Mises/global peak/equilibrium checks then pass. Largest
normalized wrench error is3.506e-11, cone excess2.2e-19 N and complete
residual budget fraction0.003629. Gauge rows are included.

The fit reference shares the selected native sample geometry/profile cloud.
CPU/GPU point counts differ: step602 has1100/1195 union points and policy876
1248/1402. Each backend's fit/field oracle and the cross-backend peak budget
pass, but this epoch does not establish the cause of that count difference
or an independent new footprint audit. No aggregate live/policy trajectory,
lifecycle matrix or large-batch acceptance is claimed.

Aggregate conservation alone cannot determine stresses from self-balanced
source forces. For example, opposing source forces can have zero total
force and moment while still compressing the material. The simple
`w_q F_total` prior in this experiment can return zero loads for that
case. Any more general aggregate law needs to retain original local source
load information, or otherwise declare and justify its distribution law,
and validate such cancellation cases before adoption. These two successful
fixtures establish feasibility and same-cloud numerical consistency only.

## Sparse compaction and dense tails

Parallel atomic collection followed by stable insertion sorting passes
**10 CUDA lifecycle/diagnostic cases in104.67 s**. The measured current-law
B32 live comparison retains all environments, input updates, retries and
37 resets per1280-step repetition, with three repeats after600 warmup steps.
On one RTX PRO6000 Blackwell Server Edition GPU it gives
3053.558→3136.285 effective env-step/s, **1.027092x**. Invalid environment
steps and overflow are zero. Source/condition hashes, all physical inputs
and numerical budgets match. Native buffers are167,626,674 bytes, including
the direct-boundary prototype's larger37-entry proof stack. This is an
unselected candidate's live cost, not a new production or policy result.

Independent packing cases exercise1200 raw slots in each of six
environments, counts0/1/8/32/64/1200, full density, release and restoration.
Both CPU/GPU preserve every valid index in its exact original stable order,
clear all original metadata and force slots, and retain valid masks.
The quadratic candidate's GPU full-density stage reaches25.419 ms versus
0.153 ms native, an unacceptable disclosed tail. A hybrid instead uses
the original linear scan above32 active contacts. It measures0.045 ms
for sparse GPU packing and0.161 ms for full density, with no added buffers.
CPU atomic collection is slower than its original scan; no CPU benefit is
claimed. The hybrid's complete live qualification is a separate epoch.

The hardware is RTX PRO6000 Blackwell Server Edition,97,887 MiB,
driver595.71.05, image `genesis:1_26`, Quadrants1.3.3, FP64. The full low
shell, Q10 squared-coordinate fans, s8 paths, verified disks/nine-panel
retry, block4/effective8, GJK mesh64/contact patches, componentwise
pair-union law and dt0.01/one substep stay fixed within timing comparisons.
The bank SHA256 is
`156d956bc5fc3827da8df236cf1695a4bf30a9a1e1d939c672f6a57d69d10930`.
No physical mesh/integration convergence or RL training ran.

## Evidence and remaining work

The [boundary/model-trials manifest](../evidence/20261011-nonconvex/boundary-model-trials-checkpoint-v1/manifest.json)
contains616 artifacts/11,528,098 bytes, SHA256
`47bbf4f887270b0bedb45724451b1e48c074b395567cfd053f5b3469d1f3e8a7`.
Allocated CPU publication verifies all originals, published hashes and source
closures, including failed epochs and complete packing/live counters.
Exact source, fixtures, diagnostic/oracle/benchmark launchers and explicit
data-side output/cache paths are retained. An initial publication allocation
fails in Pyxis/enroot before executing Python; a fresh allocated CPU node
publishes successfully. No GPU result is missing because of that transient
container failure.

Legal general boundary-crossing integration, an explicit robust physical
load model, complete policy acceptance, diverse1024/2048/larger four-scope
measurements and a measured performance plateau remain open. These trials
do not complete the goal.
