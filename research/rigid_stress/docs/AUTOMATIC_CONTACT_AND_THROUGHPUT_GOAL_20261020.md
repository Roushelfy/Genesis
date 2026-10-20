# Next goals: physical contact regions, robust recovery and throughput

User direction, 2026-10-20: finish the existing comparison, investigate automatic
contact regions on CPU, address the distinct SCR failures, and optimize native
throughput. This extends the [active comparison goal](QUALIFIED_STRESS_RANKING_GOAL.md)
and [accuracy/throughput protocol](ACCURACY_AND_THROUGHPUT_PROTOCOL.md).
The implementation agent should choose the simplest useful approach from measured
results, rather than build another framework or expose many tuning options.

## 1. Finish the current comparison independently

Complete the running fine-reference histories, qualification, DPR/SCR/Shell
error rankings, inspected figures and configuration-matched cost/stability
summary. Reuse validated continuation and analysis infrastructure; prioritize
scientific results over expanding generic scheduling, recovery or audit tools.
Do not hot-edit submitted sources or restart valid work to adopt this goal.

Retain fixed-radius/manual configurations and original/repaired Shell variants
as named baselines. Automatic contact regions are a separate comparison and
must not delay the current results. Performance work proceeds independently
of expensive reference qualification.

## 2. Separate and address current SCR failures

Earlier balanced-load normalization and numerical-fit repairs remain accepted
within their scope. The current failures are different:

| Case | Established result | Next objective |
| --- | --- | --- |
| Diverse banks: B128 step767/env9 and B512 step645/env330 | Independent continuous-wrench certificates prove incompatibility within the declared surface regions and Coulomb cones. Actual first-contact radii are about5.47/5.62mm, mu about0.439/0.713. | Attribute geometry/normal, support-region and force/moment-transfer contributions; repair input/model inconsistencies where justified. |
| Pad every-step SCR observation38 | Zero normal load but about1.76e-20N total residual tangential load conflicts with the instantaneous unilateral input contract. DPR has a separate observation37 local-source rejection. | Reconcile friction time levels and justified roundoff treatment, then verify the actual failing every-step path. |

Reuse the [native diagnosis](NATIVE_CONTACT_CONSISTENCY_20261019.md) and
[pad input analysis](FRICTION_KEYFRAME_COST_REUSE_20261019.md).
Use compact controlled experiments, such as force-only versus full-wrench
feasibility and matched geometry/normal/support variants, to identify causes.
Continuous infeasibility is not an iteration-limit or quadrature-resolution
diagnosis. It proves a conflict between the declared model and its inputs,
not impossibility of real physical contact.

Align source generation and recovery with a consistent declared friction law.
If that changes the old law, label the variant and preserve its baseline.
Derive roundoff budgets from meaningful physical scales, with bounded effects
on force, moment and stress; do not normalize solely by the tiny residual or
globally loosen tolerances. Keep genuinely infeasible loads rejected.
A larger patch or diagnostic DPR fallback does not itself admit an SCR result.

## 3. CPU study: determine contact support and pressure from physics

Distinguish physical pressure-bearing support, reconstruction-kernel width
and numerical search region. Predict support and pressure together; inserting
a radius into the current Gaussian kernel is not a physical validation.

Compare a compact set of candidates: current fixed radius, cheap
curvature/material/load predictions, and precomputed elastic compliance with
a small contact solve when a local formula is inadequate.

For smooth isotropic nonadhesive half-space contact, use Hertz only in its
applicable regime:

```text
a = (3 Fn R* / (4 E*))^(1/3)
1/E* = (1-nu1^2)/E1 + (1-nu2^2)/E2
1/R* = 1/R1 + 1/R2                 # two convex spherical surfaces
```

For the declared independent-column pad p=k_A*[delta-g]+, a quadratic spherical
gap g(r)=r^2/(2R), and negligible link deformation, direct integration gives:

```text
Fn = pi*k_A*a^4/(4R)
a = (4 Fn R/(pi*k_A))^(1/4)
```

These are initial models, not universal shell laws. Account for two principal
curvatures and orientation, finite flat lands/edges, pad thickness/backing,
shell thickness and global compliance as needed. Nonconvex contact can have
separate regions. Do not apply the spherical formula at zero/negative relative
gap curvature or substitute numerical penalty stiffness/penetration for physical
material properties/indentation without validation.

Group samples into physical contact regions before predicting force-controlled
size; collision points are not independent physical patches. Splitting one
region into more sources must not arbitrarily change its support. For already
distributed pressure samples, test consistent conservative transfer instead
of spreading each sample over a new finite disk. Quantify possible double
spreading in the current all-centroid path.

For a compliance-based model, reuse fixed elastic/pad operators, retain necessary
coupling between regions and consistent free-link balance/inertia, and solve
nonnegative pressure with separation/compression consistency. Do not pin the
free shell to invert its stiffness. Expand search regions when needed without
imposing them as the physical support. Reuse previous pressure/active sets as
initial guesses during smooth contact motion.

First compare area/axes/connectivity, pressure shape/centroid, force/moment and
force-indentation response; then compare complete stress and peaks. Reuse
independent references and a small discriminating set of convex/nonconvex,
asymmetric, changing-position/load, thickness/stiffness and friction/unloading
cases. Separate source sampling, load-model and elastic-discretization errors.
Do not fit radii to evaluation stress; retain useful negative results and limits.
Normal support does not uniquely determine tangential stress: include the
friction/slip history or coupling required by the chosen physical model.

## 4. Native integration and measured throughput optimization

Prioritize DPR for the first useful integration given its current measured
speed and broader passing condition banks. Continue SCR optimization on feasible
inputs and preserve its diverse-condition failures. Production remains a feature
of the existing Genesis rigid solver, entirely in Quadrants. Only rest
geometry/material is fixed: changing convex/nonconvex and asymmetric contacts,
normal/friction loads, max/full output, batching, isolation and resets remain
required. CPU prototypes stay validation tools.

Integrate the useful CPU-validated model with a small public configuration:
automatic physical-region selection plus the explicit manual baseline and
physically meaningful material/pad inputs. Keep missing data or unsupported
regimes visible. Exact API and algorithm choices are left to the agent.

Precompute curvature/compliance data and interpolate at current contacts.
Exploit temporal coherence with current-frame acceptance and correct invalidation.
A footprint below mesh resolution needs suitable integration/resolution or an
unresolved result, not a larger artificial physical radius.

Profile source grouping/geometry, region prediction/contact solve, traction
fitting, scatter, elastic recovery, max/full output, history work, dispatch,
copies and synchronization. Follow measured bottlenecks, including the known
SCR fitting/scatter/readback costs. Consider batching, changed-environment
compaction, cheaper history updates, fusion and lower precision where matched
numerical checks and measured benefit justify them.

Use low-resolution moving Franka/egg and diverse contact histories. Measure
paired pure-rigid/max/full modes, useful batch sizes, cadence, preprocessing
and memory, including all new prediction work. Keep module replay and actual
scene throughput separate. Do not drop hard environments or time only passing
prefixes. Sparse updates do not establish every-step peak reliability.
Pure scheduling/copy optimizations need focused consistency checks, not another
fine-reference sweep; physical load-model changes need appropriate small-case
physical validation.

## Delivery

Publish the fixed-radius comparison, the automatic-versus-manual CPU study,
the useful native implementation and measured tradeoffs. Provide contact and
stress maps, errors/uncertainty, stability/failure classification, separate
module/scene throughput and practical configuration recommendations. Join
accuracy and speed only for matching actual configurations.

Submitted jobs, a prototype, a certificate or another report alone do not
complete the work. State unresolved physical regimes and the next concrete
experiment. Reuse concise entrypoints and existing tables, figures and evidence;
keep provenance proportional to changes instead of adding sprawling frameworks.

## Mechanics references

- [Abaqus Hertz contact relations](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-demanalysis.htm).
- [Current pad declaration](COMPLIANT_PAD_DECLARATION.md): the fourth-root estimate
  above is a direct derivation under its stated quadratic-gap/rigid-link assumptions.
- [Sun and Paulose, shell indentation](https://arxiv.org/abs/2103.06307):
  geometry and pressure affect shell response.
- [Willert, tangential contact histories](https://arxiv.org/abs/2104.02436):
  history-dependent tangential reconstruction under Hertz-Mindlin assumptions.
